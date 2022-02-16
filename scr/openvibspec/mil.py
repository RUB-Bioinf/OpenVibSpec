""" Pytorch implementation of attention-based multiple instance learning (MIL)
Configured for use with tiled bags of spectral data in OpenVibSpec.

Depending on the 'depth' of spectra you try to process you need to initialize the model accordingly.
We do not recommend using more than 16 wavenumbers for the spectral depth.

This script is designed to work with up to 4 GPUs to distribute computational load. You can specify the load distribution by using a custom "device_ordinals" list.
There are three pre-defined ordinal profiles, as follows:
 - device_ordinals_cpu - Not using the GPU. All computations run on the CPU.
 - device_ordinals_single_gpu - Using the first available GPU to run all computations.
 - device_ordinals_cluster_gpu - Using a 4 GPU setup to distribute load evenly among all 4.

Data to load, preprocessing of e.g. tissue into fitting tiles and bags to use for MIL needs to be changed by the user.
Accordingly, this file contains everything that is used to define the MIL model and train/validate/test it and the MAIN part when calling directly only serves as an exemplary starting point.

Author: @Joshua Butke & @Nils Förster
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scr.openvibspec.utils import mil_callbacks
from scr.openvibspec.utils import mil_metrics

# GLOBAL
#######
device_ordinals_cpu = None
device_ordinals_single_gpu = [0, 0, 0, 0]
device_ordinals_cluster_gpu = [0, 1, 2, 3]


# UTILS
#######


def save_checkpoint(state, is_best, savepath):
    """Save model and state stuff if a new best is achieved
    Used in fit function in main.
    """
    if is_best:
        print("--> Saving new best model")
        torch.save(state, savepath)


def load_checkpoint(loadpath, model, optim):
    """loads the model and its optimizer states from disk."""
    checkpoint = torch.load(loadpath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optim_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optim, epoch, loss


# MODEL
#######
class DeepAttentionMIL(nn.Module):
    def __init__(self, device, device_ordinals=device_ordinals_cpu, spectra=10, use_bias=False, use_gated=True,
                 use_adaptive=False):
        super().__init__()

        self.device = device
        self._device_ordinals = device_ordinals
        self.linear_nodes = 512
        self.attention_nodes = 128
        self.num_classes = 1
        self.lam = 2  # lambda hyperparameter for adaptive weighting

        self.input_dim = (int(spectra), 224, 224)  # single tile dimension

        self.use_bias = use_bias  # use bias per layer (bool)
        self.use_gated = use_gated  # use gated attention (bool)
        self.use_adaptive = use_adaptive  # use adaptive weighting (bool)

        self.feature_extractor_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_dim[0],
                out_channels=16,
                kernel_size=3,
                bias=self.use_bias,
            ),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
        ).to(self.get_device_ordinal(0))

        self.feature_extractor_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, bias=self.use_bias
            ),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, bias=self.use_bias
            ),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
        ).to(self.get_device_ordinal(1))

        self.feature_extractor_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias
            ),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=(2, 2),
                bias=self.use_bias,
            ),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, bias=self.use_bias
            ),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
        ).to(self.get_device_ordinal(2))

        size_after_conv = self._get_conv_output(self.input_dim)

        self.feature_extractor_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=size_after_conv,
                out_features=self.linear_nodes,
                bias=self.use_bias,
            ),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(
                in_features=self.linear_nodes,
                out_features=self.linear_nodes,
                bias=self.use_bias,
            ),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
        ).to(self.get_device_ordinal(3))  # bag of embeddings

        if not self.use_gated:
            self.attention = nn.Sequential(
                nn.Linear(self.linear_nodes, self.attention_nodes, bias=self.use_bias),
                nn.Tanh(),
                nn.Linear(self.attention_nodes, 1),
            ).to(self.get_device_ordinal(3))
            # two-layer NN that replaces the permutation invariant pooling operator( max or mean normally, which are pre-defined and non trainable) with an adaptive weighting attention mechanism

        elif self.use_gated:
            self.attention_V = nn.Sequential(
                nn.Linear(self.linear_nodes, self.attention_nodes), nn.Tanh()
            ).to(self.get_device_ordinal(3))

            self.attention_U = nn.Sequential(
                nn.Linear(self.linear_nodes, self.attention_nodes), nn.Sigmoid()
            ).to(self.get_device_ordinal(3))

            self.attention = nn.Linear(self.attention_nodes, 1).to(self.get_device_ordinal(3))

        self.classifier = nn.Sequential(
            nn.Linear(self.linear_nodes, self.num_classes), nn.Sigmoid()
        ).to(self.get_device_ordinal(3))

    def forward(self, x):
        """Forward NN pass, declaring the exact interplay of model components"""
        x = x.squeeze(
            0
        )  # compresses unnecessary dimensions eg. (1,batch,channel,x,y) -> (batch,channel,x,y)
        # transformation f_psi of instances in a bag
        hidden = self.feature_extractor_0(x)
        hidden = self.feature_extractor_1(hidden.to(self.get_device_ordinal(1)))
        hidden = self.feature_extractor_2(hidden.to(self.get_device_ordinal(2)))
        hidden = self.feature_extractor_3(hidden.to(self.get_device_ordinal(3)))  # N x linear_nodes

        # transformation sigma: attention-based MIL pooling
        if not self.use_gated:
            attention = self.attention(hidden)  # N x num_classes

        elif self.use_gated:
            attention_V = self.attention_V(hidden)
            attention_U = self.attention_U(hidden)
            attention = self.attention(attention_V * attention_U)

        attention = torch.transpose(attention, 1, 0)  # num_classes x N
        attention = F.softmax(attention, dim=1)  # softmax over all N

        if not self.use_adaptive:
            z = torch.mm(attention, hidden)  # num_classes x linear_nodes

        elif self.use_adaptive:
            # instance-level adaptive weighing attention [Li et al. MICCAI 2019]
            mean_attention = torch.mean(attention)
            thresh = nn.Threshold(
                mean_attention.item(), 0
            )  # set elements in the attention vector to zero if they are <= mean_attention of the cycle
            positive_attention = thresh(
                attention.squeeze(0)
            )  # vector of [1,n] to [n] and then threshold
            pseudo_positive = torch.where(
                positive_attention > 0,
                torch.transpose(hidden, 1, 0),
                torch.tensor([0.0], device=self.get_device_ordinal(3)),
            )  # select all elements of the hidden feature embeddings that have sufficient attention
            positive_attention = positive_attention.unsqueeze(0)  # reverse vector [n] to [1,n]

            negative_attention = torch.where(
                attention.squeeze(0) <= mean_attention,
                attention.squeeze(),
                torch.tensor([0.0], device=self.get_device_ordinal(3)),
            )  # attention vector with zeros if elements > mean_attention
            pseudo_negative = torch.where(
                negative_attention > 0,
                torch.transpose(hidden, 1, 0),
                torch.tensor([0.0], device=self.get_device_ordinal(3)),
            )  # select all elements of the hidden feature embeddings matching this new vector
            negative_attention = negative_attention.unsqueeze(0)

            x_mul_positive = torch.mm(
                positive_attention, torch.transpose(pseudo_positive, 1, 0)
            )  # pseudo positive instances N-N_in Matrix Mult.
            x_mul_negative = self.lam * torch.mm(
                negative_attention, torch.transpose(pseudo_negative, 1, 0)
            )  # pseudo negative instances N_in Matrix Mult modfied by lambda hyperparameter (increases weightdifferences between pos/neg)
            z = (
                    x_mul_positive + x_mul_negative
            )  # see formula 2 of Li et al. MICCAI 2019

        # transformation g_phi of pooled instance embeddings
        y_hat = self.classifier(z)
        y_hat_binarized = torch.ge(y_hat, 0.5).float()
        return y_hat, y_hat_binarized, attention

    def _get_conv_output(self, shape):
        """generate a single fictional input sample and do a forward pass over
        Conv layers to compute the input shape for the Flatten -> Linear layers input size
        """
        bs = 1
        test_input = torch.autograd.Variable(torch.rand(bs, *shape)).to(self.get_device_ordinal(0))
        output_features = self.feature_extractor_0(test_input)
        output_features = self.feature_extractor_1(output_features.to(self.get_device_ordinal(1)))
        output_features = self.feature_extractor_2(output_features.to(self.get_device_ordinal(2)))
        n_size = int(output_features.data.view(bs, -1).size(1))
        del test_input, output_features
        return n_size

    # DEVICE FUNCTIONS
    def get_device_ordinal(self, index: int) -> str:
        if self._device_ordinals is None:
            return 'cpu'

        if self.is_cpu():
            return 'cpu'

        return 'cuda:' + str(self._device_ordinals[index])

    def is_cpu(self) -> bool:
        return self.device.type == 'cpu'

    def get_device_ordinals(self) -> [int]:
        return self._device_ordinals.copy()

    # COMPUTATION METHODS
    def compute_loss(self, X, y):
        """otherwise known as loss_fn
        Takes a data input of X,y (batches or bags) computes a forward pass and the resulting error.
        """
        y = y.float()

        y_hat, y_hat_binarized, attention = self.forward(X)
        y_prob = torch.clamp(y_hat, min=1e-5, max=1.0 - 1e-5)

        loss_func = nn.BCELoss()
        loss = loss_func(y_hat, y)
        return loss, attention

    def compute_accuracy(self, X, y):
        """compute accuracy"""
        y = y.float()
        y = y.unsqueeze(dim=0)

        y_hat, y_hat_binarized, _ = self.forward(X)
        y_hat = y_hat.squeeze(dim=0)

        acc = mil_metrics.binary_accuracy(y_hat, y)
        return acc


# Training Routine
##################


@torch.no_grad()
def get_predictions(model, dataloader):
    """takes a trained model and validation or test dataloader
    and applies the model on the data producing predictions
    binary version
    """
    model.eval()

    all_y_hats = []
    all_preds = []
    all_true = []
    all_attention = []

    for batch_id, (data, label) in enumerate(dataloader):
        label = label.squeeze()
        bag_label = label[0]
        bag_label = bag_label.cpu()

        y_hat, preds, attention = model(data.to(model.get_device_ordinal(0)))
        y_hat = y_hat.squeeze(dim=0)  # for binary setting
        y_hat = y_hat.cpu()
        preds = preds.squeeze(dim=0)  # for binary setting
        preds = preds.cpu()

        all_y_hats.append(y_hat.numpy().item())
        all_preds.append(preds.numpy().item())
        all_true.append(bag_label.numpy().item())
        attention_scores = np.round(attention.cpu().data.numpy()[0], decimals=3)
        all_attention.append(attention_scores)

        print("Bag Label:" + str(bag_label))
        print("Predicted Label:" + str(preds.numpy().item()))
        print("attention scores (unique ones):")
        print(np.unique(attention_scores))
        # print(attention_scores)

        del data, bag_label, label

    return all_y_hats, all_preds, all_true


@torch.no_grad()
def evaluate(model, dataloader):
    """Evaluate model / validation operation
    Can be used for validation within fit as well as testing.
    """
    model.eval()
    test_losses = []
    test_acc = []
    result = {}

    for batch_id, (data, label) in enumerate(dataloader):
        label = label.squeeze()
        bag_label = label[0]
        data = data.to(model.get_device_ordinal(0))
        bag_label = bag_label.to(model.get_device_ordinal(3))

        loss, attention_weights = model.compute_loss(data, bag_label)
        test_losses.append(float(loss))
        acc = model.compute_accuracy(data, bag_label)
        test_acc.append(float(acc))

        del data, bag_label, label

    result["val_loss"] = sum(test_losses) / len(test_losses)
    result["val_acc"] = sum(test_acc) / len(test_acc)
    return result, attention_weights


def fit(model, optim, train_dl, validation_dl, model_savepath, callbacks: [mil_callbacks.BaseTorchCallback]):
    """Trains a model on the previously preprocessed train and val sets.
    Also calls evaluate in the validation phase of each epoch.
    """
    best_acc = 0
    history = []
    cancel_requested = False

    # Notifying callbacks
    for i in range(len(callbacks)):
        callback: mil_callbacks.BaseTorchCallback = callbacks[i]
        callback.on_training_start(model=model)

    for epoch in range(1, args.epochs + 1):
        if cancel_requested:
            break

        # TRAINING PHASE
        model.train()
        train_losses = []
        train_acc = []

        for batch_id, (data, label) in enumerate(train_dl):
            label = label.squeeze()
            bag_label = label[0]

            # Notifying Callbacks
            for i in range(len(callbacks)):
                callback: mil_callbacks.BaseTorchCallback = callbacks[i]
                callback.on_batch_start(model=model, batch_id=batch_id, data=data, label=bag_label)

            data = data.to(model.get_device_ordinal(0))
            bag_label = bag_label.to(model.get_device_ordinal(3))

            model.zero_grad()  # resets gradients

            loss, _ = model.compute_loss(data, bag_label)  # forward pass
            train_losses.append(float(loss))
            acc = model.compute_accuracy(data, bag_label)
            train_acc.append(float(acc))

            loss.backward()  # backward pass
            optim.step()  # update parameters
            del data, bag_label, label

        # VALIDATION PHASE
        result, _ = evaluate(model, validation_dl)  # returns a results dict for metrics
        result["train_loss"] = sum(train_losses) / len(train_losses)
        result["train_acc"] = sum(train_acc) / len(train_acc)
        history.append(result)

        print(
            "Epoch [{}] : Train Loss {:.4f}, Train Acc {:.4f}, Val Loss {:.4f}, Val Acc {:.4f}".format(
                epoch,
                result["train_loss"],
                result["train_acc"],
                result["val_loss"],
                result["val_acc"],
            )
        )
        # Save best model / checkpointing stuff

        is_best = bool(result["val_acc"] >= best_acc)
        best_acc = max(result["val_acc"], best_acc)
        state = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        save_checkpoint(state, is_best, model_savepath)

        # Notifying Callbacks
        for i in range(len(callbacks)):
            callback: mil_callbacks.BaseTorchCallback = callbacks[i]
            callback.on_epoch_finished(model=model, epoch=epoch, epoch_result=result, history=history)
            cancel_requested = cancel_requested or callback.is_cancel_requested()

        if cancel_requested:
            print('Model was canceled before reaching all epochs.')

    # Notifying callbacks that training has finished
    for i in range(len(callbacks)):
        callback: mil_callbacks.BaseTorchCallback = callbacks[i]
        callback.on_training_finished(model=model, was_canceled=cancel_requested, history=history)

    return history


#############
#############

def get_device(gpu_preferred: bool = True):
    ''' Pick GPU if available, else run on CPU.
    Returns the corresponding device.
    '''
    if gpu_preferred:
        torch.cuda.init()

        if torch.cuda.is_available():
            print('Running on GPU.')
            return torch.device('cuda')
        else:
            print('  =================')
            print('Wanted to run on GPU but it is not available!!')
            print('  =================')

    print('Running on CPU.')
    return torch.device('cpu')


#############
#############

# MAIN
######

if __name__ == "__main__":

    # Init
    device = get_device()
    print(device)

    device_ordinals = device_ordinals_single_gpu
    metric_savepath = "path/to/your/metrics_folder"
    model_savepath = "path/to/your/model_folder"
    #############

    # Data
    datapath = "your/data/path"  # CHANGE THIS LINE! below is an example for np array data as most commonly used with OpenVibSpec
    # data = np.load(datapath, allow_pickle=True)
    # X = data["X"]
    # y = data["y"]

    # Preprocess your data
    print("Preprocessing data into bags and labels...")

    # resulting in training_ds and validation_ds
    training_ds, validation_ds = [], []
    #############

    # Model
    model = DeepAttentionMIL(spectra=10, device=device, device_ordinals=device_ordinals)
    loader_kwargs = {}
    if torch.cuda.is_available():
        loader_kwargs = {"num_workers": 4, "pin_memory": True}

    optim = torch.optim.Adadelta(
        model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0
    )
    print("\nSuccessfully build and compiled the chosen model!")
    #############

    # Data Generators
    train_dl = torch.utils.data.DataLoader(
        training_ds, batch_size=1, shuffle=True, **loader_kwargs
    )
    validation_dl = torch.utils.data.DataLoader(
        validation_ds, batch_size=1, shuffle=False, **loader_kwargs
    )

    # Setting up callbacks
    callbacks = []
    callbacks.append(mil_callbacks.UnreasonableLossCallback(loss_max=40.0))
    callbacks.append(mil_callbacks.EarlyStopping(epoch_threshold=30))

    # Training
    ##########
    history = fit(model, optim, train_dl, validation_dl, model_savepath, callbacks=callbacks)
    # Get best saved model from this run
    model, optim, _, _ = load_checkpoint(model_savepath, model, optim)

    print("Computing and plotting confusion matrix...")
    y_hats, y_pred, y_true = get_predictions(model, validation_dl)
    mil_metrics.plot_conf_matrix(
        y_true,
        y_pred,
        metric_savepath,
        target_names=["Inflammation", "Cancer"],
        normalize=False,
    )

    print("Computing and plotting binary ROC-Curve")
    fpr, tpr, _ = mil_metrics.binary_roc_curve(y_true, y_hats)
    mil_metrics.plot_binary_roc_curve(fpr, tpr, metric_savepath)

# END OF FILE
#############
