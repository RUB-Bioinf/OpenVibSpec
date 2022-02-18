"""
author: Dajana Mueller
date: January, 2022
"""

import datetime
import os
import random
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

warnings.filterwarnings("ignore")

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Activation,
)

# ----------------------- Eager execution -------------
# tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()

# ------------------------ Seed ------------------------
seed_value = 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ["PYTHONHASHSEED"] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


class AE(Model):
    """
    Stacked autoencoder (SAE) which trains each hidden layer seperate
    and uses the code layer as the input for the next autoencoder.
    Initialized with glorot_uniform and trained with a contractive
    loss.
    nb_name: = None: builds SAE with names as continious numbers e.g.: enc_0, enc_1
                     = str : builds SAE with names as given string (Warning: given string
                             has to differ for training of each SAE, e.g. AE(str(test_name)+str[i]), ....)
    layers: List of dict of hidden layers: [ {'n_nodes': 256 }, { 'n_nodes': 128 }]
    input_shape : training_data.shape[1]

    """

    def __init__(self, nb_name=None, layers=None, input_shape=(427), **hyperparameters):
        super(Model, self).__init__(**hyperparameters)
        super(AE, self).__init__()  # Necessary in tf >=2.3
        # Configure base (super) class
        # Model.__init__(self, hyperparameters, **hyperparameters)
        self.initializer = tf.compat.v1.keras.initializers.glorot_uniform(seed=0)
        self._layers = layers
        self._input_shape = input_shape
        self._target_layer = None
        self._nb_name = nb_name
        inputs = Input(input_shape, name="input_0")
        encoder = self.encoder(inputs, layers=layers)
        outputs = self.decoder(encoder, layers=layers)
        self._model = Model(inputs, outputs)
        self._enc = Model(inputs, encoder)

    def encoder(self, x, **metaparameters):
        layers = metaparameters["layers"]
        for _ in range(len(layers)):
            if not self._nb_name:
                _name = "_" + str(_)
            else:
                _name = str(self._nb_name)
            # _name = 'enc' + _name
            n_nodes = layers[_]["n_nodes"]
            x = Dense(
                n_nodes, name=("enc" + str(_name)), kernel_initializer=self.initializer
            )(x)
            x = BatchNormalization(name=("bn" + str(_name)))(x)
            x = Activation(activation="sigmoid")(x)
        self._target_layer = "enc" + str(_name)
        return x

    def decoder(self, x, **metaparameters):
        layers = metaparameters["layers"]
        for _ in range(len(layers) - 2, -1, -1):
            _name = "dec" + str(_)
            n_nodes = layers[_]["n_nodes"]
            x = Dense(n_nodes, name=_name, kernel_initializer=self.initializer)(x)
            x = BatchNormalization()(x)
            x = Activation(activation="sigmoid")(x)
        outputs = Dense(self._input_shape, name="dec_last", activation="sigmoid")(x)
        return outputs

    def contractive_loss(self, y_pred, y_true):
        lam = 1e-3
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        W = K.variable(
            value=(self._model.get_layer(self._target_layer)).get_weights()[0]
        )  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = self._model.get_layer(self._target_layer).output
        dh = h * (1 - h)  # N_batch x N_hidden
        contractive = lam * K.sum(
            dh ** 2 * K.sum(W ** 2, axis=1), axis=1
        )  # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        return mse + contractive


def save_model(model, out_dir, signature="cae_model"):
    model_json = model.to_json()
    json_fp = out_dir + "/" + signature + ".json"
    with open(json_fp, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    wt_fp = out_dir + "/" + signature + ".h5"
    model.save_weights(wt_fp)
    print("Saved model to disk")
    return


def train_SCAE(
        X_train: np.ndarray,
        X_val: np.ndarray,
        out_dir: str,
        hidden_layer: list = [100, 50, 25],
        num_epochs: int = 200,
        int_epoch_start: int = 0,
        batch_size: int = 50,
        learning_rate: float = 0.003,
        early_stop_epochs: int = 200,
        l2_normalize_data: bool = False,
):
    """
    Main code to set parameters and build + train a stacked
    autoencoder. Shape of data has to be x*y,z e.g.:(4000,427)
    and can be L2 normalized if wanted.

    Stacked Auotencoder:
    For each hidden layer a stacked autoencoder is build with just
    one hidden layer, the weights will be saved and the training data
    will be predicted with the trained first SAE to serve as the
    input for the next one.
    SAE is trained with SGD and a contractive loss function (mse
    with an additive regularization term).

    Further comments included for debugging purpose.

    :param X_train: Input training data in shape (X*Y,Z)
    :param X_val: Input validation data in shape (X*Y,Z)
    :param out_dir: The directory to save weights and models
    :param hidden_layer: Number of hidden layers (Optional)
    :param num_epochs: Number of epochs to train each stacked contractive autoencoder (Optional)
    :param int_epoch_start: Start training from this epoch (Optional)
    :param batch_size: Batch size for training (Optional)
    :param learning_rate: Learning rate for training (Optional)
    :param early_stop_epochs: Number of epochs for the Early Stopping Callback (Optional)
    :param l2_normalize_data: If true, input data will be l2 normalized

    :type X_train: np.ndarray
    :type X_val: np.ndarray
    :type out_dir: str
    :type hidden_layer: list of ints
    :type num_epochs: int
    :type int_epoch_start: int
    :type batch_size: int
    :type learning_rate: float
    :type early_stop_epochs: int
    :type l2_normalize_data: bool

    :returns: the trained encoder

    """

    if not isinstance(X_train, np.ndarray) or len(X_train.shape) != 2:
        raise ValueError(
            "Passed array (X_traini) is not of the right shape. E.g. (X ,num_of_spectra)"
        )
    if not isinstance(X_val, np.ndarray) or len(X_val.shape) != 2:
        raise ValueError(
            "Passed array (X_val) is not of the right shape. E.g. (X ,num_of_spectra)"
        )

    hidden_layers = []
    for i in hidden_layer:
        hidden_layers.append({"n_nodes": i})

    int_number_input_features = X_train.shape[1]

    if not os.path.isdir(out_dir + "/logs/"):
        os.mkdir(out_dir + "/logs/")

    print(
        "Train data shape: %s\nValidation data shape: %s\n"
        % (str(X_train.shape), str(X_val.shape))
    )
    print(
        "_" * 30,
        "\nParameters:\nSaving directory:",
        out_dir,
        "\nHidden layers:",
        hidden_layer,
        "\nNumber of Epochs:",
        num_epochs,
        "\nBatch size:",
        batch_size,
        "\nLearning rate:",
        learning_rate,
        "\nEarly stopping after %s epochs" % (early_stop_epochs),
        "\nL2 Data Normalization:",
        l2_normalize_data,
        "\n",
        "_" * 30,
        "\n",
    )

    # Normalize data if needed
    if l2_normalize_data:
        print("Scale input vectors individually to L2 norm...")
        import sklearn.preprocessing as skp

        X_train = skp.normalize(
            X_train, norm="l2", axis=1, copy=True, return_norm=False
        )
        X_val = skp.normalize(X_val, norm="l2", axis=1, copy=True, return_norm=False)

    # Iterate over every hidden layer, train model with one hidden layer each
    dict_weights = {}
    input_x = None

    for hl in range(0, len(hidden_layers)):
        if hl < 1:
            input_x = X_train
            val_x = X_val
        else:
            input_x = input_x
            val_x = val_x

        nb_name = "_" + str(hl)
        ae = AE(nb_name, hidden_layers[hl: hl + 1], input_x.shape[1])
        autoencoder = ae._model
        encoder = ae._enc

        print("Encoder summary:\n", encoder.summary())
        print("Autoencoder summary:\n", autoencoder.summary())

        opt = tf.keras.optimizers.SGD(lr=learning_rate)
        autoencoder.compile(optimizer=opt, loss=ae.contractive_loss)

        log_dir = out_dir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = (
                out_dir + "/model_" + str(hl) + "_{epoch:04d}_{val_loss:.8f}.h5"
        )

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=early_stop_epochs),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=70, verbose=1, mode="auto",
                                                 min_delta=0.0000001, cooldown=0, min_lr=0),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, monitor='val_loss', mode='min',
                                               save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]

        history = autoencoder.fit(
            input_x,
            input_x,
            epochs=num_epochs,
            batch_size=batch_size,
            initial_epoch=int_epoch_start,
            shuffle=True,
            validation_data=(val_x, val_x),
            callbacks=my_callbacks,
            verbose=2,
        )

        for l in encoder.layers:
            if "enc" in l.name or "bn" in l.name:
                _id = l.name
                dict_weights[_id] = l.get_weights()
                # print("Layer", l.name, " Shape",l.get_weights()[0].shape)
                # print(l,l.get_weights()[0][0][0:5])

        input_x = encoder.predict(input_x)
        val_x = encoder.predict(val_x)

    # Build on AE Model and transfer the learned weights from the single SAE
    nb_name = None
    ae_new = AE(nb_name, hidden_layers, X_train.shape[1])
    enc = ae_new._enc
    enc.summary()

    # Set all the weights in the final encoder model
    for l in enc.layers:
        if l.name in dict_weights.keys():
            wb = dict_weights[l.name]
            l.set_weights(wb)

    # Debugging
    # for l in enc.layers:
    # 	if 'enc' in l.name:
    # 		print("Layer",l.name, "   Weights",l.get_weights()[0][0][0])
    # 	if 'bn' in l.name:
    # 		print("Layer",l.name, "   Weights",l.get_weights()[0][0])

    # Save and predict
    save_model(enc, out_dir)
    enc.save(out_dir + "/model_encoder")

    return enc


def test_example1():
    # How to build from scratch
    data = np.random.uniform(low=0.001, high=0.0099, size=(1000, 427))
    hidden_layers = [
        {"n_nodes": 256},
        {"n_nodes": 128},
        {"n_nodes": 64},
        {"n_nodes": 32},
        {"n_nodes": 16},
    ]

    ae = AE(layers=hidden_layers, input_shape=data.shape[1])
    autoencoder = ae._model
    encoder = ae._enc
    print(autoencoder.summary(), encoder.summary())
    autoencoder.load_weights("PATH TO MODEL/model_0504_0.00148203.ckpt")
    encoded_test = encoder.predict(data)
