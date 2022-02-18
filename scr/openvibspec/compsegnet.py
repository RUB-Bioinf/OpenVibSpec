"""
author: Dajana Mueller
date: February, 2022
script: CompSegNet by David Schuhmacher
"""

import sys, cv2, os, time

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')


#Set seeds
int_seed = 0
os.environ['PYTHONHASHSEED'] = str(int_seed)
tf.random.set_seed(int_seed)
np.random.seed(int_seed)


# Set gpu settings
os.environ["VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), " Physical GPUs, ", len(logical_gpus), " Logical GPUs")
    except RuntimeError as e:
        print(e)


def helper():
    print("\n\033[1m_____HOW TO TRAIN COMPSEGNET: EXAMPLE DATA______\033[0m\n")

    import string, random

    def create_random_data(a,b,c,d,e):
        for i in range(e):
            a.append(np.random.rand(64,64,427))
            b.append(np.reshape(np.float32(np.random.randint(0,2,64*64)),(64,64)))
            c.append([np.random.randint(2)])
            d.append(''.join(random.choice(string.ascii_uppercase) for _ in range(4)))
        return a,b,c,d

    n_t = 12
    n_v = 4
    train, mask_t, y_train, fnames_t = create_random_data([], [], [], [],n_t)
    vali, mask_v, y_vali, fnames_v = create_random_data([], [], [], [],n_v)


    print("Training/Validation data: ", type(train), "\nExample of elements inside list:", train[0].shape, "\nBinary mask of data:", type(mask_t), "with elements of shape", mask_t[0].shape,"\nLabels:", type(y_train), "example", y_vali,"\nFile Names: ", type(fnames_t), "example",fnames_v)
    print("\033[1m________________________________________________\033[0m\n")

def make_data_divisable(x,y,mask,fnames,batch_size):

    if len(x)%batch_size!=0:
        x = x[0:batch_size*(len(x)//batch_size)]
        y = y[0:batch_size*(len(x)//batch_size)]
        mask = mask[0:batch_size*(len(x)//batch_size)]
        fnames = fnames[0:batch_size*(len(x)//batch_size)]
    return x, y, mask, fnames

def calc_class_weights(y: list):
    _,label_counts = np.unique(np.array(y),return_counts=True)
    class_weights = [label_counts[0]/np.sum(label_counts),label_counts[1]/np.sum(label_counts)]
    return class_weights

def data_gen(data,label,lab,fnames,epochs=1, dtype="float32"):
    def groupe_data_gen():
        tracker = np.arange(0,len(data),1)

        for i in range(0,epochs):
            np.random.shuffle(tracker)
            for j in tracker:
                yield data[j],label[j],lab[j],fnames[j]

    return groupe_data_gen

class lr_scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,initial_lr,cooldown,factor,initial_epoch):
        self.initial_lr = initial_lr
        self.cooldown = cooldown
        self.factor = factor
        self.initial_epoch = initial_epoch
        self.step = 0 

    def __call__(self,step):
        self.step = step + self.initial_epoch 
        return self.initial_lr * (self.factor**(self.step//self.cooldown))

    def get_config(self):
        return {"LR":self.initial_lr * (self.factor**(self.step//self.cooldown))}

def TF_U_net(int_height_data, int_width_data, int_dim_data,dropout_rate=0.2):

    kernel_size = 3
    pool_size = 2
    output_channels = 1

    down_stack = [downsample(64,kernel_size,0,"D1"),
                  downsample(128,kernel_size,dropout_rate,"D2"),
                  downsample(256,kernel_size,dropout_rate,"D3"),
                  downsample(512,kernel_size,dropout_rate,"D4"),
                  downsample(1024,kernel_size,dropout_rate,"D5")]

    up_stack1 = [upsample1(512,pool_size,pool_size,"T2"),
                 upsample1(256,pool_size,pool_size,"T3"),
                 upsample1(128,pool_size,pool_size,"T4"),
                 upsample1(64,pool_size,pool_size,"T5")]

    up_stack2 = [upsample2(512,kernel_size,dropout_rate,"U2"),
                upsample2(256,kernel_size,dropout_rate,"U3"),
                upsample2(128,kernel_size,dropout_rate,"U4"),
                upsample2(64,kernel_size,dropout_rate,"U5")]


    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[int_height_data,int_width_data,int_dim_data])
    x = inputs
    
    skips = []
    pooling = tf.keras.layers.MaxPool2D(pool_size,padding='valid')
    cropping = []

    iterator = 0
    for down in down_stack:
        x = down(x)
        skips.append(x)
        if iterator<4:
            x = pooling(x)
            iterator = iterator+1
        
    skips = reversed(skips[:-1])
    iterator = 0
    for up1,up2,skip in zip(up_stack1,up_stack2,skips):
        x = up1(x)
        iterator = iterator+1
        x = concat([x,skip])
        x = up2(x)
    
    output = tf.keras.Sequential(name="Output")
    output.add(tf.keras.layers.Conv2D(output_channels,1,strides=1,kernel_initializer=tf.keras.initializers.GlorotNormal()))
    output.add(tf.keras.layers.BatchNormalization())
    output.add(tf.keras.layers.Activation('sigmoid'))

    x = output(x)

    return tf.keras.Model(inputs=inputs,outputs=x)

def downsample(filters,kernel_size,dropout_rate,name,activation='relu',apply_batch_norm=True):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2D(filters,kernel_size,strides=1,padding='same',kernel_initializer=tf.keras.initializers.GlorotNormal()))
    result.add(tf.keras.layers.Dropout(dropout_rate))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.Activation(activation))
    result.add(tf.keras.layers.Conv2D(filters,kernel_size,strides=1,padding='same',kernel_initializer=tf.keras.initializers.GlorotNormal()))
    result.add(tf.keras.layers.Dropout(dropout_rate))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.Activation(activation))
    return result

def upsample1(filters,kernel_size,strides,name):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2DTranspose(filters,kernel_size,strides=strides,padding='valid'))
    result.add(tf.keras.layers.BatchNormalization())
    return result

def upsample2(filters,kernel_size,dropout_rate,name,activation='relu'):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2D(filters,kernel_size,strides=1,padding='same'))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.Activation(activation))
    result.add(tf.keras.layers.Conv2D(filters,kernel_size,strides=1,padding='same'))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.Activation(activation))
    return result

def Tf_bool_vector_prediction(tf_activation_map,tf_mask):

    tf_activation     = tf.math.multiply(tf_mask,tf_activation_map)
    tf_sum_activation = tf.math.reduce_sum(tf_activation, [1,2])
    tf_sum_mask       = tf.math.reduce_sum(tf_mask, [1,2])
    tf_activation     = tf.math.divide(tf_sum_activation, tf_sum_mask)

    return tf_activation

def TF_Clipping_function_experimental(tf_float_value,tf_alpha_mask):

    tf_double_value     = tf_float_value
    tf_shifted_value1   = tf_double_value*(1/tf_alpha_mask[:,0])
    part1               = tf.clip_by_value(tf_shifted_value1, 0, 1)
 
    tf_grad             = tf_alpha_mask[:,0] + tf_alpha_mask[:,1]
    tf_shifted_value2   = -((tf_double_value - tf_grad)*(1/tf.math.abs(1-tf_grad)))
    part2               = tf.clip_by_value(tf_shifted_value2, -1, 0)

    tf_result = part1+part2

    return tf_result, part1, part2

def TF_Soft_clipping_pooling_layer_experimental(tf_activation_map,tf_alpha_mask,tf_mask):
    
    tf_unknown_segmentation_net = Tf_bool_vector_prediction(tf_activation_map,tf_mask)
    tf_output, part1, part2     = TF_Clipping_function_experimental(tf_unknown_segmentation_net,tf_alpha_mask)

    return tf_output, tf_unknown_segmentation_net, part1, part2

def TF_binary_cross_entropy(output_map1,output_map2,y_,tf_class_weights):

    output_map      = output_map1 + output_map2
    tiny            = tf.constant(0.000001, dtype=tf.float32)
    bin_true        = tf.constant(1.0,dtype=tf.float32)
    temp_output_map = tf.cast(output_map,dtype=tf.float32)
    #temp_output_map1 = tf.cast(output_map1,dtype=tf.float32)
    temp_y_         = tf.cast(y_,dtype=tf.float32)
    tf_loss1         = -tf.math.reduce_mean( temp_y_*tf_class_weights[0]*tf.math.log(tf.clip_by_value(temp_output_map, tiny, 1-tiny)))
    tf_loss2         = -tf.math.reduce_mean((bin_true - temp_y_)*tf_class_weights[1]*tf.math.log(bin_true - tf.clip_by_value(temp_output_map, tiny, 1-tiny)))
    tf_loss = tf_loss1 + tf_loss2
    return tf_loss

def TF_cross_entropy_out_of_cluster(tf_activationmap,tf_region_of_interest,y_):

    tiny = tf.constant(0.000001, dtype=tf.float32) 
    bin_true = tf.constant(1.0, dtype=tf.float32)
    
    loss1 = -tf.math.reduce_mean((bin_true-tf_region_of_interest)*tf.math.log((bin_true - tf.clip_by_value(tf_activationmap, tiny, 1-tiny))), [1,2])
    loss2 = -tf.math.reduce_mean(tf.math.log((bin_true - tf.clip_by_value(tf_activationmap, tiny, 1-tiny))), [1,2])

    tf_loss = tf.reduce_mean(loss1*(y_) + loss2*(1-y_))

    return tf_loss

def loss(model,x,y,tf_region_of_interest,tf_alpha_mask,tf_class_weights,training,fileid=[],out_dir=[],folder=[],regionID=[],save=False):

    pred = model(x,training=training)

    tf_lambda1 = 1
    tf_prediction_pooling_target, tf_sum_activationmap_target, tf_prediction_pooling_target1, tf_prediction_pooling_target2  = TF_Soft_clipping_pooling_layer_experimental(pred[:,:,:,0],tf_alpha_mask, tf_region_of_interest)
    tf_loss_target_segmentation = TF_binary_cross_entropy(tf_prediction_pooling_target1, tf_prediction_pooling_target2,y[:,0],tf_class_weights)
    tf_loss_outside_target_region = TF_cross_entropy_out_of_cluster(pred[:,:,:,0],tf_region_of_interest, y[:,0])
    tf_loss = tf_loss_target_segmentation + (tf_loss_outside_target_region*tf_lambda1)


    y_true = tf.convert_to_tensor(y, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(tf_sum_activationmap_target, dtype=tf.float32)


    if not training and save:
        save_pred(y,tf_region_of_interest,pred,fileid,out_dir,folder,regionID, tf_sum_activationmap_target,tf_prediction_pooling_target)

    return y_true,y_pred, pred, tf_loss, tf_prediction_pooling_target, tf_sum_activationmap_target, tf_loss_target_segmentation, tf_loss_outside_target_region

def grad(model,x,y,z,tf_alpha_mask,tf_class_weights,epoch_auc=None, vali_auc = None):

    with tf.GradientTape() as tape:
        y_true,y_pred,pred,loss_value, tf_prediction_pooling_target, tf_sum_activationmap_target, tf_loss_target_segmentation, tf_loss_outside_target_region = loss(model,x,y,z,tf_alpha_mask,tf_class_weights,training=True)
    return y_true,y_pred,loss_value, tape.gradient(loss_value,model.trainable_variables), tf_prediction_pooling_target, tf_sum_activationmap_target, tf_loss_target_segmentation, tf_loss_outside_target_region

def save_pred(y_true,mask,pred,fileid,out_dir,folder,regionID,tf_sum_activationmap_target,tf_prediction_pooling_target):

    if not os.path.exists(out_dir+"/vali/"+folder):
        os.mkdir(out_dir+"/vali/"+folder)


    for i in range(0,len(y_true)):
        cv2.imwrite(out_dir+"/vali/"+folder+"/"+str(fileid[i].decode("utf-8"))+"_predict_"+"label"+str(int(y_true[i]))+".png",pred.numpy()[i,:,:,0]*255)
        cv2.imwrite(out_dir+"/vali/"+folder+"/"+str(fileid[i].decode("utf-8"))+"_true_"+"label"+str(int(y_true[i][0]))+".png",mask[i,:,:]*255)
        np.save(out_dir+"/vali/"+folder+"/"+str(fileid[i].decode("utf-8"))+"_predictval_"+"label"+str(int(y_true[i][0]))+".npy", [y_true[i][0],tf_sum_activationmap_target.numpy()[i],tf_prediction_pooling_target.numpy()[i]])




def train_CSN(x_train: list,
                    mask_train: list,
                    y_train: list,
                    fnames_train: list,
                    x_vali: list,
                    mask_vali: list,
                    y_vali: list,
                    fnames_vali: list,
                    out_dir: str,
                    epochs: int = 300, 
                    initial_epoch: int = 0,
                    batch_size: int = 20,
                    learning_rate: float = 0.005,
                    alpha: float = 0.05,
                    beta: float = 0.8,
                    momentum_RMSprop: float = 0,
                    lr_scheduler_epochs: int = 30,
                    lr_scheduler_factor: float = 0.9,
                    dropout_rate_Unet: float = 0.2,
                    path_model_restore: str = None,
                    restore: bool = False,
                    save_vali_images: bool = True):

    """
    :param x_train: Training data (lost of ndarrays)
    :param mask_train: Mask of Training data (list of binary mask)
    :param y_train: Binary training labels (list of list labels)
    :param fnames_train: Name of training patches (list of str)
    :param x_vali: Validation data (lost of ndarrays)
    :param mask_vali: Mask of validation data (list of binary mask)
    :param y_vali: Binary validation labels (list of list labels)
    :param fnames_vali: Name of validation patches (list of str)
    :param out_dir: Output dir (str)
    :param epochs: Number of epochs (int)
    :param initial_epoch: Number of epoch to start training from (int)
    :param batch_size: Batch size (int)
    :param learning_rate: Learning rate (float)
    :param alpha: Lower boundary of activation that should be present to yield class 1 (float)
    :param beta: Second Boundary of activation: alpha + beta == Upper boundary (float)
    :param momentum_RMSprop: Momentum of optimizer (float),
    :param lr_scheduler_epochs: Number of epochs before learning rate will be reduced (int)
    :param lr_scheduler_factor: Factor to multiply learning rate with (float)
    :param dropout_rate_Unet: Dropout rate of Unet (float)
    :param path_model_restore: Model path that shall be restored when training is continued (str)
    :param restore: Bool to continue training 
    :param save_vali_images: Save validation predictions every epoch (bool)

    """


    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    if not os.path.exists(out_dir+"/vali/"):
        os.mkdir(out_dir+"/vali/")


    verbose = 1
    float_decay_a = learning_rate / epochs
    tf_alpha_mask = tf.Variable(np.array([alpha,beta]).reshape((1,2)),dtype=tf.float32,trainable=False)


    x_train, y_train, mask_train, fnames_train = make_data_divisable(x_train, y_train, mask_train, fnames_train, batch_size)
    x_vali, y_vali, mask_vali, fnames_vali = make_data_divisable(x_vali, y_vali, mask_vali, fnames_vali, batch_size)


    # Calculate class weights
    tf_class_weights = tf.Variable(np.ones((2))*0.5,dtype=tf.float32,trainable=False)
    class_weights_train = calc_class_weights(y_train)
    class_weights_vali  = calc_class_weights(y_vali)

    print("Class Weights Train: ", class_weights_train)
    print("Class Weights Vali: ", class_weights_vali)


    train_dataset = tf.data.Dataset.from_generator(data_gen(x_train,y_train,mask_train,fnames_train,epochs),(tf.float32,tf.float32,tf.float32,tf.string)).batch(batch_size)#.prefetch(batch_size)
    vali_dataset = tf.data.Dataset.from_generator(data_gen(x_vali,y_vali,mask_vali,fnames_vali,epochs),(tf.float32,tf.float32,tf.float32,tf.string)).batch(batch_size)#.prefetch(batch_size)



    # Build model + restore weights if restore == True
    model = TF_U_net(x_train[0].shape[0],x_train[0].shape[1],x_train[0].shape[2],dropout_rate_Unet)

    if restore:
        print("Restore models ...")
        model.load_weights(path_model_restore)

    model.summary()

    optimizer = tf.keras.optimizers.RMSprop(lr_scheduler(learning_rate,lr_scheduler_epochs*(len(x_train)//batch_size),lr_scheduler_factor, initial_epoch*(len(x_train)//batch_size)),float_decay_a, momentum_RMSprop)



    # Initialize variables, list {...} for training
    train_loss_result,\
    train_loss1_result,\
    train_loss2_result,\
    train_auc_results, \
    vali_loss_result, \
    vali_loss1_result,\
    vali_loss2_result, \
    vali_auc_results = [],[],[],[],[],[],[],[]

    number_of_batches_train = 0
    number_of_batches_vali = 0

    best_train_loss = 1000


    file_writer_train = tf.summary.create_file_writer(out_dir + "/train")
    file_writer_vali = tf.summary.create_file_writer(out_dir+"/validation")
    file_writer_lr = tf.summary.create_file_writer(out_dir + "/lr")



    # Start Training
    for i in range(initial_epoch,epochs):

        epoch_loss_avg, epoch_loss1_avg, epoch_loss2_avg = tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        epoch_auc, vali_auc = tf.keras.metrics.AUC(), tf.keras.metrics.AUC()
        vali_loss_avg, vali_loss1_avg, vali_loss2_avg = tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), tf.keras.metrics.Mean()


        batch_counter=0
        start_time = time.time()
        tf.keras.backend.set_value(tf_class_weights,class_weights_train)


        print("Epoch:",i,"/",epochs)

        for x,y,z,fileid in train_dataset.take(len(x_train)//batch_size).as_numpy_iterator():
            if i==0 or i==initial_epoch:
                number_of_batches_train += 1

            y_true,y_pred,loss_value,grads,prediction_pooling, sum_activationmap, loss_target_segm, loss_outside_region = grad(model,x,y,z,tf_alpha_mask,tf_class_weights,epoch_auc, vali_auc)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            epoch_loss1_avg.update_state(loss_target_segm)
            epoch_loss2_avg.update_state(loss_outside_region)
            epoch_auc.update_state(np.reshape(y_true,(y_true.shape[0])), y_pred)

            batch_counter+=1


        checkpoint_dir = out_dir + "/checkpoints/cp-"+ str(i) +".ckpt"
        end_time=time.time()

        train_loss_result.append(epoch_loss_avg.result())
        train_loss1_result.append(epoch_loss1_avg.result())
        train_loss2_result.append(epoch_loss2_avg.result())
        train_auc_results.append(epoch_auc.result())


        with file_writer_train.as_default():
            tf.summary.scalar("Loss",train_loss_result[-1].numpy(),step=i)
            tf.summary.scalar("Loss target",train_loss1_result[-1].numpy(),step=i)
            tf.summary.scalar("Loss outside",train_loss2_result[-1].numpy(),step=i)
            tf.summary.scalar("AUC",train_auc_results[-1].numpy(),step=i)
        with file_writer_lr.as_default():
            tf.summary.scalar("LearningRate", optimizer.get_config()['learning_rate']['config']['LR'].numpy(),step=i)

        if train_loss_result[-1].numpy()<best_train_loss:
            best_train_loss = train_loss_result[-1].numpy()


        model.save_weights(checkpoint_dir)

        np.save(out_dir + "/Train_loss.npy", np.array(train_loss_result))
        np.save(out_dir + "/Train_AUC.npy", np.array(train_auc_results))


        print("Train: ",batch_counter,"/",number_of_batches_train,"Time:",np.around(end_time-start_time,2),"Time per Step:",np.around((end_time-start_time)/number_of_batches_train,2),"    TrainLoss: ",train_loss_result[-1].numpy(), "    TrainAUC: ",train_auc_results[-1].numpy())


        batch_counter = 0
        start_time = time.time()
        tf.keras.backend.set_value(tf_class_weights, class_weights_vali)


        for x,y,z,fileid in vali_dataset.take(len(x_vali)//batch_size).as_numpy_iterator():
            if i==0 or i==initial_epoch:
                number_of_batches_vali +=1


            y_true,y_pred,pred,loss_value, prediction_pooling, sum_activationmap, loss_target_segm, loss_outside_region = loss(model,x,y,z,tf_alpha_mask,tf_class_weights,training=False,fileid=fileid,out_dir=out_dir,folder="vali"+str(i),save=save_vali_images)
            vali_loss_avg.update_state(loss_value)
            vali_loss1_avg.update_state(loss_target_segm)
            vali_loss2_avg.update_state(loss_outside_region)
            vali_auc.update_state(np.reshape(y_true,(y_true.shape[0])), y_pred)

            batch_counter+=1


        end_time = time.time()
        vali_loss_result.append(vali_loss_avg.result())
        vali_loss1_result.append(vali_loss1_avg.result())
        vali_loss2_result.append(vali_loss2_avg.result())
        vali_auc_results.append(vali_auc.result())


        with file_writer_vali.as_default():
            tf.summary.scalar("Loss",vali_loss_result[-1].numpy(),i)
            tf.summary.scalar("Loss target",vali_loss1_result[-1].numpy(),i)
            tf.summary.scalar("Loss outside",vali_loss2_result[-1].numpy(),i)
            tf.summary.scalar("AUC",vali_auc_results[-1].numpy(),i)


        np.save(out_dir + "/Vali_loss.npy", np.array(vali_loss_result))
        np.save(out_dir + "/Vali_AUC.npy", np.array(vali_auc_results))


        print("Vali: ",batch_counter,"/",number_of_batches_vali,"Time:",np.around(end_time-start_time,2),"Time per Step:",np.around((end_time-start_time)/number_of_batches_vali,2),"    ValiLoss: ", vali_loss_result[-1].numpy(), "    Vali AUC: ", vali_auc_results[-1].numpy())


    model.save(out_dir + "/")




def predict_CSN(x_test: list,
                    mask_test: list,
                    y_test: list,
                    fnames_test: list,
                    out_dir: str,
                    epochs_to_test: list,
                    batch_size: int = 20,
                    alpha: float = 0.05,
                    beta: float = 0.8,
                    dropout_rate_Unet: float = 0.2):

    restore = True

    if not os.path.exists(out_dir+"test_tiles"):
        os.mkdir(out_dir+"test_tiles")

    tf_alpha_mask = tf.Variable(np.array([alpha,beta]).reshape((1,2)),dtype=tf.float32,trainable=False)

    x_test, y_test, mask_test, fnames_test = make_data_divisable(x_test, y_test, mask_test, fnames_test, batch_size)

    test_dataset = tf.data.Dataset.from_generator(data_gen(x_test, y_test, mask_test, fnames_test),(tf.float32,tf.float32,tf.float32,tf.string)).batch(batch_size).prefetch(batch_size)

    tf_class_weights = tf.Variable(np.ones((2))*0.5,dtype=tf.float32,trainable=False)
    class_weights_test = calc_class_weights(y_test)
    tf.keras.backend.set_value(tf_class_weights, class_weights_test)


    int_number_iterations = int(np.ceil((len(y_test)/batch_size)))



    for e in range(len(epochs_to_test)):

        if not os.path.exists(out_dir+"test_tiles/epoch"+str(epochs_to_test[e])):
            os.mkdir(out_dir+"test_tiles/epoch"+str(epochs_to_test[e]))

        # Prepare model
        model = TF_U_net(x_test[0].shape[0],x_test[0].shape[1],x_test[0].shape[2],dropout_rate_Unet)
        model_to_load = out_dir + "checkpoints/cp-" +str(epochs_to_test[e]) + ".ckpt"
        if restore:
            print("Restoring model of epoch:   ", str(epochs_to_test[e]))
            model.load_weights(model_to_load)
            print("Done!")


        list_y, list_pred = [],[]

        print("Predict testing data: .. ")
        for x,y,z,fileid in test_dataset.take(len(x_test)//batch_size).as_numpy_iterator():
            y_true,_,pred, _, tf_prediction_pooling_target, tf_sum_activationmap_target, _, _ = loss(model,x,y,z,tf_alpha_mask,tf_class_weights,training=False,save=False)

            for i in range(0,batch_size):

                cv2.imwrite(out_dir+"test_tiles/epoch"+str(epochs_to_test[e])+"/"+str(fileid[i].decode("utf-8"))+"_predict_label"+str(int(y_true[i][0]))+".png",pred.numpy()[i,:,:,0]*255)
                cv2.imwrite(out_dir+"test_tiles/epoch"+str(epochs_to_test[e])+"/"+str(fileid[i].decode("utf-8"))+"_true_label"+str(int(y_true[i][0]))+".png",z[i,:,:]*255)
                np.save(out_dir+"test_tiles/epoch"+str(epochs_to_test[e])+"/"+str(fileid[i].decode("utf-8"))+"_predictval_label"+str(int(y_true[i][0]))+".npy", [y_true[i][0],tf_sum_activationmap_target.numpy()[i],tf_prediction_pooling_target.numpy()[i]])

