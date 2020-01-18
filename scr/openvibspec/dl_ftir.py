from __future__ import absolute_import
###########################################
# hier alle ein-/auslese routinen aus Matlab, R
# und was noch kommen koennte
#
#
#
###########################################
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, GaussianNoise
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD
import time
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
#from keras.models import model_from_json
t0 = time.time()
import numpy as np
import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)
import h5py
###########################################






#						with np.load('/bph/puredata1/bioinfdata/user/butjos/master/data/C_data_fullVerticalSlice_column4_noNormalization_3D.npz') as data:
#						    trainx = data['trainx']
#						    trainy = data['trainy']
#						
#						
#						
#						z,y,x = trainx.shape
#						
#						trainx = trainx.reshape(-1,x*y)
#						trainy = trainy.reshape(-1,x*y)
#						
#						
#						trainx = trainx.T
#						trainy = trainy.T
#						
#						#from sklearn.model_selection import train_test_split
#						
#						#X, testx, y, testy = train_test_split(trainx, trainy, test_size = 0.33, random_state =  59)
#						from sklearn.preprocessing import normalize
#						
#						x_ = normalize(trainx, axis=1, norm='l2')
#						
#						
#						
#						from sklearn.cross_validation import train_test_split
#						#y = np.argmax(label_401, axis=1)
#						#x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.4,random_state=654)# hist: 42,34,54,347,654
#						x_train, x_test, y_train, y_test = train_test_split(x_, trainy, test_size=0.4,random_state=71683)# hist: 42,34,54,347,654
#						x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5,random_state=71683)# hist: 42,34,54,347,654
#						import numpy as np
#						noise = np.random.normal(0, 1, x_train.shape)
#						
#						
#						#np.random.rand(3,2)
#						#signal = x_train + noise/9600
#						#signal = x_train + noise/9900
#						signal = x_train + noise/15000
#						#signal = x_train + noise/990
#						
#						
#						


def load_pretrainedweights():
	"""
	Pretrained weights from a CSAE on Biomax CO722 Data

	For further information see: Raulf et al.,
	Deep representation learning for domain adaptatable 
	classification of infrared spectral imaging data, Bioinformatics, 2019
	"""
	w1 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_w1d_500E_F450WVN.npy")
	b1 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_b1d_500E_F450WVN.npy")
	w2 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_w2d_500E_F450WVN.npy")
	b2 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_b2d_500E_F450WVN.npy")
	w3 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_w3d_500E_F450WVN.npy")
	b3 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_b3d_500E_F450WVN.npy")
	w4 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_w4d_500E_F450WVN.npy")
	b4 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_b4d_500E_F450WVN.npy")
	w5 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_w5d_500E_F450WVN.npy")
	b5 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_b5d_500E_F450WVN.npy")
	w6 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_w6d_500E_F450WVN.npy")
	b6 = np.load("params/ftir/ffpe/parameters_200718SAE32f_constracLossAE_Pretrain4ApproxMieCorr_L2Norm_pretain_b6d_500E_F450WVN.npy")
	return w1,b1,w2,b2,w3,b3,w4,b4,w5,b5,w6,b6












def model_d(input_shape):
    
    inp = Input(shape=input_shape)


    x = Dense(units=909, input_dim=909,activation='relu', name="layer_1")(inp)
    x = Dropout(0.1)(x, training=True)
    #x = Activation("relu")
    x = Dense(units=500,activation='relu', name="layer_2")(x)
    x = Dropout(0.1)(x, training=True)
    #x = Activation("relu")
    x = Dense(units=300,activation='relu', name="layer_3")(x)
    x = Dropout(0.1)(x, training=True)
    #x = Activation("relu")
    x = Dense(units=200,activation='relu', name="layer_4")(x)
    x = Dropout(0.1)(x, training=True)
    #x = Activation("relu")
    x = Dense(units=100,activation='relu', name="layer_5")(x)
    x = Dropout(0.1)(x, training=True)
    #x = Activation("relu")
    #model.add(Dense(units=100, name="layer_6"))
    #model.add(Activation("relu"))
    #model.add(Dense(units=200, name="layer_7"))
    #model.add(Activation("relu"))
    #model.add(Dense(units=300, name="layer_8"))
    #model.add(Activation("relu"))
    #model.add(Dense(units=500, name="layer_9"))
    #model.add(Activation("relu"))
    #model.add(Dense(units=909, name="layer_10"))
    #model.add(Activation("relu"))
    x = Dense(units=909, kernel_initializer="glorot_uniform", name="output_layer")(x) #applying no activation defaults to linear activation a(x)=x
    
    return Model(inp, x, name='model_d')




# prepare the model
model = model_d
model = model(input_shape=trainx.shape[1:] )



w1,b1,w2,b2,w3,b3,w4,b4,w5,b5,w6,b6 = load_pretrainedweights()

model.layers[1].set_weights((w1,b1)) #set dense1
model.layers[3].set_weights((w2,b2)) #set dense2
model.layers[5].set_weights((w3,b3)) #set dense3
model.layers[7].set_weights((w4,b4)) #set dense4
model.layers[9].set_weights((w5,b5)) #set dense5







model.compile(loss='mean_squared_error',
              #optimizer='rmsprop'),
              optimizer='adam')
              #metrics=['accuracy'])

# geht nicht akltuell filepath="model_cnn_cars_combi_weights_randomState26134_26d_imbal_onlyLG_true.best.hdf5"
#   filepath="model_cnn_cars_combi_weights_randomState26134_all42c.best.hdf5"
filepath="model_ptMLP_mcDrop_relu_5.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min') # monitor='loss' beim naechsten mal
callbacks_list = [checkpoint]

#model.fit(trX, trY,
#          #class_weight='auto',
#          class_weight=d_class_weights,
#          nb_epoch=1000,
#          batch_size=32,callbacks=callbacks_list, verbose=1,  validation_data=(teX, teY))
#score = model.evaluate(teX, teY, batch_size=32)







model.fit(signal, y_train, 
          batch_size=64, 
          epochs=100,
          shuffle=True, 
          callbacks=callbacks_list, 
          verbose=1
          #validation_data=(x_test, y_test)
          #callbacks=[callbacks]
         )  








#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


# serialize model to JSON
model_json = model.to_json()
# geht nicht aktuell with open("model_cnn_cars_combi_randomState26134_26d_imbal_onlyLG_true.json", "w") as json_file:
with open("model_ptMLP_mcDrop_relu_5.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# geht nicht akltuellmodel.save_weights("model_cnn_cars_combi_randomState26134_26d_imbal_onlyLG_true.h5")
model.save_weights("model_model_ptMLP_mcDrop_relu_5.h5")
print("Saved model to disk")




#--------------------------------------------------
#--------------------------------------------------
# LOAD AND SAVE MODELS SKLEARN
#--------------------------------------------------
#--------------------------------------------------
def save_model(model, name='model', timestamp=None):
	from  sklearn.externals import joblib
	import datetime
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
	
	if timestamp == True:
		joblib.dump(model, name+st)
	else:
		joblib.dump(model, name)
	
	return

def load_model(model):
	from  sklearn.externals import joblib
	m = joblib.load(model)
	return(m)


