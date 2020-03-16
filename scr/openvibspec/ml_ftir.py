from __future__ import absolute_import
###########################################
# hier alle ein-/auslese routinen aus Matlab, R
# und was noch kommen koennte
#
#
#
###########################################
import h5py
import numpy as np 
#import scipy.io as sio 
import sklearn
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
# python 2.7 from sklearn.cross_validation import train_test_split 
import sys
import openvibspec.models
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
###########################################
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, GaussianNoise
#from keras.models import Model
#from keras.optimizers import RMSprop, Adam, SGD
#from keras.models import model_from_json
#from keras.callbacks import ModelCheckpoint
###########################################




MODELPATH = Path('openvibspec/models').absolute()

def randomforest_train(x,y,n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False,n_jobs=2, save_file_path=str()):
	"""
	an dieser stelle if einbauen um None save model einzubauen

	"""
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.datasets import make_classification
	
	clf = RandomForestClassifier(n_jobs=2, random_state=0)
	
	clf.fit(x, y)
	#filename = 'finalized_model.sav'
	
	filename = save_file_path
	
	pickle.dump(clf, open(filename, 'wb'))
	
	return x,y ,clf


def randomforest_load_eval(x,clf):
	preds = clf.predict(x)
	
	return preds, clf

	return x

def kmeans(x, c=4):
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=c)
	kmeans.fit(x)
	y = kmeans.predict(x)
	return y

def hca(x):
	return x

def pca(x,pc):
	from sklearn.decomposition import PCA
	
	pca = PCA(n_components=pc)
	
	pca.fit(x)
	
	p = pca.transform(x)
	
	return p



def pca_all(x,pc):
	"""
	Alles was an Daten durch sklearn ausgegeben wird auch mglich machen
	- eventuell sogar plots einbauen und ausgeben
	"""
	from sklearn.decomposition import PCA
	
	pca = PCA(n_components=pc)
	
	pca.fit(x)
	
	p = pca.transform(x)
	
	#	sv = pca.singular_values_(x) funktioniert nicht-> anpassen!
	
	vr = pca.explained_variance_ratio_
	
	print("Explained Variance based on N PCs =",vr)
	
	cov = pca.get_covariance()
	
	it  = pca.inverse_transform(p)	
	
	scores = pca.score_samples(x)
	
	#return p, sv, vr, cov, it, sc
	return p, vr, cov, it, scores


	
def plot_pca(p):
	import matplotlib.pyplot as plt 
	
	plt.style.use('ggplot')
	
	plt.scatter(p[:,1],p[:,2], color=['r']);
	
	plt.scatter(p[:,0],p[:,1], color=['b']);
	
	plt.show()
	
	plt.plot(np.cumsum(vr))
	
	plt.xlabel('number of components')
	
	plt.ylabel('cumulative explained variance');
	
	plt.show()
	return

def plot_specs_by_class(x,classes_array,class2plot):
	class2plot = int()
	dat = x[np.where(g == class2plot)]
	plt.plot(dat.T)
	plt.show()
	return(dat)
####################################################################################################
####################################################################################################
# PRETRAINED DEEP NEURAL NETWORKS
####################################################################################################


#class DeepLearn(object):
class DeepLearn:
	
	
	"""	
	HERE YOU CAN FIND ALL PRETRAINED DEEP NEURAL NETWORKS FOR THE PURPOSE OF CLASSIFICATION AND RMIES-CORRECTION OF 
	FTIR-SPECTROSCOPIC MEASUREMENTS OF TISSUE

	THE USED DATASETS FOR TRAINING WERE ALL BASED ON FFPE (formaldehyde-fixed paraffin-embedded) TISSUE FROM COLON
	
	BIOMAX CO1002b/CO722

	FOR FURTHER INFORMATION PLEASE REFER TO Raulf et al.,2019

	"""
	

	def net(self,x,classify=False, miecorr=False, predict=False,train=False ,show=False):
		import keras
		from keras.optimizers import RMSprop, Adam, SGD
		from keras.models import model_from_json
		from keras.callbacks import ModelCheckpoint
		#import tensorflow as tf
		#import tensorflow.compat.v1 as tf
		#tf.disable_v2_behavior()
		"""
		TODO: UPDATE TO TENSORFLOW VERSION 2 
		"""

		####################################################################################################
		#	DETERMINE WICH MODEL PARAMETERS YOUN WNAT TO USE
		#		CLASSIFY == TRUE GIVES THE MODEL TRAINED TO CLASSIFY ALL CELLUAR COMPONENTS BASED ON SPECTRA
		#					BETWEEN 950-1800 WVN 
		#	
		#		MIECORR == TRUE GIVES THE CORRESPONDING NEURAL NETWORK FOR PERFORMING EFFICIENT RMIE-CORRECTION
		#				   ON FFPE-BASED TISSUE SPECTRA
		#	
		####################################################################################################

		if classify == True:
			
			json_file = open(os.path.join(str(MODELPATH)+'/model_weights_classification.json'), 'r')
			loaded_model_json = json_file.read()
			loaded_model = model_from_json(loaded_model_json)
			
			if show == True:
				print(loaded_model.summary())
			
			loaded_model.load_weights(os.path.join(str(MODELPATH)+"/model_weights_classification.best.hdf5"))
			
			print("Loaded model from disk")
			
			model = loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

			return loaded_model.predict(x), load_model

		if miecorr == True:
			####################################################################################################
			#	THIS MODEL NEEDS THE FIRST 909 WVN. RANGE FROM 950-23XX WVN
			#	
			#	
			#	
			####################################################################################################x

			#json_file = open(os.path.join(MODELPATH, 'model_weights_regression.json'), 'r')
			json_file = open(os.path.join(str(MODELPATH)+'/model_weights_classification.json'), 'r')
			loaded_model_json = json_file.read()
			loaded_model = model_from_json(loaded_model_json)
			
			if show == True:
				print(loaded_model.summary())
			
			loaded_model.load_weights(os.path.join(str(MODELPATH)+"/model_weights_regression.best.hdf5"))
			
			print("Loaded model from disk")
			
			loaded_model.compile(loss='mean_squared_error', optimizer='adam')

			
			
		
			return loaded_model.predict(x), load_model







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



