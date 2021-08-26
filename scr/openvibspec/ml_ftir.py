from __future__ import absolute_import
###########################################
# ML and AI Procedures for FTIR/Raman Spectroscopy
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
import matplotlib.pyplot as plt
# python 2.7 from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
#from tqdm import tqdm

plt.style.use('ggplot')
###########################################
import sys
import openvibspec.models

from pathlib import Path
MODELPATH = Path('openvibspec/models').absolute()

def prepare_dat2train(x,y, testsize=0.2, random=0, normalising='l2' ):
	"""
FUNCTION PREPARING DATA FOR THE USE IN A RANDOM FOREST CLASSIFIER:

	Preparation of the data includes splitting into training and test-set and normalising the data with the sklearn 'StandardScaler'!


	Parameters
	----------
	x : numpy array (2D)
			classes per point (as int) with 2 dimensions:  x*y,spectra = shape(), e.g. output of kmeans or similiar clsutering approach
	y : numpy array with int (1D)
			classes:  x*y = shape()
	test_size : float
			value for the amount of used test-data. This value has to be below 1, given example of '0.2' 
			indicates that 20% of the data is used for testing, while 80% is used for the purpose of fitting parameters to the model aka train the model
	random_state : int
			seed for the 'random_state'

	
	Returns
	-------s
	X_train : umpy array (2D) of shape x*y,spectra = shape
			selected data points of size equal (1 - random_state)
	
	y_train : numpy array (1D)
			Corresponding classes to 'X_train'
	
	X_test  : numpy array (2D) of shape x*y,spectra = shape
			selected data points of size equal random_state

	y_test  : numpy array (1D)
			Corresponding classes to 'X_test'
	


	"""


	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn import preprocessing
	#X_normalized = preprocessing.normalize(X, norm='l2')	

	X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(preprocessing.normalize(x, norm=normalising)), y, test_size=float(testsize), random_state=int(random))
	

	#sc = StandardScaler()
	
	#X_train = sc.fit_transform(X_train)
	
	#X_test = sc.transform(X_test)

	return X_train, X_test, y_train, y_test


def randomforest_train(x,y,
						trees=20,
						jobs=2, 
						random=0,
						out=1,
						save_file_path=str()):
							#n_estimators=20,
							#n_samples=250, 
							#n_features=4,
						   #n_informative=2, 
						   #n_redundant=0,
						   #random_state=0, 
						   #shuffle=False,
						   #n_jobs=2, 
						   
	"""
	Parameters
	----------
	x : numpy array (2D)
			X_train from prepare_dat2train() with 2 dimensions:  x*y,spectra = shape(), e.g. output of kmeans or similiar clsutering approach
	y : numpy array (1D)
			wavenumbers:  x*y = shape()

	trees : int
		Number of Trees used in the construction of the tree ensemble
	
	jobs : int
			The number of jobs to run in parallel
	
	random: int
			Controls both the randomness of the bootstrapping of the samples used when building trees

	save_file_path: str() as filename

	
	Returns
	-------
	X_train : umpy array (2D) of shape x*y,spectra = shape
			selected data points of size equal (1 - random_state)
	
	y_train : numpy array (1D)
			Corresponding classes to 'X_train'
	
	X_test  : numpy array (2D) of shape x*y,spectra = shape
			selected data points of size equal random_state

	y_test  : numpy array (1D)
			Corresponding classes to 'X_test'




	"""
	
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.datasets import make_classification
	
	clf = RandomForestClassifier(n_estimators=int(trees),n_jobs=int(jobs), random_state=int(random), verbose=out)
	
	rf = clf.fit(x, y)


	filename = save_file_path
	from joblib import dump, load
	dump(rf, filename)
	
	#return x,y ,clf, yhat
	return rf


def randomforest_load_eval(x,y,rf, norm=True, report=True, normalising='l2'):
	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
	from joblib import load
	from sklearn import preprocessing
	
	clf = load(str(rf)) 
	
	if norm == True:

		preds = clf.predict(preprocessing.normalize(x, norm=str(normalising)))
	else:
		preds = clf.predict(x)

	if report ==True:
		print("CLASSIFICATION REPORT")
		print(classification_report(y,preds))
		print("CONFUSION MATRIX")
		print(confusion_matrix(y,preds))
		print("ACCURACY SCORE:")
		print(accuracy_score(y,preds))
		return preds
	else:
	
		return preds

	#return x

def dec_bound_plot(X_train, X_test,y_train, y_test, forest):
	"""
	This function provides the decision boundary plot.
	In case of a multidimensional Input Classifier you need to specify the certain features.
	Now every feature except the first 2 features are filtered

	TODO: Specify Filter-List

	"""
	import matplotlib.pyplot as plt
	from mlxtend.plotting import plot_decision_regions
	#from sklearn.decomposition import PCA

	
	
	
 
	X_combined = np.vstack((X_train, X_test))
	y_combined = np.hstack((y_train, y_test))

	#pca = PCA(n_components = 2)
	#X_train2 = pca.fit_transform(X_combined)
	
	 
	#
	# plot_decision_regions function takes "forest" as classifier
	#
	feature_values = {i:1 for i in range(2, X_combined.shape[1])}  
	feature_width = {i:1 for i in range(2, X_combined.shape[1])}  

	fig, ax = plt.subplots(figsize=(7, 7))
	plot_decision_regions(X_combined, y_combined, clf=forest,
						#feature_index=[0,2],
						filler_feature_values=feature_values,
						filler_feature_ranges=feature_width,
						res=0.02, legend=2, ax=ax)
	#plt.xlabel('petal length [cm]')
	#plt.ylabel('petal width [cm]')
	#plt.legend(loc='upper left')
	#plt.tight_layout()
	plt.show()
	return

#------------------------------------------------------
#
#	Features from RF
#
#------------------------------------------------------

#	TODO: install
#def get_rf_features(rf, wvn, X_test=None, y_test=None, method=1):
#	"""
#
#
#	"""
#	if method==1:
#		plt.barh(wvn, rf.feature_importances_)
#		plt.show()
#
#	if method==2:
#		perm_importance = permutation_importance(rf, X_test, y_test)
#		plt.barh(wvn, perm_importance.importances_mean)
#		plt.show()
#
#	
#	if method==3:
#		explainer = shap.TreeExplainer(rf)
#		shap_values = explainer.shap_values(X_test)
#		shap.summary_plot(shap_values, X_test)#, plot_type="bar")
#
#	return
#






	

def kmeans(x, c=4, jobs=2, out=15):
	"""
	Wrapper of the scikit learn Kmeans implementation

	Kmeans: standard clustering approach to vector quantization.
	https://projecteuclid.org/euclid.bsmsp/1200512992

	Parameters
	----------
	x : numpy array (2D)
			spectroscopic data with 2 dimensions:  x*y,spectra = shape(), e.g. output of kmeans or similiar clsutering approach
	c : int 
			number of putative centroids

	n_jobs : int
			The number of jobs to run in parallel
	verbose: int
			provides information about the progress 

	Returns
	-------
	y : numpy array (1D) of shape x*y
			selected data points with cluster affiliation / Corresponding classes


	"""


	from sklearn.cluster import KMeans

	kmeans = KMeans(n_clusters=c,n_jobs=jobs,verbose=out)

	kmeans.fit(np.nan_to_num(x))
	
	y = kmeans.predict(np.nan_to_num(x))
	
	return y

def hca(x):
	return x

def pca(x,pc):
	from sklearn.decomposition import PCA
	
	pca = PCA(n_components=pc)
	
	pca.fit(np.nan_to_num(x))
	
	p = pca.transform(np.nan_to_num(x))
	
	return p



def pca_all(x,pc):
	"""
	
	"""
	from sklearn.decomposition import PCA
	
	pca = PCA(n_components=pc)
	
	pca.fit(np.nan_to_num(x))
	
	p = pca.transform(np.nan_to_num(x))
	
	
	vr = pca.explained_variance_ratio_
	
	print("Explained Variance based on N PCs =",vr)
	
	cov = pca.get_covariance()
	
	it  = pca.inverse_transform(p)	
	
	scores = pca.score_samples(x)
	
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


class DeepLearn:
	
	"""
	Deep learning based procedures for the use in spectroscopy.

	Here you can find pretrained deep neural networks, which can be used for classification / RMieS-Correction, or further training.
	To ensure a smooth use of your data, the spectroscopic specifications of the used training data are shown below:
		
		Attributes of the raw data:

			- Spectroscopic data recording was performed on a Agilent Cary 620/670 FTIR Imaging System with 128x128 pixel MCT (Mercury Cadmium Telluride) and FPA for whole slide.
			- Data recording with 128 scans, results in a wavenumber range of 950 to 3700 cm^1.
			- With a spectral resolution of ca. 4 cm^1 this resulted in 1428 datapoints on the z-axis.
			- The pixel has a edge length of 5.65µm.
			- Per detector field this results in a FOV of ca. 715x715 µm^2.

		Data sources:
			Tissue was formaldehyde-fixed paraffin-embedded (FFPE) from human colon.
			- https://www.biomax.us/CO1002b
			- https://www.biomax.us/tissue-arrays/Colon/CO722
		
		You can find further information on data collection in: 

			[1] https://pubs.rsc.org/en/content/articlelanding/fd/2016/c5fd00157a#!divAbstract
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

	Components of the class are the following methods:

	DeepLearn.net()

		This method returns the basic structure of the used deep neural networks in form of a graph. 
		For further use, this structure was bound to the python interfaces of TensorFlow and Keras to allow a permanent integration in most modern workflows.
		It is currently divided into two main classes. First, the spectral classification and second the RMieS-correction of FTIR spectral data, using a fast deep learning algorithm.
		

		Specifications of the used training data:

			We used complete uncorrected FTIR data in the range of the fingerprint region between a wavenumber of 950 to 1800 cm^1.

			The groundtruth was based on the segmentation of the used random forest from [1]. 
			These in turn were created from a multi-step process of pre-segmentation of RMieS-corrected spectra and pathologist annotation. 

			With regard to the learning behaviour of the deep neuronal networks, it could be shown that no new classifier has to be built but 
			that the existing networks in transfer learning can be used for a variety of applications,  while the false positive number could be significantly reduced. [2]
			

			The data from groundtruth was processed under the following conditions:
				
				- Agilent Resolution Pro Software.
				- Fourier Transformation using Merz phase correction.
				- Blackman-Harris-4-term apodization and zero filling of 2.





		Specifications for own use:

			The spectral data must be available as 2d-numpy array which is structured as follows:

				x_data = x_axis*y_axis, z_axis 

				It is important for the application to observe the data points on the z-axis

				The classification ( dl.net(x_data,classify=True) ) of the individual RMieS-uncorrected spectra (semantic segmentation) is carried 
				out on the first 450 wavenumbers between 950 and 1800 cm^1.

				The correction ( dl.net(x_data, miecorr=True) ) of the raw data is done on the first 909 wavenumbers between 950 and 2300 cm^1.




		Examples:

			import openvibspec.ml_ftir as ovml

			dl = ovml.DeepLearn()
			
			x_pred, model = dl.net(x_data[:,:450],classify=True)

			x_corr, model = dl.net(x_data[:,:909], miecorr=True)

		Args:
			x_data(numpy array):
			
			classify=False(str): if True it uses the entered data (x_data) to predict previously learned 19 classes on uncorrected FTIR spectra of human colon tissue
			
			miecorr=False(str):  if True it uses the entered data (x_data) to predict the regression of the RMieS-Correction Function based on Bassan

		References:
			[2] Classification of (RMieS) uncorrected FTIR spectra with deep neural networks.
			https://academic.oup.com/bioinformatics/article-abstract/36/1/287/5521621
			
			[3] Deep neural networks for the correction of RMie Scattering in FTIR data.
			https://arxiv.org/abs/2002.07681	
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

	DeepLearn.transfer()

		The transfer function is based on using the data representations discovered by the existing networks for faster learning on new data. 
		For example, the networks trained on ffpe can be used to create classification networks for other tissues and their laboratory preparation with significantly less data. 
		For further informations regarding the theoretical part of this procedure, please see reference [2].

		Besides the spectral data a groundtruth as label is needed for the transfer learning.

		Models and weights are automatically saved in the working directory in *h5 and *json format using following naming convention:


			model_ptMLP_MieReg_%d-%m-%Y_%I-%M-%S_%p

		



		Examples:
			import openvibspec.ml_ftir as ovml

			dl = ovml.DeepLearn()

			dl.transfer(x_data[:5,:909],y_data, batch=10, train_epochs=10, miecorr=True, trainable=False)

			dl.transfer(x_data[:5,:909],x_data_corrected[:5,:909], batch=10, train_epochs=10, miecorr=True, trainable=False)
		
		Args:

			x_data(numpy array): 2D array shape(x_axis*y_axis, z_axis)  

			y_data(numpy array): label vector with classes assigned as numbers from 1 to n 

			batch(int): number of examples per batch

			train_epochs(int): number of iterations per training

			add_l(list of int()): possible list for adding layers

			classify=True(str): classification modus
			
			miecorr=True(str): regresssion modus

			trainable=False(str): if trainable=True: allows the adjustment of the already loaded weights from the pretrained networks 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	DeepLearn.load_and_predict()

		This function allows to load and use the trained network which was saved under DeepLearn.transfer()
	

		Examples:

				import openvibspec.ml_ftir as ovml
	
				dl = ovml.DeepLearn()
				
				a =  dl.load_and_predict(x_new_data[:,:450],'model_ptMLP_class_DATE_TIME')

		Args:

			x_new_data(numpy array): 2D array shape(x_axis*y_axis, z_axis)  
			
			model_ptMLP_class_DATE_TIME(str): model_ptMLP_MieReg_* or model_ptMLP_class_*

			
	"""
		

	def net(self,x,classify=False, miecorr=False, predict=False,train=False ,show=False):
		import keras
		from keras.layers import Input, Dense
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

			if x.shape[1] != 450:
			
				raise ValueError('This is a classification problem: Your spectral data needs 450 datapoints in WVN range of 950-1800 1/cm')
			
			json_file = open(os.path.join(str(MODELPATH)+'/model_weights_classification.json'), 'r')

			loaded_model_json = json_file.read()

			loaded_model = model_from_json(loaded_model_json)
			
			if show == True:
				print(loaded_model.summary())
			
			loaded_model.load_weights(os.path.join(str(MODELPATH)+"/model_weights_classification.best.hdf5"))
			
			print("Loaded model from disk")
			
			model = loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

			from sklearn.preprocessing import normalize
			

			trX = normalize(x, axis=1, norm='l2')

			return loaded_model.predict(trX), load_model

		if miecorr == True:

			if x.shape[1] != 909:
			
				raise ValueError('This is a regression problem: Your spectral data needs 909 datapoints in WVN range of 950-2300 1/cm')
			####################################################################################################
			#	THIS MODEL NEEDS THE FIRST 909 WVN. RANGE FROM 950-2300 WVN 1/cm
			#	
			#	
			#	
			####################################################################################################x

			json_file = open(os.path.join(str(MODELPATH)+'/model_weights_regression.json'), 'r')
			
			loaded_model_json = json_file.read()

			loaded_model = model_from_json(loaded_model_json)
			
			if show == True:
				print(loaded_model.summary())
			
			loaded_model.load_weights(os.path.join(str(MODELPATH)+"/model_weights_regression.best.hdf5"))
			
			print("Loaded model from disk")
			
			loaded_model.compile(loss='mean_squared_error', optimizer='adam')

			from sklearn.preprocessing import normalize
			
			trX = normalize(x, axis=1, norm='l2')
			return loaded_model.predict(trX), load_model

	
	def transfer(self,x, y, batch,train_epochs,add_l=[], classify=False, miecorr=False, trainable=False ):
		import keras
		from keras.models import Model
		from keras.optimizers import RMSprop, Adam, SGD
		from keras.models import model_from_json
		from keras.callbacks import ModelCheckpoint
		from keras.models import Sequential
		from datetime import datetime
		from sklearn.preprocessing import normalize
			

		"""
		ALL PARTS OF THE TRANSFER-LEARNING NETWORKS ON FTIR SPECTROSCOPIC DATA

		"""
		trX = normalize(x, axis=1, norm='l2')
		def onehot(y):
			import keras
			from keras.utils import np_utils

			c = np.max(y) + 1
			
			y1hot = np_utils.to_categorical(y, num_classes=c)
			
			return(y1hot)

		def add_layer():
			from keras.utils import np_utils
			from keras.layers import Input, Dense 
			from keras.models import Model
			from keras import models

			yoh = onehot(y)
			sm = int(yoh.shape[1])
			
			print("training on",sm,"classes")
			json_file = open(os.path.join(str(MODELPATH)+'/model_weights_classification.json'), 'r')
			
			loaded_model_json = json_file.read()
			
			loaded_model = model_from_json(loaded_model_json)
			
			loaded_model.load_weights(os.path.join(str(MODELPATH)+"/model_weights_classification.best.hdf5"))
			
			
			
			if trainable == False:
				for layer in loaded_model.layers:
					layer.trainable = False
			else: 
				for layer in loaded_model.layers:
					layer.trainable = True
			
			
			
			if not add_l:
				preds = Dense(sm, name='newlast', activation='softmax')(loaded_model.layers[-3].output)

				model = Model(inputs=loaded_model.input, outputs=preds)

				model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

				history = model.fit(trX,yoh,batch_size=batch,epochs=train_epochs )

				print(model.summary())

			if add_l:
				
				
				def add_2_model(add_l):
					
					base = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-3].output)
					
					model = Sequential()
					model.add(base)
					model.add(Dense(add_l[0], input_dim=450,activation='relu'))

					for layer_size in add_l[1:]:
						model.add(Dense(layer_size,activation='relu'))
					

					model.add(Dense(sm,activation='softmax'))

					return model

				model = add_2_model(add_l)
				
				model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
				
				history = model.fit(trX,yoh,batch_size=batch,epochs=train_epochs )
				
				print(model.summary())


			dtstr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
			
			model_json = model.to_json()
			
			with open("model_ptMLP_class_"+dtstr+".json", "w") as json_file:
				json_file.write(model_json)

			model.save_weights("model_model_ptMLP_class_"+dtstr+".h5")

			print("Saved model to disk to","model_model_ptMLP_class_"+dtstr+".json")
			print("and weights to")
			print("Saved model to disk to","model_model_ptMLP_class_"+dtstr+".h5")






			###########################PLOTTING##########################
			history_dict = history.history
			
			history_dict.keys()
			
			a = np.array(history_dict['acc'])
			
			print(a.shape)
			
			l = np.array(history_dict['loss'])
			
			e = range(1, len(a) +1)
			
			plt.plot(e, a, 'bo',color='red', label='Acc Training')
			
			plt.plot(e, l, 'b', label='Loss Training')
			
			plt.xlabel('Epochs')
			
			plt.legend()
			
			plt.savefig('model.pdf')
			
			
			return(model, history_dict)
		


		def simple_val_of_data(x,y):
			from sklearn.model_selection import train_test_split
			from random import randrange
			from sklearn.preprocessing import normalize

			trX = normalize(x, axis=1, norm='l2')
			
			seed = randrange(999)
			
			print('used random seed was', seed)
			
			x_train, x_test, y_train, y_test = train_test_split(trX, y, test_size=0.4,random_state=seed)
			
			return x_train, x_test, y_train, y_test

		def train_layer():
			from keras.utils import np_utils
			from keras.layers import Input, Dense 
			from keras.models import Model
			from keras import models
			
			sm = int(y.shape[1])
			
			
			json_filer = open(os.path.join(str(MODELPATH)+'/model_weights_regression.json'), 'r')
			
			loaded_model_jsonr = json_filer.read()
			
			loaded_modelr = model_from_json(loaded_model_jsonr)
			
			loaded_modelr.load_weights(os.path.join(str(MODELPATH)+"/model_weights_regression.best.hdf5"))
			
			
			
			if trainable == False:
				for layer in loaded_modelr.layers:
					layer.trainable = False
			else: 
				for layer in loaded_modelr.layers:
					layer.trainable = True
			
			loaded_modelr.compile(loss='mean_squared_error', optimizer='adam')
			
			history = loaded_modelr.fit(x,y, batch_size=batch, epochs=train_epochs )

			dtstr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
			
			print(loaded_modelr.summary())
			
			model_json = loaded_modelr.to_json()
			with open("model_ptMLP_MieReg_"+dtstr+".json", "w") as json_file:
				json_file.write(model_json)
			
			loaded_modelr.save_weights("model_model_ptMLP_MieReg_"+dtstr+".h5")
			
			print("Saved model to disk to","model_model_ptMLP_MieReg_"+dtstr+".json")
			print("and weights to")
			print("Saved model to disk to","model_model_ptMLP_MieReg_"+dtstr+".h5")
			
			return

		if classify == True:
			if x.shape[1] != 450:
				raise ValueError('This is a classification problem: x needs to be 450 datapoints in WVN range of 950-1800 1/cm')

			mod, h  = add_layer()

		
		if miecorr == True:
			if y.shape[1] != x.shape[1]:
				raise ValueError('This is a regression problem: x and y need 909 datapoints in WVN range of 950-2300 1/cm')

		
			train_layer()

	#def gan_ir_upsample(self,lst_class, lst_ir):
	#def gan_ir_upsample(lst_class, lst_ir):
	#def gan_ir_upsample(path_class, path_ir):
	def gan_ir_upsample(self,path_class, path_ir):
		"""
		INPUT:
				lst_class: python path containing groundtruth maps
				lst_ir:    python path containing IR images
	
		OUTPUT:
	
		
	
	
		"""
	
	
	
	
	
		import matplotlib.pyplot as plt
		import numpy as np
		import os
		import re
		import scipy.io as sio
		import tensorflow as tf
		#import torchvision as tv
		
		
		def convert_data_2_tensor(lst_class, lst_ir, BATCH_SIZE):
			"""Convert images from numpy array to an TensorFlow dataset object."""
			tensor_class = tf.convert_to_tensor(lst_class)
			tensor_ir = tf.convert_to_tensor(lst_ir)
		
			dataset = tf.data.Dataset.from_tensor_slices((tensor_class, tensor_ir))
			dataset = dataset.batch(BATCH_SIZE)
			return dataset
		
		
		def read_data_2_array_np(path_to_class, path_to_ir):
			"""Read data from filesystem to array and return the array lists.
		
			Will be further proceed by the method 'convert_data_2_tensor'.
			"""
			lst_class = []
			for file_path in sorted_nicely(os.listdir(path_to_class)):
				img_class = sio.loadmat(path_to_class + "/" + file_path)
				img_class = img_class['multi'].transpose((2, 0, 1))
				img_class = img_class.astype(dtype=np.float32)
				img_class = np.divide(img_class, 255)
				lst_class.append(img_class[(1, 2, 3), :, :])
			lst_ir = []
			for file_path in sorted_nicely(os.listdir(path_to_ir)):
				img_ir = sio.loadmat(path_to_ir + "/" + file_path)
				img_ir = img_ir['xx'].transpose((2, 0, 1))
				img_ir = img_ir.astype(dtype=np.float64)
				lst_ir.append(img_ir[0:1, :, :])
			return lst_class, lst_ir
		
		# obsolete
		def read_data_2_array_tv(path_to_class, path_to_ir):
			"""Read data from filesystem to array and return the array lists.
		
			Will be further proceed by the method 'convert_data_2_tensor'.
			"""
			lst_class = []
			for file_path in sorted_nicely(os.listdir(path_to_class)):
				img_class = sio.loadmat(path_to_class + "/" + file_path)
				normalize_in_torch = tv.transforms.ToTensor()
				img_class = normalize_in_torch(img_class['multi']).numpy() # normalize image
				lst_class.append(img_class[(1,2,3),:,:])
			lst_ir = []
			for file_path in sorted_nicely(os.listdir(path_to_ir)):
				img_ir = sio.loadmat(path_to_ir + "/" + file_path)
				normalize_in_torch = tv.transforms.ToTensor()
				img_ir = normalize_in_torch(img_ir['xx']).numpy()
				lst_ir.append(img_ir[0:1,:,:])
			return lst_class, lst_ir
		
		
		def show_img_mat(path):
			"""Print image as a matplot-plot."""
			fig = plt.figure()
			data = sio.loadmat(path)
			x = data['gen']
			x = x-x.min() #shift/normalize  x_min to be 0 and x_max to be 1
			x = x/x.max()
			plt.title(path)
			plt.imshow(x[0][0], cmap='binary')
			plt.show()
		
		
		def sorted_nicely( l ):
			""" Sort the given iterable in the way that humans expect."""
			convert = lambda text: int(text) if text.isdigit() else text
			alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
			return sorted(l, key = alphanum_key)
		
		
		def gan(lst_class_files, lst_ir_files, MODEL_LOAD_PATH=str(MODELPATH), BATCH_SIZE_TEST=1):
			
			#---------------------------------------
			#  CHANGE MODEL DIRECTORY TO GLOBAL VAR. 
			#---------------------------------------
	
	
	
			"""Run GeneratorUNet (Main Method)."""
			# Convert data to an TensorFlow dataset
			dataset = convert_data_2_tensor(lst_class_files, lst_ir_files, BATCH_SIZE_TEST)
		
			# Read saved TensorFlow model (pb file)
			generator = tf.keras.models.load_model(
				filepath=MODEL_LOAD_PATH,
			)
		
			loop_count = 1
			iterator = iter(dataset)
			for img_class, img_ir in iterator:
				img_class = tf.transpose(img_class, perm=(0, 2, 3, 1))
				img_ir = tf.transpose(img_ir, perm=(0, 2, 3, 1))
				_out_img = generator((
					img_ir,
					img_class))
				_out_img_mat = tf.make_ndarray(tf.make_tensor_proto(_out_img))
				_out_img_mat = np.transpose(_out_img_mat, [0, 3, 1, 2])
				sio.savemat('higher_res{}.mat'.format(loop_count), dict({"gen": _out_img_mat}))
	
				#------------------------------------------------------------
				# TODO change to internal OpenVibSpec fromat instead of *.mat
				#------------------------------------------------------------
	
	
				#show_img_mat('out/tf_generated_imgs/{}.mat'.format(loop_count))
				loop_count = loop_count + 1
		
		
		#if __name__ == "__main__":
			# Read data to an list and convert 2 array
	
		lst_class_files_np, lst_ir_files_np = read_data_2_array_np(str(path_class), str(path_ir))
		# Obsolete because we changend to numpy instead of torch
		# lst_class_files_tv, lst_ir_files_tv = read_data_2_array_tv("datasets/Class", "datasets/IR")
		
		#if np.array_equal(np.asarray(lst_class_files_np), np.asarray(lst_class_files_tv)):
		#	print("True")
		
		gan(lst_class_files_np, lst_ir_files_np)
	
	
		return








def ocsvm(x, noise=0.03, g=0.001,show=False):
	"""
	Outlier detection for use in spectroscopy.

	https://papers.nips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf
	
	One can also use it for novelty detection, as Schoelkopf had in mind

	Parameters
	----------
	x : numpy array (2D)
			classes per point (as int) with 2 dimensions:  x*y,spectra = shape(), e.g. output of kmeans or similiar clsutering approach
	
	noise : float
			estimate about the given noise

	
	Returns
	-------
	anom: numpy array (2D) of shape x*y,spectra = shape
			selected data points which fall under the assumption of "bad" data points
	
	filt_dat: numpy array (2D)
			selected data points which fall under the assumption of "good" data points aka filtered data




	TODO:
		Adjusting the axis label and legend of the plot!


	"""
	import numpy as np 
	from sklearn.svm import OneClassSVM

	svm = OneClassSVM(kernel='rbf', gamma=g, nu=noise)
	if len(x.shape) ==2:
		
		svm.fit(x)

		pred = svm.predict(x)

		a_indx = np.where(pred==-1)

		n_indx = np.where(pred==1)

		anom = x[a_indx]

		filt_dat = x[n_indx]
	else:
		print("Spectra must be in 2D array")
		print()
		print("n = Axis 0")
		print("Spectra = Axis 1")


	if show == True:

		ax = plt.gca()

		ax.invert_xaxis()

		

		
		

		plt.style.use('ggplot')

		plt.plot(filt_dat.T, color='blue', label='Normal');
		plt.plot(anom.T, color='orange', label='Anomalies');
		

		#plt.legend()
		plt.show()	

		#plt.savefig(name)

		#plt.clf()
	return anom, filt_dat









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
