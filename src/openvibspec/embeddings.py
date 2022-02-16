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

from matplotlib.pyplot import imshow
import requests
import random
import os
import base64
from sklearn.decomposition import TruncatedSVD
from keras.models import Model
from hashlib import md5
import pickle
try:
	from urllib import unquote
except ImportError:
	from urllib.parse import unquote
from PIL import Image
import requests
try:
	from io import BytesIO
except ImportError:
	from StringIO import StringIO as BytesIO
from IPython.display import HTML, Image as iPythonImage, display
import numpy as np
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
######################################################################################################




def prep_dat():



def mod(model_name="VGG16"):
	"""
	
	"""
	import keras
	from keras.models import Model
	from keras.applications.vgg16 import VGG16
	from keras.applications.inception_v3 import InceptionV3
	from keras.applications import ResNet50
	from keras.preprocessing import image


	
	if model_name=="VGG16":
		base_model = VGG16(weights='imagenet', include_top=True)
		return base_model

	if model_name=="InceptionV3":
		base_model = InceptionV3(weights='imagenet', include_top=True)
		return base_model

	if model_name=="ResNet50":
		base_model = ResNet50(weights='imagenet', include_top=True)
		return base_model

def extractor(x, base_model=mod("VGG16"), xcoor=224, ycoor=224):
	"""
	
	"""
	import numpy as np
	import scipy
	from keras.models import Model
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	from keras.preprocessing import image
	
	images = [scipy.misc.imresize(xi,(xcoor,ycoor),'nearest') for xi in x]
	
	np_imgs = [image.img_to_array(img) for img in images]
	model_2 = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
	
	pre_processed = preprocess_input(np.asarray(np_imgs))
	
	return model_2.predict(pre_processed)

def neighbours(vec, no_of_neigh=10):
	"""
	
	"""
	from sklearn.neighbors import NearestNeighbors

	nbrs = NearestNeighbors(n_neighbors=no_of_neigh, algorithm='ball_tree').fit(vec)

	distances, indices = nbrs.kneighbors(vec)

	return distances,indices


def tsne(vec, no_of_components=2,perplexity_steps=40, iterations=500, best=True):
	"""
	Based on Maarten et al. 2008 
	"""
	from sklearn.manifold import TSNE
	from sklearn.neighbors import NearestNeighbors
	import time

	if best==True:
		tsne1 = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=iterations)
		tsne_results1 = tsne1.fit_transform(vec)

		tsne2 = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=iterations)
		tsne_results2 = tsne2.fit_transform(vec)

		return tsne_results1, tsne_results2
	else:
		time_start = time.time()
	
		tsne = TSNE(n_components=no_of_components, verbose=1, perplexity=perplexity_steps, n_iter=iterations)
	
		tsne_results = tsne.fit_transform(vec)

		print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))



		return tsne_results

