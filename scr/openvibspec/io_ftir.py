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
import scipy.io as sio 
import sys
###########################################

def read_mat2py(x):
	with h5py.File(x, 'r') as f:
		fk = [str(r) for r in f.keys()]
		print("VARIABELS FROM MATLAB ARE GIVEN AS STRINGS")
		print(fk)
		return(fk)
		#return(fk, f)
		 
#def import_all(x):
#	read_mat2py(x)



def import_dat(x, var):
	""" 
	very important to ensure that the data is:
	x,y,z = data.shape
	inside the python environment i make a transpose on the incoming Matlab File 
	TODO:
	PRINT ONLY SPECIFIC PARTS OF THE LIST WITH VARIABLE-NAMES FROM MATLAB
	"""
	with h5py.File(x, 'r') as f:
		#f.keys()
		#f.items()
		dat =  np.asarray( f[var] )
		return( dat.T )



def import_mat2py(x):
	with h5py.File(x, 'r') as f:
		fk = [str(r) for r in f.keys()]
		for k, v in fk:
			arrays[k] = np.array(v)
		return(arrays[k])

#def importmatlab(x):
#	for k, v in x.items():
#		arrays[k] = np.array(v)
#	return(arrays[k])


def write2HDF(x):
	
	return(x)


def read_old_matlab(x):
	"""
	Matlab-Files before Version 7.3
	"""
	#with sio.loadmat(x) as f:
	#try:
		#f = sio.loadmat(x)
	f = sio.loadmat(x)
	#f = sio.loadmat(x)
	fk = [str(r) for r in f.keys()]
	print("VARIABELS FROM MATLAB ARE GIVEN AS STRINGS")
	print("IN THE FIRST AND THRID POSITION")
	print(fk)

	#except:
		#print("Wrong Function For This Matlab-Format:", sys.exc_info()[0])
	print("Wrong Function For This Matlab-Format:", sys.exc_info()[0])
	return(fk)


def import_old_matlab(x, var):
	"""
	Matlab-Files before Version 7.3
	"""
	#try:
	f = sio.loadmat(x)
	dat =  np.asarray( f[var] )
	#except:
	print("Wrong Function For This Matlab-Format:", sys.exc_info()[0])
	return( dat)















# Python 2.7
#def import_agielnt(x):
#	with open("a01.dmt", "rb") as binary_file:
#    	binary_file.seek(2236)
#    	couple_bytes = binary_file.read()
#    	print(couple_bytes)
#    return 

# Python 3.5
def import_agilent(x):
	with open("a01.dmt", "rb") as binary_file:
		binary_file.seek(2236)
		couple_bytes = binary_file.read()
		sys.stdout.buffer.write(couple_bytes)
	return 
def bands(x):
	import sys
	with open("A01.hdr", "rb") as binary:
		binary.seek(76,0)
		bytes_ = binary.read(6)
		return (sys.stdout.buffer.write(bytes_))








