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
import os, glob, re
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








#-----------------------------------------------------------------------#
# IMPORT RAW QCL DATA FROM WITHIN A DATA DIRECTORY
#
#
#
#
#-----------------------------------------------------------------------#
#import os, glob, re
#import openvibspec as ov
#import numpy as np
class Imports():
	
	def import_raw_qcl(self,file_dir, new_file_name):


		#self.file_dir = file_dir
		#self.new_file_name = new_file_name

		#def read_raw_qcl(self, file_dir):
		def read_raw_qcl():
			from os import listdir
			import numpy as np
			from os.path import isfile, join
			import scipy.io as sio 
		
			onlyfiles = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
		
		
			its = []
		
			for ii, jj in enumerate(onlyfiles):
		
				its.append(ii)
		
			itsn = np.asarray(its)
		
			return(onlyfiles,itsn, file_dir)
	
	
	
	
	
	
		def cut_info(onlyfiles,itsn,file_dir):
		
			ll_sub = []
			for it in onlyfiles:
			    ll_sub.append(it.split('[')[1].split(']')[0].split('-')[1].split('_'))
			
			
			table = np.asarray(ll_sub, dtype='i4')   
		
			all_n =  np.column_stack((table,itsn))
		
			x_sort = all_n[all_n[:,0].argsort()] 
		
			y_sort = x_sort[x_sort[:,1].argsort()] 
			
			
			sorted_ids = y_sort[y_sort[:,0].argsort()]
			
		
			return sorted_ids
		
		
		
		
		
		def info():
			import glob
			import scipy.io as sio 
		
			files = glob.glob(file_dir+'*')
		
			i = sio.loadmat(files[0])
		
			print(i.keys())
		
			wn = i['wn']
		
			dy = i['dY']
		
			dx = i['dX']
			
			head = i['__header__']
			
			return (wn, wn.shape[0], int(dy),int(dx), head )
		
		
		
		
		
		def load_tiles_by_id(dats, id_,fpasize,wn ):
			import scipy.io as sio
			
			i = sio.loadmat(dats[id_])
		
			d = i['r']
			
			return  d.reshape(fpasize,fpasize,wn)
		
		
		
		
		
		def load_tc():
			from os import listdir
			import numpy as np
			from os.path import isfile, join
			import scipy.io as sio 
			import os

			ll,lon,pf = read_raw_qcl()

			sorted_ids = cut_info(ll,lon,file_dir)
			
			wwn, wnd, dx, dy, head = info()
			
			
			
			os.chdir(file_dir)
		
			l = []
		
			for i in sorted_ids[:,2]:
		
				id_ = load_tiles_by_id(ll,i, dx, wnd )
				l.append(id_)
			
			matrices  = np.asarray(l)

			del l
		
			
			return matrices, wwn, wnd, dx, dy, head
		
		       
		         
		m1, wwn, wnd, dx, dy, head = load_tc()           
		
		ms1 = m1.swapaxes(1,2)[::-1]  
		
	
		def create_mosaic_arrofarrs(images):
			# very strongly inspired by:
			# https://pypi.org/project/ImageMosaic/1.0/
			#		
		
			# Trivial cases
			if len(images) == 0:
				return None
		
			if len(images) == 1:
				return images[0]
		
			imshapes = [image.shape for image in images]
			imdims = [len(imshape) for imshape in imshapes]
			dtypes = [image.dtype for image in images]
			n_cols = int(np.ceil(np.sqrt(len(images))))
			n_rows = int(np.ceil(len(images) / n_cols))
		
			if n_rows <= 0 or n_cols <= 0:
				raise ValueError("'nrows' and 'ncols' must both be > 0")
		
			if n_cols * n_rows < len(images):
				raise ValueError(
					"Grid is too small: n_rows * n_cols < len(images)")
		
			fpa_size_x = images[0].shape[0]
			fpa_size_y = images[0].shape[1]
		
		
			# Allocation
			out_block =  np.ones( (fpa_size_x * n_rows , fpa_size_y * n_cols, images[0].shape[2]), dtype=dtypes[0])
		
			for fld in range(len(images)):
				index_1 = int(np.floor(fld / n_rows))
				index_0 = fld - (n_rows * index_1)
		
				start0 = index_0 * (fpa_size_x)
				end0 = start0 + fpa_size_x
				start1 = index_1 * (fpa_size_y )
				end1 = start1 + fpa_size_y
				out_block[start0:end0, start1:end1] = images[fld]
		
			return out_block
		 
		out = create_mosaic_arrofarrs(ms1)	
	

		def save_mosaic(out):

			import h5py
			import gzip,bz2


			# Standard should be gzip based on the distribution
			# internally choose between gzip,bzip or others

			# ToDo:
			#from mpi4py import MPI
			#import h5py
			#
			#rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
			#
			#f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
			#
			#dset = f.create_dataset('test', (4,), dtype='i')
			#dset[rank] = rank
			#
			#f.close()


			hf = h5py.File(str(new_file_name)+'.h5', 'w')
			hf.create_dataset('HyperSpecCube', data=out, compression="gzip", compression_opts=6)
			hf.create_dataset('WVN', data=wwn, compression="gzip", compression_opts=6)
			#hf.create_dataset('FPA_Size', data=dx, compression="gzip", compression_opts=4)
			#hf.create_dataset('header', data=str(head), compression="gzip", compression_opts=4)
			hf.close()
			return

		def save_overview_image(out):
			import scipy.misc as sm
			
			sm.imsave(str(new_file_name)+'.png',out.mean(axis=2))
			
			return
		save_mosaic(out)
		save_overview_image(out)




		return
