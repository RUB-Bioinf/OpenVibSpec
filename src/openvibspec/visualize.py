from __future__ import absolute_import
###########################################
# Plotting
# 
#
#
#
###########################################
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
###########################################

def segmentation(km, x, y, show=None, name='fig.pdf'):
	'''

FUNCTION TO GET A READY SEGMENTATION PLOT

SYNOPSIS: segmentation(classes, x, y, show=None, name='fig.pdf')

	Parameters
	----------
	km : numpy array of ints with x*y dimension
			classes per point (as int)
	x : int
			x-axis dimension
	y : int
			y-axis dimension
	show : bool
			changes plot and save behaviour
				show=True plots segmentation to screen
	name : str
			pdf file name for saving the plot
		
	'''
	

	from datetime import datetime
	from matplotlib import pyplot as plt
	plt.rcParams["axes.grid"] = False
	# current date and time
	now = datetime.now()

	n = str(now)

	namedate = '%s%s'%( n.strip(), name)

	if show == True:
		
		plt.imshow(km.reshape(x,y),cmap='gist_rainbow');

		plt.show()

	else:
		if name != 'fig.pdf':
	
			plt.savefig(name)
		
		else:
		
			plt.savefig(namedate)
		
		plt.clf()


def change2color_profile(dat):
	
	"""
	FUNCTION TO GET SEGMENTATION PLOT WITH THE COLOR SCHEME OF MATLAB USED IN THE FOLLOWING PAPER:

	https://pubs.rsc.org/en/content/articlelanding/2016/fd/c5fd00157a#!divAbstract

		Parameters
		----------
		dat : numpy array (2D)
				classes per point (as int) with dimensions x,y
		
		Returns
		-------

		gtnl_array : numpy array
					output array with dimensions x,y,z, where 'z' has 3 channels 

	
	"""
	x,y = dat.shape
	
	gtnl = dat.reshape(x*y).tolist()
	
	
	
	for i,j in enumerate(gtnl): 
		if j==0: 
			gtnl[i]=[255, 145, 145]
	
		if j==1: 
			gtnl[i]=[255, 255, 0]	
		
		if j==2: 
			gtnl[i]=[171, 135, 115]
		
		if j==3: 
			gtnl[i]=[0, 255, 255]
		
		if j==4: 
			gtnl[i]=[0, 255, 0]
		
		if j==5: 
			gtnl[i]=[0, 128, 255]
		
		if j==6: 
			gtnl[i]=[255, 0, 128]
		
		if j==7: 
			gtnl[i]=[255, 0, 255]
		
		if j==8: 
			gtnl[i]=[255, 128, 0]
		
		if j==9: 
			gtnl[i]=[0, 0, 0]
		
		if j==10: 
			gtnl[i]=[255, 0, 0]
		
		if j==11: 
			gtnl[i]=[128, 0, 128]
		
		if j==12: 
			gtnl[i]=[0, 0, 196]
		
		if j==13: 
			gtnl[i]=[64, 191, 255]
		
		if j==14: 
			gtnl[i]=[255, 255, 255]
		
		if j==15: 
			gtnl[i]=[0, 0, 255]
		
		if j==16: 
			gtnl[i]=[0, 196, 0]
		
		if j==17: 
			gtnl[i]=[128, 128, 0]
		
		if j==18: 
			gtnl[i]=[196, 0, 0]
		
		if j==19: 
			gtnl[i]=[0,196,196]

	gtnl_array = np.asarray(gtnl)
	return gtnl_array.reshape(x,y,3)
	



#def plot_spec(cc,w,x=int(),y=int(), name='fig.pdf'):
def plot_spec(cc,w,show=None,name='fig.pdf'):
	'''
FUNCTION PLOTTING SPECTRA:

	Parameters
	----------
	cc : numpy array (2D)
			classes per point (as int) with 2 dimensions:  x*y,spectra = shape()
	w : numpy array (1D)
		wavenumbers:  x*y = shape()
	show : bool
			changes plot and save behaviour
				show=True plots segmentation to screen
	name : str
			pdf file name for saving the plot
		
	'''

	from datetime import datetime
	# current date and time
	now = datetime.now()

	n = str(now)

	namedate = '%s%s'%( n.strip(), name)

	if show == True:

		plt.ylabel('a.u.')

		plt.xlabel('WVN (1/cm)')

		plt.plot(w.T,cc.T)
		
		ax = plt.gca()
		
		ax.invert_xaxis()
		
		plt.show()
	else:
		plt.ylabel('a.u.')
		
		plt.xlabel('WVN (1/cm)')
		
		plt.plot(w.T,cc.T)
		
		ax = plt.gca()
		
		ax.invert_xaxis()
		
		if name != 'fig.pdf':

			plt.savefig(name)
		
		else:

			plt.savefig(namedate)
		
		plt.clf()

	return

def plot_class_spec(cc,w,classnames, cl=1,name='fig.pdf'):
	"""
FUNCTION PLOTTING ALL SPECTRA OF A CERTAIN CLASS IN GREYTONES AND THE CLASS MEAN IN ORANGE:

Examples in the Supplementary data of:

https://academic.oup.com/bioinformatics/article-abstract/36/1/287/5521621

	Parameters
	----------
	cc : numpy array (2D)
			classes per point (as int) with 2 dimensions:  x*y,spectra = shape(), e.g. output of kmeans or similiar clsutering approach
	w : numpy array (1D)
		wavenumbers:  x*y = shape()
	cl : int
				classname as indicated in the 'cc' array
	show : bool
			changes plot and save behaviour
				show=True plots segmentation to screen
	name : str
			pdf file name for saving the plot
		
	"""
	plt.style.use('grayscale')
	
	plt.ylabel('a.u.')
	
	plt.xlabel('WVN (1/cm)')
	
	ax = plt.gca()
	
	ax.invert_xaxis()

	#y = classnames[np.where(classnames == cl)]
	
	#d = cc[y]
	d =  cc[np.where(classnames == cl)]
	print(d.shape)
	
	plt.plot(w.T,d.T)#,label='Spectra of Class' +'%cl' );
	
	#ax = plt.gca()
	#ax.invert_xaxis()
	cm = np.mean(d,axis=0)

	plt.style.use('ggplot')

	plt.plot(w.T,cm.T, color='orange', label='Mean Spec');
	#ax = plt.gca()
	#ax.invert_xaxis()
	plt.legend()
	
	plt.savefig(name)
	
	plt.clf()
	return

def change_rgb_to_classes(pl):
	"""
Changes RGB to CLasses from

FUNCTION TO GET SEGMENTATION PLOT WITH THE COLOR SCHEME OF MATLAB USED IN THE FOLLOWING PAPER:

	https://pubs.rsc.org/en/content/articlelanding/2016/fd/c5fd00157a#!divAbstract

		Parameters
		----------
		dat : numpy array (2D)
				classes per point (as int) with dimensions x,y
		
		Returns
		-------

		gtnl_array : numpy array
					output array with dimensions x,y,z, where 'z' has 3 channels 

	"""

	for n,i in enumerate(pl):
	       if i==[255, 145, 145]:
	               pl[n]=0
	       if i==[255, 255, 0]:
	               pl[n]=1
	       if i==[171, 135, 115]:
	               pl[n]=2
	       if i==[0, 255, 255]:
	               pl[n]=3
	       if i==[0, 255, 0]:
	               pl[n]=4
	       if i==[0, 128, 255]:
	               pl[n]=5
	       if i==[255, 0, 128]:
	               pl[n]=6
	       if i==[255, 0, 255]:
	               pl[n]=7
	       if i==[255, 128, 0]:
	               pl[n]=8
	       if i==[0, 0, 0]:
	               pl[n]=9
	       if i==[255, 0, 0]:
	               pl[n]=10
	       if i==[128, 0, 128]:
	               pl[n]=11
	       if i==[0, 0, 196]:
	               pl[n]=12
	       if i==[64, 191, 255]:
	               pl[n]=13
	       if i==[255, 255, 255]:
	               pl[n]=14
	       if i==[0, 0, 255]:
	               pl[n]=15
	       if i==[0, 196, 0]:
	               pl[n]=16
	       if i==[128, 128, 0]:
	               pl[n]=17
	       if i==[196, 0, 0]:
	               pl[n]=18
	return(np.asarray(pl))

#				def plot_class_spec2(cc,w,classnames, cl=1,name='fig.pdf'):
#					plt.style.use('grayscale')
#					plt.ylabel('a.u.')
#					plt.xlabel('WVN (1/cm)')
#					ax = plt.gca()
#					ax.invert_xaxis()
#					y = classnames[np.where(classnames == cl)]
#					d = cc[y]
#					plt.style.use('ggplot')
#					cm = np.mean(d,axis=0)
#					#plt.plot(w.T,cm.T, color='orange', label='Mean Spec');
#				
#				
#				
#				
#				
#				
#				
#					print(d.shape)
#					plt.plot(w.T,cc[y].T)#,label='Spectra of Class' +'%cl' );
#					#cm = np.mean(d,axis=0)
#					#plt.style.use('ggplot')
#					#ax = plt.gca()
#					#ax.invert_xaxis()
#					#plt.plot(w.T,cm.T, color='orange', label='Mean Spec');
#					plt.legend()
#					plt.savefig(name)
#					#plt.show()
#					plt.clf()
#					return
#				