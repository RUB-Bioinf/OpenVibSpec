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

from scr.openvibspec.nn_utils import get_plt_as_tex
from scr.openvibspec.nn_utils import pgf_plot_colors

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


def spectra_to_tex(spectra: np.ndarray, wave_numbers, plot_title: str, label_y: str, label_x: str = '$cm^{-1}$',
				   included_wave_number_indices: [int] = None, round_plot_bounds: bool = False,
				   max_entries_per_wave_number: int = 200, out_file_path: str = 'my_plot.tex',
				   graph_colors: [str] = None, legend_entries: [str] = None, legend_pos: str = 'north west',
				   used_spectra_mask: np.ndarray = None) -> str:
	"""
	Sets up a tikz pgf-plot for a 2D spectral image.
	If a path is specified, the spectra can be saved as a .tex
	formatted file.
	The formatted text will be returned as a string.

	You can mask out a (int) list of wave-numbers that should be ignored in the output.
	Also a 2D binary array can be supplied to mask out specific areas that will be ignored in the spectral image.

	Note that the required array dimensions are:
	 - "spectra" shape: (x, y, z)
	 - "wave_numbers" shape: (z, 1)
	 - "included_wave_number_indices" shape: (z, 1)
	 - "used_spectra_mask" shape: (x, y)

	x, y, and z should be greater than 0.

	Created by Nils Förster.

	:param spectra: The spectra to be rendered (as a 2D array for every wave-number).
	:param wave_numbers: The corresponding wave-numbers to be rendered (x-axis).
	:param plot_title: The text above the plot.
	:param label_y: The text next to the y-axis.
	:param label_x: The text below the x-axis. Default: cm^(-1)
	:param included_wave_number_indices: A 1D array containing all wave-number indices to be rendered. Optional. If set to None, all well-numbers will be rendered. Default: False
	:param round_plot_bounds: Set to True, if you want to extend the axis bounds a bit and round to 3 digits. Default: False.
	:param max_entries_per_wave_number: How many entries per wave-number should be rendered? Set to lower, if you run into RAM issues during rendering. Best set to default: 200.
	:param out_file_path: A filename where the tikz text should be saved to. File is overwritten. If None, output is not saved. Default: None.
	:param graph_colors: A 1D string array to overwrite graph colors. If None, default colors will be used. Default: None.
	:param legend_entries: A 1D string array to specify legend labels. IF None, no legend will be rendered. Default: None.
	:param legend_pos: The position of the legend. Default: Auto.
	:param used_spectra_mask: A 2D boolean array to mask out spectra from rendering. If None, all spectra will be rendered. Default: None.

	:type spectra: np.ndarray
	:type wave_numbers: np.ndarray
	:type plot_title: str
	:type label_y: str
	:type label_x: str
	:type included_wave_number_indices: [int]
	:type round_plot_bounds: bool
	:type max_entries_per_wave_number: int
	:type out_file_path: str
	:type graph_colors: [str]
	:type legend_entries: [str]
	:type legend_pos: str
	:type used_spectra_mask: np.ndarray

	:returns: The formatted string to be rendered as a LaTeX tikz-pgf plot.
	:rtype: str
	"""

	# Setting up holders to use later for data
	points_x = {}
	points_y = {}
	all_colors = pgf_plot_colors[:-1]

	# Converting the mask to bool. If none have been given via args, a new one is created (consisting of only 1s).
	if used_spectra_mask is None:
		used_spectra_mask = np.ones((spectra.shape[0], spectra.shape[1]))
	used_spectra_mask = used_spectra_mask.astype(np.bool)

	# Asserting data and masks
	assert used_spectra_mask.shape[0] == spectra.shape[0]
	assert used_spectra_mask.shape[1] == spectra.shape[1]

	# Collecting the spectra input dim and reshaping mask accordingly
	dim = spectra.shape[0] * spectra.shape[1]
	used_spectra_mask = used_spectra_mask.reshape(dim)

	# Filling the previously setup dicts with empty list...
	for d in range(dim):
		# ... but only if they are in bounds of the mask
		if used_spectra_mask[d]:
			points_x[d] = []
			points_y[d] = []

	# If graph colors are not given via arguments, they are picked from the list of all possible colors
	if graph_colors is None:
		graph_colors = []
		for d in range(dim):
			graph_colors.append(all_colors[d % len(all_colors)])

	# If the included wave numbers are not given via arguments, they are created new to include all
	if included_wave_number_indices is None:
		included_wave_number_indices = list(range(len(wave_numbers)))
	assert len(included_wave_number_indices) == len(wave_numbers)

	# If the entries for the legend are set, they are asserted to have the right format
	if legend_entries is not None:
		assert legend_pos is not None
		assert len(legend_entries) == len(wave_numbers)

	# iterating over all indices within the wave-number array
	for i in range(len(wave_numbers)):
		# Check if they are not masked out
		if i not in included_wave_number_indices:
			continue

		# Getting the current wave number
		current_wave_number = wave_numbers[i]

		# Getting the current spectrum 2D-image slice and reshaping it to 1D
		current_slice = spectra[:, :, i]
		current_slice = current_slice.reshape(dim)

		# Iterating over all spectra from the slice
		for d in range(dim):
			# Check if they are not masked out
			if used_spectra_mask[d]:
				# Adding the current wave-number and spectrum to the x / y arrays to be plotted
				points_y[d].append(current_slice[d])
				points_x[d].extend(current_wave_number)

	# Converting the x / y dicts to lists
	#  (used dicts to allow sparse entries)
	points_x = list(points_x.values())
	points_y = list(points_y.values())

	# Converting to NP arrays to easily find min / max entries
	points_x_np = np.array(points_x)
	points_y_np = np.array(points_y)
	min_x = float(points_x_np.min())
	max_x = float(points_x_np.max())
	min_y = float(points_y_np.min())
	max_y = float(points_y_np.max())
	# Rounding to the 3rd decimal point if desired
	if round_plot_bounds:
		min_x = round(min_x * 1.05, 3)
		max_x = round(max_x * 1.05, 3)
		min_y = round(min_y * 1.05, 3)
		max_y = round(max_y * 1.05, 3)

	# Formatting to LaTeX tikz string
	tex = get_plt_as_tex(data_list_y=points_y, data_list_x=points_x, plot_colors=graph_colors, label_x=label_x,
						 label_y=label_y, min_x=min_x, max_x=max_x, title=plot_title, plot_titles=legend_entries,
						 max_entries=max_entries_per_wave_number, min_y=min_y, max_y=max_y, legend_pos=legend_pos)

	# If the path is not None, the tex-str is written to the corresponding filename
	if out_file_path is not None:
		f = open(out_file_path, 'w')
		f.write(tex)
		f.close()

	# Returning the text
	return tex


