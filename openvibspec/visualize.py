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




#def plot_spec(cc,w,x=int(),y=int(), name='fig.pdf'):
def plot_spec(cc,w,show=None,name='fig.pdf'):
	'''


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