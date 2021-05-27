from __future__ import absolute_import
###########################################
# Registration procedures for FTIR/Raman spectroscopy and regular mircoscopic images like H&E
# The first registration is based on the paper/code: 
#	https://onlinelibrary.wiley.com/doi/full/10.1002/jbio.201960223
#	by Stanislau Trukhan
###########################################
import h5py
import numpy as np 

#----------------------------------------------------------------------------------------------------------
class ImageRegistration:
	"""
	A class registration procedures in FTIR/Raman spectroscopy and regular mircoscopic images like H&E

	...

	Attributes
	----------
	name : str
		name of file in the end
	ircube : str
		name/loc of IR data cube 
	he_image : int
		name/loc of corresponding microscopic image
	method : str
		type of method = affine | bspline
	landmarks_IR : str
		name of IR.csv file for pre-registration 
	landmarks_HE : str
		name of HandE.csv file for pre-registration
	profile : int
		determine the preloaded profile for the registration




	Methods
	-------
	info(additional=""):
		
	"""



	def __init__(self, name, ircube, he_image, method,landmarks_IR,landmarks_HE, profile):
		self.name = name # als prefix 2 output
		self.ircube = ircube
		self.he_image = he_image
		self.method = method
		self.landmarks_IR = landmarks_IR	
		self.landmarks_HE = landmarks_HE
		self.profile = profile # int()

	import os
	import numpy as np
	from PIL import Image as ImagePil
	import matplotlib.pyplot as plt
	#---------------------------------------------------------------------------------
	from irreg.io import read_spimage
	from irreg.representations import emsc_b, linear_YCbCr
	from irreg.normalizations import standard_scaler, min_max_scaler
	from irreg.utils.spectra import SpectraStatistics, Spectra
	from irreg.utils.image import Spimage
	from irreg.registration.landmark_based_registrator import LandmarkBasedRegistrator
	from irreg.registration.sitk_registrator_from_json import SitkRegistratorFromJson, load_and_validate_params

	def grayfromIR(self):
		
		ir_filename = os.path.join(os.getcwd(), '..', 'data', self.ircube)
		
		spectral_cube, wavenumbers = read_spimage(ir_filename)
	
		ir_gray = emsc_b(spectral_cube, wavenumbers)

		ir_gray_im = ImagePil.fromarray(min_max_scaler(ir_gray, 0, 255).astype('uint8'))
		
		ir_gray_im = ImagePil.fromarray(min_max_scaler(standard_scaler(ir_gray), 0, 255).astype('uint8'))

		if save == True:

			ir_gray_im.save(os.path.join(os.getcwd(), '..', 'data', 'ir_gray.png'), compress_level=9)

		else:

			pass

		return ir_gray, ir_gray_im

	def grayfromHandE(self):
		
		he_filename = os.path.join(os.getcwd(), '..', 'data', self.he_image)

		he_image = ImagePil.open(he_filename)

		he_gray_im = ImagePil.fromarray(min_max_scaler(he_gray, 0, 255).astype('uint8'))

		he_gray_im.save(os.path.join(os.getcwd(), '..', 'data', 'he_gray.png'), compress_level=9)

		return he_gray, he_gray_im

	def manual_registration(self):
		"""
		Use Fiji to find such landmarks (https://imagej.net/Downloads). 
		You need to install import/export macros to Fiji (see scripts/multiPointSet_export(import).ijm). 
		Follow the manual at https://borda.github.io/dataset-histology-landmarks/ on how to create annotations using Fiji. 
		3-4 matched points should be enough to estimate the initial transformation.

		fixed image Is IR image = fixed_initial.csv

		moving image Is H&E image = moving_initial.csv


		"""
		landmark_registrator = LandmarkBasedRegistrator(

		os.path.join(os.getcwd(), '..', 'data', 'landmarks', self.landmarks_IR),
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', self.landmarks_HE),
		ir_gray,
		he_gray,
		'test',
		os.path.join(os.getcwd(), '..', 'data', 'initial_results'),)
		
		landmark_registrator.register()
		
		landmark_registrator.write_results()

		print(landmark_registrator.final_transform)

		landmark_registrator.evaluate(
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', 'fixed_test.csv'),
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', 'moving_test.csv'),)

		ImagePil.open(os.path.join(os.getcwd(), '..', 'data', 'initial_results', 'test_tre.png'))

		return

	def affine_registration(self, profile):
		"""
		2 moeglichkeiten: json als predefined laden lassen oder eben eigenen laden lassen
		am besten über 
		if json= str():
			lade deinen
		if jason = 1:
			lade pre-json1
		if json = 2:
			lade pre-json2
		"""
		if json == 1:

			affine_params_filename = os.path.join(os.getcwd(), '..', 'irreg', 'registration', 'schemas', '1_affine_cg.json')
			
			affine_params = load_and_validate_params(affine_params_filename)
		
		if json == 2:
		
			affine_params_filename = os.path.join(os.getcwd(), '..', 'irreg', 'registration', 'schemas', '2_affine_gd.json')
			
			affine_params = load_and_validate_params(affine_params_filename)
		
		else:
			
			affine_params_filename = os.path.join(os.getcwd(), '..', 'irreg', 'registration', 'schemas', str(self.profile))
			
			affine_params = load_and_validate_params(affine_params_filename)

		
		affine_registrator = SitkRegistratorFromJson(affine_params, ir_gray, he_gray,'test',
		os.path.join(os.getcwd(), '..', 'data', 'affine_results'),
		os.path.join(os.getcwd(), '..', 'data', 'initial_results', 'test_transform.txt'),)

		affine_registrator.register()

		affine_registrator.write_results()

		print(affine_registrator.final_transform)


		affine_registrator.evaluate(
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', 'fixed_test.csv'),
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', 'moving_test.csv'),)

		ImagePil.open(os.path.join(os.getcwd(), '..', 'data', 'affine_results', 'test_tre.png'))


		return

	def bspline_registration(self, profile):
		

		if json == 1:

			bspline_params_filename = os.path.join(os.getcwd(), '..', 'irreg', 'registration', 'schemas', '3_bspline_lbfgs2.json')
			
			bspline_params = load_and_validate_params(bspline_params_filename)
		
		if json == 2:
			
			bspline_params_filename = os.path.join(os.getcwd(), '..', 'irreg', 'registration', 'schemas', '4_affine_gd.json')
			
			bspline_params = load_and_validate_params(bspline_params_filename)

		else:
			
			bspline_params_filename = os.path.join(os.getcwd(), '..', 'irreg', 'registration', 'schemas', '3_bspline_lbfgs2.json')
			
			bspline_params = load_and_validate_params(bspline_params_filename)

		bspline_registrator = SitkRegistratorFromJson(
			bspline_params,
			ir_gray,
			he_gray,
			'test',
			os.path.join(os.getcwd(), '..', 'data', 'bspline_results'),
			os.path.join(os.getcwd(), '..', 'data', 'affine_results', 'test_transform.txt'),)


		bspline_registrator.register()
		bspline_registrator.write_results()

		bspline_registrator.evaluate(
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', 'fixed_test.csv'),
		os.path.join(os.getcwd(), '..', 'data', 'landmarks', 'moving_test.csv'),)

		ImagePil.open(os.path.join(os.getcwd(), '..', 'data', 'bspline_results', 'test_tre.png'))

		return