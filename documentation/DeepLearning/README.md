# Using Pre-trained Deep Neural Networks in Spectroscopy 

In image classification, significant steps were taken in the last decade with the use of pre-trained NNs, which were then trained in transfer learning for their own application. The models contained in OpenVibSpec can be imported and used via the DeepLearn class. Models for semantic segmentation, Mie scatter correction and interpolation of the IR resolution limit are included.
## The Mie Correction Deep Neural Network

The provided code starts with all the necessary imports in terms of code and test data. Within the DeepLearn class the user gets access to different methods. The first one shows the use of NN to get a fast Mie scattering correction.

```
import openvibspec.ml_ftir as ovml
dl = ovml.DeepLearn()
```
In following lines you can see test data import and transformation. Because all the NNs use the 2D representation of the hyperspectral cube, which is indicated as the variable d2.

```
import numpy as np
d = np.load('openvibspec/data/testdat_noMieffpe.npy')
d2 = d.reshape(25*25,1428)
w = np.load('openvibspec/data/wvn_agilent.npy')
wvn = w[:909,0]
```
In order to get the correct data representation one can use the allready built-in plotting of OpenVibSpec via the specific methods:
```
import openvibspec.visualize as ovviz 
ovviz.plot_spec(x_corr,wvn)
```
```
import openvibspec.visualize as ovviz 
ovviz.plot_spec(x_corr,wvn,show=True)
```
Those methods require the data in 2D representation and a certain vector with the device-specific wavenumbers. We'll include those vectors into the data directory. As one can see from the code above, you can use the plotting with and without 'show' argument. If you only use the first command, OpenVibSpec will save the plot into the currently used directory. OpenVibSpec uses the time-stamp as an name for the file in order to minimize the user input at this point.

```
%timeit x_corr = dl.net(d2[:,:909], miecorr=True)

Loaded model from disk
Loaded model from disk
Loaded model from disk
Loaded model from disk
Loaded model from disk
Loaded model from disk
Loaded model from disk
Loaded model from disk
585 ms ± 35.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

```
The screenshot below shows the corrected data as an OpenVibSpec-plot on the left and the corresponding raw data on the right side.

![dir DeepLearning](/documentation/DeepLearning/mie_corrNN.png) 

Important with all the inconsistencies that arise from the many different options are two things. Firstly, internally we always use the numpy array as the data standard, so that OpenVibSpec can be embedded well in all Python processes and secondly, always remember the help!

```
help(DeepLearn)
```

## Deep Neural Network for Semantic Segmentation of IR data
Also represented in the DeepLearn class is the model for semantic segmentation of IR data from colon tissue. Here we show how how this model can be used to classify raw IR data from the colon tissue. An important part of the raw data classification is to pay attention to the number of wavenumbers on the z-axis of the hyperspectral cube. This model uses only the data range between 950 and 1800 wavenumbers. This range for Agilent FTIR spectroscopes for example is in the first 450 points. Due to the maximum data size that can be uploaded to Github, the image below was created with a larger data set than is stored in the OpenVibSpec data directory.

```
daclass = dl.net(d2[:,:450],classify=True)

max_cl = np.argmax(daclass,axis=1)

img_cl = max_cl.reshape(25,25)

rgb_img = ovviz.change2color_profile(img_cl)

```

From the code you can see how to get a Numpy array from the model with the size xy,z. Here xy are the 2D axes of the actual image and the z vector is a 19 class long vector with the assignment of each pixel to one of the 19 classes. With the OpenVibSpec built-in feature colormap-conversion, you get access the the actual groundtruth labels, by using the last line of code above. You can use this procedure and the conversion to plot this via matplotlib like this, in ipython/jupyter enviroment:

```
import openvibspec.visualize as ovviz 
import matplotlib.pyplot as plt

rgb_img = ovviz.change2color_profile(img_cl)

plt.imshow(rgb_img);
plt.show()
```
By using this deep learning model on uncorrected FITR data from FFPE colon tissue, you get the prediction of the 228.250 data points, used in the given example image below, within less than 20 seconds on an i5 Laptop w/o an GPU.

![dir DeepLearning](/documentation/DeepLearning/ir_classificationNN.png) 


## Transfer Learning with Pre-trained Deep Neural Network

## GANs as a framework for morphologic details beyond the diffraction limit in infrared spectroscopic imaging
A very interesting deep learning application in the field of IR tissue spectroscopy will be implemented in OpenVibSpec via the pre-trained GAN of Falahkheirkhah et al. With this, a generative model can be used to improve the downscaled resolution in spectroscopic image data without having to use light microscopy. The model itself needs the class membership and the IR data in the form of a Matlab file. The data is stored in the folder with label ```out```. In the screenshot shown you can see the loading of the final image plus the display via the python library matplotlib.



![dir DeepLearning](/documentation/DeepLearning/gan_res2.png) 


```
a15 = sio.loadmat('higher_res15.mat')

arr15 = np.array(a15['gen'])

plt.imshow(arr15[0,0,:,:]);

plt.show()
```

From the point of view of the source codes, this is quickly applicable and simple. In the data folder, we have saved two possible test data which were provided to us by Falahkheirkhah and others. For using the pre-trained GAN, please read the paper carefully to extract the correct use cases.

```
import openvibspec.ml_ftir as ovml

dl = ovml.DeepLearn()

dl.gan_ir_upsample('openvibspec/data/datasets/Class/','openvibspec/data/datasets/IR/')
```