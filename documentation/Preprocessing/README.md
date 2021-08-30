# Correction of Mie Scattering in IR Spectroscopy 

When using infrared light and biological samples (not only there), Mie scattering very often occurs in the spectra. A number of different methods are available to correct this. Some of these methods come from the BioSpec LAb of Achim Kohler at NMBU, Norway. These can be used in Openvibspeec as follows.
```
from  openvibspec.preprocessing import EMSC
import numpy as np
d = np.load('/data/emsc_testdata.npy')
w = np.load('/data/emsc_data_wns.npy')
raw_spectra = d[2:4]

```
The provided code starts with all the necessary imports in terms of code and test data. Within the EMSC class the user gets access to different methods. 
```
emsc = EMSC(raw_spectra.mean(axis=0), w, poly_order=4)
corrected, inn = emsc.transform(raw_spectra, internals=True)
```
As one can see from the screenshot, you get the correted data in form of an numpy array back.

![dir RawIO](/documentation/Preprocessing/emsc_test.png) 


```
help(EMSC)

```

In the end your directory will look like the one on the image below. Before the import function was executed, only the .mat files were in the folder. When the raw data import is completed, the file with the data is added, here called "newdat_test.h5", and the overview image is added.

