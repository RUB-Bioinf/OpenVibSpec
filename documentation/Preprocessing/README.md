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

Important with all the inconsistencies that arise from the many different options are two things. Firstly, internally we always use the numpy array as the data standard, so that OpenvibSpec can be embedded well in all Python processes and secondly, always remember the help!

```
help(EMSC)

```
