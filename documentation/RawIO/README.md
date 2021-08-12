# Raw Spectral Data Input

Here we want to show how OpenVibSpec covers the areas of raw data import. In particular, this is about the import of data from the devices of Daylight Solutions and Agilent. In our example we show usage and synopsis for data based on Daylight Instruments spectroscopes.
```
from openvibspec.io_ftir import RawIO
rio = RawIO()
rio.import_raw_qcl('/testdata/Test_QCL_raw/','newdata_test' ) 

```
The first two lines of code show how to import everything what's needed for this. The third line shows the actual application on the data of a Daylight Solutions spectroscope. 
The only parameters that need to be specified are the location as a path, here in the example of a Linux environment, and the name of the output-file. In the context of measurements on tissue sections, the output file can become very large. This is one of the reasons why OpenVibSpec uses HDF5 as format and lets it be packed by gzip.
At any time it is worthwhile to use the help function of Python.
```
help(rio)

```
And to be more specific regarding the specific method

```
help(rio.import_agilent_mosaic)

```

In the end your directory will look like the one on the image below. Before the import function was executed, only the .mat files were in the folder. When the raw data import is completed, the file with the data is added, here called "newdat_test.h5", and the overview image is added.

![dir RawIO](/documentation/RawIO/workflow.png) 
