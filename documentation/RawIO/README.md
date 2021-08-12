# Raw Spectral Data Input

Here we want to show how OpenVibSpec covers the areas of raw data import. In particular, this is about the import of data from the devices of Daylight Solutions and Agilent. In our example we show usage and synopsis for data based on Daylight Instruments spectroscopes.
```
from openvibspec.io_ftir import RawIO
rio = RawIO()
rio.import_raw_qcl('/testdata/Test_QCL_raw/','newdata_test' ) 

```