# OpenVibSpec

[![GitHub Stars](https://img.shields.io/github/stars/RUB-Bioinf/OpenVibSpec.svg?style=social&label=Star)](https://github.com/RUB-Bioinf/OpenVibSpec) 
&nbsp;
[![GitHub Downloads](https://img.shields.io/github/downloads/RUB-Bioinf/OpenVibSpec/total?style=social)](https://github.com/RUB-Bioinf/OpenVibSpec/releases) 
&nbsp;

***

![alt text](/docs/assets/img/vibspec_logo4b.png)

***

[![Contributors](https://img.shields.io/github/contributors/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/graphs/contributors)
&nbsp;
[![License](https://img.shields.io/github/license/RUB-Bioinf/OpenVibSpec?color=green&style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/LICENSE)
&nbsp;
![Size](https://img.shields.io/github/repo-size/RUB-Bioinf/OpenVibSpec?style=flat)
&nbsp;
[![Issues](https://img.shields.io/github/issues/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/issues)
&nbsp;
[![Pull Requests](https://img.shields.io/github/issues-pr/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/pulls)
&nbsp;
[![Commits](https://img.shields.io/github/commit-activity/m/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/)
&nbsp;
[![Latest Release](https://img.shields.io/github/v/release/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/)
&nbsp;
[![Release Date](https://img.shields.io/github/release-date/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec/releases)


### Tool for an integrated analysis workflow in Vibrational Spectroscopy
Our Python library is specialised in the application of machine and deep learning (ML/DL) in the field of biospectroscopic applications. 
In recent years, most of the applications have been based on proprietary software solutions that were in the way of the FAIR principles of open science.
With strong scientific partners from the USA, England, Norway and Germany we want to pave the way for a free source code use of the common standards and latest deep learning models in biospectroscopy.  

### Contributors

- Manchester Institute of Biotechnology, The University of Manchester, (https://www.research.manchester.ac.uk/portal/en/researchers/alex-henderson(662eea20-e79b-424a-ab85-336adc2d9eb0)/contact.html)
- Biospectroscopy and Data Modeling Group, Norwegian University of Life Sciences, Ås, (https://www.nmbu.no/en/faculty/realtek/research/groups/biospectroscopy) 
- Chemical Imaging and Structures Laboratory, University of Illinois at Urbana-Champaign, (http://chemimage.illinois.edu/group.html)
- Leibniz IPHT, Jena, (https://www.leibniz-ipht.de/en/departments/photonic-data-science-2/)
- Quasar (https://quasar.codes/)
- Biomedical Analysis Group, National Academy of Sciences of Belarus, (https://image.org.by/)


### Methods and models which are included
OpenVibSpec offers the possibility to handle everything from data import of raw measurements to ML/DL based data analysis in one ecosystem.
We draw on established groundwork in the Python ecosystem to guarantee the best possible longevity.

You can find all pre-trained models in the [wiki](https://github.com/RUB-Bioinf/OpenVibSpec/wiki/Pretrained-Models).

#### Infrared-based biospectroscopy of Tissue
At this point, a distinction can be made between the fields of the established ML models and the new DL models.
The former is based on the model-based correction of Mie scattering.
The Mie correction based worflow is based on the fundamental paradigm of making the data understandable to the individual domain experts from spectroscopy.
There are several options for correcting Mie scattering, all of which we would like to introduce.
But first, for an overview, see Fig.1., here we see what the workflow is based on. 

***

![alt text](/docs/assets/img/ir_workflow_MieCorr.png)

#### Machine Learning Models with Mie-Correction
In short:
1) After the measurement, the raw data is imported into OpenVibSpec.
2) The correction of the Mie scattering in OpenVibSpec. 
3) Selection of the training data and the training of the machine learning algorithms in OpenVibSpec (mostly Random Forest).
4) Validation of the training data and the models on independent data.


The literature on Mie-corrected data analysis in biospectroscopy, predominantly shows the use of Random Forest class classifiers after correction and selection of the training data.
Currently, data can be imported into OpenVibSpec via the HDF5 and Matlab interfaces.
Further, the raw data import is currently available for the Agilent FTIR and the DRS Daylight Solutions QCL spectroscopes.

#### Deep Learning Models without Mie-Correction
The second major area includes the latest methods around deep learning.
Here, the immense amount of data could be used to generate models and approaches that do not require direct correction of the IR spectroscopic data.
This makes the model as a whole transferable and also significantly faster to analyse.
This gives us two possible workflows for segmenting DNNs in Openvibspec, as shown in Figure 2.

Option A) shows that there are FTIR spectra based on the FFPE embedding of tissue and originating from the entity colon that can be classified in this way.

Possibility B) shows the procedure e.g. for spectra from other entities or embeddings like Fresh Frozen Tissue. This results in a short transfer learning stage to make the models transferable for the own data.

***

![alt text](/docs/assets/img/github_workflowDL.png)


## Getting Started with OpenVibSpec

[![Language](https://img.shields.io/github/languages/top/RUB-Bioinf/OpenVibSpec?style=flat)](https://github.com/RUB-Bioinf/OpenVibSpec)
&nbsp;
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/RUB-Bioinf/OpenVibSpec.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RUB-Bioinf/OpenVibSpec/context:python)
&nbsp;

***

What your are reading right now is the main README.
We advise you to read the guides in the wiki section of this repository.
The first article to start with is: https://github.com/RUB-Bioinf/OpenVibSpec/wiki/Getting-Started

***

![alt text](/docs/assets/img/copy_paste_tutor.png)

## Example Data

Exhaustive Example data is documented and available to download for free in the [wiki](https://github.com/RUB-Bioinf/OpenVibSpec/wiki/Example-Data).
This includes real world data used for training and predicting.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5956845.svg)](https://doi.org/10.5281/zenodo.5956845)

## Docker
You can also download an image from [DockerHub](https://hub.docker.com/u/openvibspec).

| Miniconda Version | Anaconda3 Version |
| --- | --- |
| [![Docker Build](https://img.shields.io/docker/automated/openvibspec/miniconda?style=flat)](https://hub.docker.com/r/openvibspec/miniconda)       | [![Docker Build](https://img.shields.io/docker/automated/openvibspec/anaconda3?style=flat)](https://hub.docker.com/r/openvibspec/anaconda3)       |
| [![Docker Image Size](https://img.shields.io/docker/image-size/openvibspec/miniconda?style=flat)](https://hub.docker.com/r/openvibspec/miniconda) | [![Docker Image Size](https://img.shields.io/docker/image-size/openvibspec/anaconda3?style=flat)](https://hub.docker.com/r/openvibspec/anaconda3) |
| [![Docker Downloads](https://img.shields.io/docker/pulls/openvibspec/miniconda?style=flat)](https://hub.docker.com/r/openvibspec/miniconda)       | [![Docker Downloads](https://img.shields.io/docker/pulls/openvibspec/anaconda3?style=flat)](https://hub.docker.com/r/openvibspec/anaconda3)       |
| [![Docker Stars](https://img.shields.io/docker/stars/openvibspec/miniconda?style=flat)](https://hub.docker.com/r/openvibspec/miniconda)           | [![Docker Stars](https://img.shields.io/docker/stars/openvibspec/anaconda3?style=flat)](https://hub.docker.com/r/openvibspec/anaconda3)           |
| [![Docker Version](https://img.shields.io/docker/v/openvibspec/miniconda?style=flat)](https://hub.docker.com/r/openvibspec/miniconda)             | [![Docker Version](https://img.shields.io/docker/v/openvibspec/anaconda3?style=flat)](https://hub.docker.com/r/openvibspec/anaconda3)             |


#### References
[1](https://www.nature.com/articles/nprot.2014.110)
[2](https://onlinelibrary.wiley.com/doi/abs/10.1002/jbio.201200132)
[3](https://pubs.rsc.org/en/content/articlelanding/2010/an/b921056c/unauth)
[4](https://academic.oup.com/bioinformatics/article/36/1/287/5521621?login=true)
[5](https://onlinelibrary.wiley.com/doi/full/10.1002/jbio.202000385)
