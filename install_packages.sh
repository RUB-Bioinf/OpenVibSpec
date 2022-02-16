#
#  Original solution via StackOverflow:
#    http://stackoverflow.com/questions/35802939/install-only-available-packages-using-conda-install-yes-file-requirements-t
#
#
#  Install via `conda` directly.
#  This will fail to install all
#  dependencies. If one fails,
#  all dependencies will fail to install.
#-------------------------------------------------------------------------------------------------------------
conda create -n py38 python=3.8
source activate py38
# ------------------------------------------------------------------------------------------------------------
conda install --yes --file requirements.txt

#
#  To go around issue above, one can
#  iterate over all lines in the
#  requirements.txt file.
#
while read requirement; do conda install --yes $requirement; done < requirements.txt
# Registration Problem
pip install git+https://github.com/BioSpecNorway/IRmHiRegistration.git
# MRMR 4 best_practices in (Raman)Spectrsocopic DL
pip install git+https://github.com/smazzanti/mrmr