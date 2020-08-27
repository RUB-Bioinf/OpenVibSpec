import os

import numpy as np


MODELPATH = os.path.dirname(__file__)


class FittedKerasModel:

    @property
    def wavenumbers(self):
        """ An array of wavenumbers needed for a model. """
        raise NotImplementedError

    @property
    def name(self):
        return self.NAME

    def load(self):
        """ Return a Keras model """
        return NotImplementedError


class JsonModel:

    def __init__(self):
        self.load = self.load_json  # replace the loading function

    def load_json(self):
        from keras.models import model_from_json
        with open(self.JSON, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(self.WEIGHTS)
        return model


class CellularComponentsClassification(FittedKerasModel, JsonModel):
    """
    Training dataset was based on FFPE (formaldehyde-fixed paraffin-embedded) tissue from colon.
    BIOMAX CO1002b/CO722
    For further information refer to Raulf et al., 2019
    """

    NAME = "Cellular components classification"
    JSON = os.path.join(MODELPATH, 'model_weights_classification.json')
    WEIGHTS = os.path.join(MODELPATH, "model_weights_classification.best.hdf5")

    @property
    def wavenumbers(self):
        return np.linspace(950, 1800, 450)


class RMIECorrection(FittedKerasModel, JsonModel):
    """
    Training dataset was based on FFPE (formaldehyde-fixed paraffin-embedded) tissue from colon.
    BIOMAX CO1002b/CO722
    For further information refer to Raulf et al., 2019
    """

    NAME = "Efficient RMIE correction"
    JSON = os.path.join(MODELPATH, 'model_weights_regression.json')
    WEIGHTS = os.path.join(MODELPATH, "model_weights_regression.best.hdf5")

    @property
    def wavenumbers(self):
        return np.linspace(950, 2300, 909)
