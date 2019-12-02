"""Define a general class for all models."""
import importlib
import numpy as np

#reg_metrics = importlib.import_module("aihc-stats.stats.regression_metrics")


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, bottom_coding, transform):
        """Initialize hyperparameters."""
        self.model = None
        self.best_params = None
        self.bottom_coding = bottom_coding
        self.transform = transform
        self.tuning_metric = None
        self.training_set = None
        self.model = None
        self.best_params = None
        self.space = {}
        self.max_evals = 50

    def fit(self, training_set,
            logger=None, saver=None):
        """Train on training_set."""
        self.tune(training_set, logger=logger, saver=saver)
        if saver is not None:
                saver.save(self)

    def predict(self, X):
        """Predict based on X.

        Args:
            X: numpy matrix of features

        Return:
            pred: Prediction
        """
        y = self.model.predict(X)
        return np.clip(y, self.bottom_coding, self.transform)

    def tune(self, training_set, logger, saver=None):
        """Select optimal hyperparameters

        Refits the model with optimal hyperparameters on the
        full training set.

        Args:
            training_set: BaseDataset object containing training data
            logger: optional Logger object
            saver: optional ModelSaver object
        """
        raise NotImplementedError

    def instantiate_model(self, params):
        """
        Given the params from the hyperopt space,
        return params for model score function and model
        :param params: hyperopt space params for a given iteration
        :return: params for model
        """
        raise NotImplementedError
