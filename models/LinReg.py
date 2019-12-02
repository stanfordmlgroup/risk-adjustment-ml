"""Define Linear Regression Model."""
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval, rand
from hyperopt.mongoexp import MongoTrials
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from .base_model import BaseModel


def extract_model(params):
    """
    Given hyperopt params, return model params and model
    :param params: hyperopt params
    :return: model
    """
    model_map = {
        'none': LinearRegression,
        'l1': Lasso,
        'l2': Ridge
    }
    model = model_map[params['classifier_type']['type']]
    del params['classifier_type']['type']
    del params['classifier_type']['normalize_outcome']
    model_params = params['classifier_type']
    model_params['copy_X'] = True
    model_params['normalize'] = True
    model = model()
    model.set_params(**model_params)
    return model

def generate_objective(dataset):
    """
    Train model and return mae score given current hyperopt params for linreg
    """
    train_X, train_y = dataset.X, dataset.y

    def objective(params):
        print(f'Trying {params}')
        if params['classifier_type']['normalize_outcome']:
            fold_scaler = StandardScaler()
            ys = fold_scaler.fit_transform(np.expand_dims(train_y, -1))
        else:
            ys = train_y

        m = extract_model(params)
        score = cross_val_score(m, train_X, ys, cv=3,
                                scoring='neg_mean_squared_error')
        loss = -1 * np.mean(score)
        print(loss)
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    return objective


class LinReg(BaseModel):
    """Class for all linear regression models."""

    def __init__(self, tuning_metric, trials='trials', bottom_coding=None,
                 transform=None, type=None, **kwargs):
        """Initialize the linear regression model."""
        super(LinReg, self).__init__(bottom_coding=bottom_coding,
                                     transform=transform)
        self.tuning_metric = tuning_metric
        self.alpha_mu, self.alpha_std = -0.5, 2
        self.trials = Trials() if trials == 'trials' else \
            MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
        self.scaler = None

        # Create param search space
        if type is None:
            params = [{
                'type': 'none',
                'normalize_outcome': False
            }]
            self.max_evals = 1
        elif type == 'regular':
            params = [{
                'type': 'none',
                'normalize_outcome': False
            }],
            self.max_evals = 1
        elif type == 'l1':
            params = [{
                'type': 'l1',
                'alpha': hp.lognormal('alpha',
                                      self.alpha_mu,
                                      self.alpha_std),
                'normalize_outcome': False
            }]
        elif type == 'l2':
            params = [{
                'type': 'l2',
                'alpha': hp.lognormal('alpha',
                                      self.alpha_mu,
                                      self.alpha_std),
                'normalize_outcome': False
            }]
        else:
            raise ValueError('Invalid linear model type')

        self.space = {
            'classifier_type': hp.choice('classifier_type', params)
        }

    def tune(self, training_set, logger=None, saver=None):
        """Select optimal hyperparameters

        Refits the model with optimal hyperparameters on th
        full training set.

        Solve hyperopt minimization problem given the space and score function
        `space_eval` returns the true values, while `fmin` only gives indices
        into the val arrays given per tunable parameters

        Args:
            training_set: BaseDataset object containing training data
            logger: optional Logger object
            saver: optional ModelSaver object
        """
        self.training_set = training_set

        objective = generate_objective(self.training_set)
        best = space_eval(self.space, fmin(objective,
                                           self.space,
                                           trials=self.trials, algo=tpe.suggest,
                                           max_evals=self.max_evals))

        if best['classifier_type']['normalize_outcome']:
            self.scaler = StandardScaler()
            self.scaler.fit(np.expand_dims(training_set.y, -1))
            ys = self.scaler.transform(np.expand_dims(training_set.y, -1))
        else:
            ys = training_set.y

        print(f'Best hyperparams: {best}')
        self.model = extract_model(best)
        self.model.fit(training_set.X, ys)

    def predict(self, X):
        """Predict based on X.

        Args:
            X: numpy matrix of features

        Return:
            pred: Prediction
        """
        y = self.model.predict(X)
        if self.scaler is not None:
            y = self.scaler.inverse_transform(np.expand_dims(y, -1))
        return np.clip(y, self.bottom_coding, self.transform).flatten()
