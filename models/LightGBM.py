"""Define LightGBM Class."""
from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgb
import importlib
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll.base import scope
import numpy as np

from .base_model import BaseModel

#reg_metrics = importlib.import_module("aihc-stats.stats.regression_metrics")


def generate_objective(dataset, tuning_metric):
    """
    Train model and return r2 score given current hyperopt params for LightGBM
    """
    train_X, train_y = dataset.X, dataset.y
    train_set = lgb.Dataset(train_X, train_y)
    n_folds = 3
    cv_metric_name = {"mse": "l2-mean",
                      "mae": "l1-mean",
                      "huber": "huber-mean"}

    def objective(params):
        print(f'Trying {params}')
        cv_results = lgb.cv(params,
                            train_set,
                            nfold=n_folds,
                            num_boost_round=100,
                            metrics=tuning_metric,
                            seed=50,
                            stratified=False)

        best_score = np.mean(cv_results[cv_metric_name[tuning_metric]])
        loss = best_score
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    return objective


class LightGBM(BaseModel):
    """XGBoost Class."""

    def __init__(self, tuning_metric='mse',
                 trials='trials', bottom_coding=None,
                 transform=None, **kwargs):
        """Initialize hyperparameters."""
        super(LightGBM, self).__init__(bottom_coding=bottom_coding,
                                       transform=transform)
        self.model = LGBMRegressor
        self.tuning_metric = tuning_metric
        self.trials = Trials() \
            if trials == 'trials' \
            else MongoTrials('mongo://localhost:1234/foo_db/jobs',
                             exp_key='exp1')
        self.set_parameters()

    def set_parameters(self):
        """Set the model hyperparameter sweep."""
        self.space = {
            "objective": self.tuning_metric,
            "device": "gpu",
            'min_data_in_leaf': hp.choice('min_data_in_leaf', [100, 1000, 300]),
            'boosting_type': hp.choice('boosting_type', ['gbdt']),
            'num_leaves': scope.int(hp.quniform('num_leaves', 30, 250, 1)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01),
                                           np.log(0.2)),
            'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin',
                                                       20000, 300000, 20000)),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
        }

    def tune(self, training_set, logger=None, saver=None):
        self.training_set = training_set
        objective = generate_objective(self.training_set,
                                       self.tuning_metric)
        best = space_eval(self.space, fmin(fn=objective,
                                           space=self.space,
                                           trials=self.trials,
                                           algo=tpe.suggest,
                                           max_evals=self.max_evals))

        print(f'Search space: {self.space}')
        print(f'Best hyperparams: {best}')

        self.model = LGBMRegressor()
        self.model.set_params(**best)
        self.model.fit(training_set.X, training_set.y)

    def instantiate_model(self, params):
        model = LGBMRegressor()
        model.set_params(**params)
        return model
