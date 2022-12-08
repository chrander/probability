from typing import NamedTuple
import numpy as np
from scipy import stats

from distribution import Distribution


class CategoricalHyperparameters(NamedTuple):
    """Class for hyperparameters of the categorical distribution"""
    alpha: np.array


class CategoricalDistribution(Distribution):
    """Class for a Categorical Probability Distribution"""

    def __init__(self, prior_hyperparameters: CategoricalHyperparameters) -> None:
        self.prior_hyperparameters = prior_hyperparameters 
        self.prior = stats.dirichlet(self.prior_hyperparameters.alpha)
        self.posterior = None
        self.posterior_hyperparameters: CategoricalHyperparameters = None
        self.posterior_predictive = None

    @classmethod
    def fit(cls, 
            prior_hyperparameters: CategoricalHyperparameters, 
            data: np.array) -> 'CategoricalDistribution':
        """Fits a distribution and updates with an initial batch of data
        
        Parameters
        ----------
        prior_hyperparameters: CategoricalHyperparameters
        data: np.array

        Returns
        -------
        CategoricalDistribution
        """
        distribution = cls(prior_hyperparameters)
        distribution.update(data)
        return distribution
    
    def update(self, data: np.array) -> None:
        """Updates a distribution a new batch of data

        Parameters
        ----------
        data: np.array
        """
        # Check that prior is set before we try computing the posterior
        if (self.prior is None) or (self.prior_hyperparameters is None):
            raise ValueError("No prior has been defined. Cannot update the posterior.")

        # If distribution hasn't been updated yet, update the prior
        if self.posterior_hyperparameters is None:
            new_alpha = self.prior_hyperparameters.alpha + data
            self.posterior_hyperparameters = CategoricalHyperparameters(alpha=new_alpha)
        # Otherwise, update the posterior
        else:
            new_alpha = self.posterior_hyperparameters.alpha + data
            self.posterior_hyperparameters._replace(alpha = new_alpha)

        self.posterior = stats.dirichlet(self.posterior_hyperparameters.alpha)
        self.posterior_predictive = self.posterior_hyperparameters.alpha / np.sum(self.posterior_hyperparameters.alpha)
