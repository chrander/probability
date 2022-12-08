from typing import NamedTuple
import numpy as np
from scipy import stats

from distribution import Distribution


class GaussianHyperparameters(NamedTuple):
    """Class for hyperparameters of the categorical distribution"""
    m0: float
    k0: float
    s0: float
    mu0: float #TODO: get the right parameters here


class GaussianDistribution(Distribution):
    """Class for a Gaussian Probability Distribution"""
    # TODO: update functions so that they agree with Gaussian arguments

    def __init__(self, prior_hyperparameters: GaussianHyperparameters) -> None:
        self.prior_hyperparameters = prior_hyperparameters 
        self.prior = stats.gausshyper(self.prior_hyperparameter.alpha)
        self.posterior = None
        self.posterior_hyperparameters: GaussianHyperparameters = None
        self.posterior_predictive = None

    @classmethod
    def fit(cls, 
            prior_hyperparameters: GaussianHyperparameters, 
            data: np.array) -> 'GaussianDistribution':
        """Fits a distribution and updates with an initial batch of data
        
        Parameters
        ----------
        prior_hyperparameters: GaussianHyperparameters
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
            self.posterior_hyperparameters = GaussianHyperparameters(alpha=new_alpha)
        # Otherwise, update the posterior
        else:
            new_alpha = self.posterior_hyperparameters.alpha + data
            self.posterior_hyperparameters._replace(alpha = new_alpha)

        self.posterior = stats.dirichlet(self.posterior_hyperparameters.alpha)
        self.posterior_predictive = self.posterior_hyperparameters.alpha / np.sum(self.posterior_hyperparameters.alpha)
