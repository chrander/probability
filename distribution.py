from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class Distribution(ABC):
    """Abstract class for probability distributions"""

    @abstractmethod
    def fit(cls, 
            prior_hyperparameters: NamedTuple, 
            data: np.array) -> 'Distribution':
        raise NotImplementedError

    @abstractmethod
    def update(self, data: np.array) -> None:
        raise NotImplementedError
