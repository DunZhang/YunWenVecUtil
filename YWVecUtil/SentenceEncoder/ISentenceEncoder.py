import abc
import numpy as np
from abc import abstractmethod
from typing import List, Iterable


class ISentenceEncoder(metaclass=abc.ABCMeta):

    @abstractmethod
    def get_sens_vec(self, **kwargs) -> np.ndarray:
        pass
