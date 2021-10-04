from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    Abstract superclass for the different probability distributions used by the forecast models.
    """

    @staticmethod
    @abstractmethod
    def pdf(*args):
        pass

    @staticmethod
    @abstractmethod
    def cdf(*args):
        pass

    @staticmethod
    @abstractmethod
    def mean(*args):
        pass

    @staticmethod
    @abstractmethod
    def var(*args):
        pass

    @staticmethod
    @abstractmethod
    def percentile(*args):
        pass

    @staticmethod
    @abstractmethod
    def crps(*args):
        pass
