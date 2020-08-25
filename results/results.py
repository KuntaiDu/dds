import numpy as np
from abc import abstractmethod
import pickle


class Results:
    def __init__(self):
        self.inference_results = {}

    @abstractmethod
    def toJSON():
        pass

    def load(self, fname):
        self.inference_results = pickle.load(open(fname, "rb"))

    def save(self, fname):
        pickle.dump(self.inference_results, open(fname, "wb"))