
from .results import Results
import torch
import pickle

class Masks(Results):

    def __init__(self, masks = {}):

        assert isinstance(masks, dict)

        self.masks = {}

        for key in masks.keys():

            int_key = key
            if not isinstance(int_key, int):
                int_key = int(int_key)
            self.masks[int_key] = torch.ByteTensor(masks[key])

    def toJSON(self):

        return_masks = {}
        for key in self.masks.keys():
            return_masks[f'{key}'] = self.masks[key].tolist()
        
        return return_masks

    def combine_results(self, additional_results, config=None):
        
        assert isinstance(additional_results, Masks), f'Must combine masks with masks, rather than {type(additional_results)}'

        self.masks.update(additional_results.masks)

    def write(self, video_name):

        with open(video_name, 'wb') as f:
            pickle.dump(self.masks, f)