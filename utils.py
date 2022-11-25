import torch
from torch.nn import functional as F
import numpy as np
import torchvision
from torchvision import transforms

class Map:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.vectorize(self.mapper.__getitem__, otypes=[np.float32])(input)

class Map2:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.array([[self.mapper[element] for element in row]for row in input], dtype=np.float32)

class ToTensor:
    """
    Convert into a tensor of float32: differently from transforms.ToTensor() this function does not normalize the values in [0,1] and does not swap the dimensions
    """
    def __call__(self, input):
        return torch.as_tensor(input, dtype=torch.float32)

class ToNumpy:
    """
    Convert into a tensor into a numpy array
    """
    def __call__(self, input):
        return input.numpy()

class ToTensorSwap:
    """
    Convert into a tensor of float32: differently from transforms.ToTensor() this function does not normalize the values in [0,1] and does not swap the dimensions
    """
    def __call__(self, input):
        return torch.as_tensor(input, dtype=torch.uint8).permute(2,0,1)

def colorLabel(label, palette):
    composed = torchvision.transforms.Compose([ToNumpy(), Map2(palette), ToTensorSwap(), transforms.ToPILImage()])
    label = composed(label)
    return label

def save_images(palette, predict, path_to_save):
    predict = torch.tensor(predict.copy(), dtype=torch.uint8).squeeze()
    predict = colorLabel(predict, palette) 
    predict.save(path_to_save)
