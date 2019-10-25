from torch.utils.data import Dataset
from scipy import misc
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
from torchvision import transforms
import matplotlib.pyplot as plt



class ComDataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2):
        self.dataset1 = data1
        self.dataset2 = data2

    def __getitem__(self, i):
        return self.dataset1[i],self.dataset2[i]

    def __len__(self):
        return len(self.dataset1)