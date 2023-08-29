import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
import numpy
from IPython import embed
'''
annotations_file should be the full path of the csv file with rows [image name, # of spots]
'''


def process_img(img, useSqrt):
    pro_img = img

    lt = 0
    pro_img[pro_img < lt] = lt
    if useSqrt:
        pro_img = numpy.sqrt(pro_img)
    return pro_img

    # image = torch.tensor(image)

    # image = image.squeeze(dim=0)
    # if len(image.shape) == 2:
    #     image = image[None]  # this is called broadcasting, adds an extra dimension
    # image = torch.tensor(image.astype(numpy.float32)) #converts the 3d numpy integer array into a 3d floating point tensor

    # image = self.transform(image).unsqueeze(dim=0) #Added for ResNet
    # image = numpy.repeat(image, 3, axis=0) #take this out and modify the ResNet

#LAL = 1 #Use 1/LAL of the dataset
class CustomImageDataset(Dataset):

    def __init__(self, annotations_file, path_to_hdf5, transform=None, target_transform=None, test=False, useSqrt=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.path_to_hdf5 = path_to_hdf5
        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        self.useSqrt = useSqrt

    def __len__(self):
        return len(self.img_labels)#//LAL


    def process_label(self, label):
        pro_lab = label
        pro_lab = torch.tensor([pro_lab]).type(torch.float32)  # label transformation
        return pro_lab

    def __getitem__(self, idx):
        #get the image as a numpy array
        with h5py.File(self.path_to_hdf5, 'r') as f:
            image = f[self.img_labels.iloc[idx, 0]][()]
        image = process_img(image, useSqrt=self.useSqrt)

        #get the number of spots
        label = self.img_labels.iloc[idx, 1]
        label = self.process_label(label)

        return image, label #, self.img_labels.iloc[idx + idx_offset, 0] #third return value should be the expt_filename
