import cv2
import imutils
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        else:
            gray = cv2.GaussianBlur(img, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)


def read_images(root_address="tumor dataset/"):
    image_address = {
        "yes": [root_address + "yes/" + file_name for file_name in os.listdir(os.path.join(root_address, "yes"))],
        "no": [root_address + "no/" + file_name for file_name in os.listdir(os.path.join(root_address, "no"))]
    }

    images = {
        "yes": list(map(lambda file_name: cv2.imread(file_name, cv2.IMREAD_GRAYSCALE), image_address["yes"])),
        "no": list(map(lambda file_name: cv2.imread(file_name, cv2.IMREAD_GRAYSCALE), image_address["no"]))
    }
    return images


def resize_images(images, size=(224, 224), interpolation=cv2.INTER_LANCZOS4):
    resized_images = {
        "yes": [],
        "no": []
    }
    for key in images.keys():
        resized_images[key] = [cv2.resize(img, size , interpolation=interpolation) for img in images[key]]
        
    return resized_images


import random
from torchvision import transforms
import copy
from PIL import Image
import PIL

def augmentation(images, transformations, increase_rate=1):
    dataset_size = len(images["yes"]) + len(images["no"])
    
    print(dataset_size)
    augmented_images = {
        "yes": [],
        "no": []
    }

    for i in range(int(increase_rate * dataset_size)):
        if len(transformations) > 0:
            transform = random.choice(transformations)
        else:
            transform = None

        index = random.randrange(dataset_size)
        if index < len(images["no"]):
            augmenting_image = images["no"][index]
            if transform:
                new_image = transform(Image.fromarray(augmenting_image))
            else:
                new_image = copy.deepcopy(augmenting_image)
            augmented_images["no"].append(np.array(new_image))
        else:
            augmenting_image = Image.fromarray(images["yes"][index - len(images["no"])])
            if transform:
                new_image = transform(augmenting_image)
            else:
                new_image = copy.deepcopy(augmenting_image)
            augmented_images["yes"].append(np.array(new_image))
        
        print("I is: " + str(i))
#         plt.figure()
#         plt.imshow(augmenting_image)
#         plt.figure()
#         plt.imshow(new_image)

    temp_images = copy.deepcopy(images)
    temp_images["yes"].extend(augmented_images["yes"])
    temp_images["no"].extend(augmented_images["no"])

    return temp_images, augmented_images


def write_images(images, root_address="preprocessed dataset/"):
    if not os.path.exists(os.path.join(root_address, "no")):
        os.makedirs(os.path.join(root_address, "no"))
    if not os.path.exists(os.path.join(root_address, "yes")):
        os.makedirs(os.path.join(root_address, "yes"))
    for key in images.keys():
        for i, img in enumerate(images[key]):
            print(i, key)
            cv2.imwrite(root_address + str(key) + "/" + str(i) + ".png", img)
  

from torch.utils.data import Dataset, DataLoader, random_split

class TumorDataset(Dataset):
    def __init__(self, root_address="preprocessed dataset/"):
        super(TumorDataset, self).__init__()  
        self.root_address = root_address
        self.image_address = {
            "yes": [root_address + "yes/" + file_name for file_name in os.listdir(os.path.join(root_address, "yes"))],
            "no": [root_address + "no/" + file_name for file_name in os.listdir(os.path.join(root_address, "no"))]
        }
        
    def __len__(self):
        return len(self.image_address["yes"]) + len(self.image_address["no"])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if idx < len(self.image_address["no"]):
            image = cv2.imread(self.image_address["no"][idx], cv2.IMREAD_GRAYSCALE)
            label = 0
        else:
            image = cv2.imread(self.image_address["yes"][idx - len(self.image_address["no"])], cv2.IMREAD_GRAYSCALE)
            label = 1
            
        return torch.tensor(image), torch.tensor(label)
        