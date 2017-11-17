import matplotlib.pyplot as plt
import glob
import imageio
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

global labels_name
labels_name = None

def load (path='./face-data/final_data/') :
    path = path + '/' if path[-1] != '/' else path
    
    images_train, images_val, images_test = [], [], []
    labels_train, labels_val, labels_test = [], [], []

    global labels_name
    labels_name = []
    for i, path_label in enumerate (glob.glob (path + '*')) :
        label_name = path_label.split ('/')[-1]
        labels_name.append (label_name)

        imgs, labs = [], []
        for path_image in glob.glob (path_label + '/*') :
            imgs.append (imageio.imread (path_image))
            labs.append (i)

        imgs_train, imgs_test, labs_train, labs_test = train_test_split (imgs, labs, test_size=0.2)
        imgs_train, imgs_val, labs_train, labs_val = train_test_split (imgs_train, labs_train, test_size=0.2)

        images_train.append (imgs_train)
        labels_train.append (labs_train)

        images_val.append (imgs_val)
        labels_val.append (labs_val)

        images_test.append (imgs_test)
        labels_test.append (labs_test)
    labels_name = np.array (labels_name)

    images_train, labels_train = np.vstack (images_train), np.hstack (labels_train)
    images_val, labels_val = np.vstack (images_val), np.hstack (labels_val)
    images_test, labels_test = np.vstack (images_test), np.hstack (labels_test)
    
    images_train, labels_train = shuffle (images_train, labels_train)
    images_val, labels_val = shuffle (images_val, labels_val)
    images_test, labels_test = shuffle (images_test, labels_test)

    return images_train, labels_train, images_val, labels_val, images_test, labels_test

def preprocess (images, labels=[]) :
    images_new = images
    # TODO: Some preprocess for images ...
    
    if len (labels) == 0 :
        return images_new
    else :
        n_classes = len (labels_name)
        labels_new = np.zeros ((len (labels), n_classes))
        for i, label in enumerate (labels) :
            labels_new[i, label] = 1.
        
        return images_new, labels_new