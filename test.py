import matplotlib.pyplot as plt
import numpy as np
from model import Model
import helper

images_train, labels_train, images_val, labels_val, images_test, labels_test = helper.load ()

images_train, labels_train = helper.preprocess (images_train, labels_train)
images_val, labels_val = helper.preprocess (images_val, labels_val)
images_test, labels_test = helper.preprocess (images_test, labels_test)

model = Model ()
model.init ()
model.train (images_train, labels_train, images_val, labels_val)
model.evaluate (images_test, labels_test)