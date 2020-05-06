from keras.preprocessing.image import ImageDataGenerator
import keras.applications as  keras_applications
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import sys
from timeit import default_timer as timer
from keras import backend as K
from keras.callbacks import CSVLogger
import json
from sklearn.metrics import confusion_matrix

import argparse
# from colorama import Fore, Style, init

module_name = "mlsymmetric"

parser = argparse.ArgumentParser(description = "mlsymmetric")
parser.add_argument('model_num', 
                        metavar = 'MODEL_NUMBER', type=int,
                        help='Model number')
parser.add_argument('exp_name', 
                        metavar = 'EXP_NAME', type=str,
                        help='Name of experiment directory')
parser.add_argument('dataset_name', 
                        metavar = 'DATASET_NAME', type=str,
                        help='Dataset directory')
parser.add_argument('classes', 
                        metavar = 'CLASSES', type=str,
                        help='List of classes')
parser.add_argument('-ver', '--version',
                        action='version',  version="1.0",
                        help='Display version information and dependencies.')
parser.add_argument('-nocol', '--nocolor',
                        action='store_true', default = False, 
                        help='Disables color in terminal')
detail = parser.add_mutually_exclusive_group()
detail.add_argument('-q', '--quiet',
                        action='store_true', 
                        help='Print quiet')
detail.add_argument('-v', '--verbose',
                        action='store_true', 
                        help='Print verbose')
args = parser.parse_args()

rootoutput='outputs/'
rootdataset='dataset/'

model_num = args.model_num - 1
exp_name = args.exp_name
dataset_name = args.dataset_name
class_list = args.classes.split(',')
n_classes = len(class_list)

test_path = rootdataset + dataset_name + "/test/"
checkpoint_dir = rootoutput + exp_name + "/models/"

'''
if not args.nocolor:
    init()
'''

if not args.quiet:        
    print("Test Path:", test_path)
    print("Checkpoint Directory:", checkpoint_dir)
    print("Classes:", class_list)


start = timer()

# When is this used?
calculatepercentage = 0

input_shape = (200,200,1) 
img_width, img_height = 200, 200

V_batch_size=32

names = [
        'ResNet50',
        'MobileNet',
        'MobileNetV2',
        'NASNetMobile',
        'NASNetLarge',
        'VGG16',
        'VGG19',
        'Xception',
        'InceptionResNetV2',
        'DenseNet121',
        'DenseNet201'
        ]

models = [
    keras_applications.ResNet50, 
    keras_applications.MobileNet,  
    keras_applications.MobileNetV2, 
    keras_applications.NASNetMobile, 
    keras_applications.NASNetLarge, 
    keras_applications.VGG16, 
    keras_applications.VGG19, 
    keras_applications.Xception, 
    keras_applications.InceptionResNetV2,
    keras_applications.DenseNet121,
    keras_applications.DenseNet201
    ]

model_name = str(model_num)+ "_" + names[model_num]

if not args.quiet:
    print("Model:", model_name)

model = models[model_num](weights = None, input_shape = input_shape, classes = n_classes)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

weightfile = checkpoint_dir + model_name + '_checkpoint.best.hdf5'
if not args.quiet:
    print("loading model", weightfile)
    
model.load_weights(weightfile)

# When is wild_datagen used?
wild_datagen = ImageDataGenerator(rescale = 1. / 255)

test_datagen = ImageDataGenerator(rescale = 1. / 255)
test_generator = test_datagen.flow_from_directory(test_path, classes = class_list, 
                target_size = (img_width, img_height), batch_size = V_batch_size, shuffle = False,
                class_mode = 'categorical', color_mode = "grayscale")

steps =  np.ceil(test_generator.samples / V_batch_size)
Y_pred = model.predict_generator(test_generator, steps = steps)

acclasses = test_generator.classes[test_generator.index_array]
y_pred = np.argmax(Y_pred, axis = -1)
print(model_name, "acc percentage", sum(y_pred == acclasses)/len(Y_pred))
if not args.quiet:
    print(confusion_matrix(acclasses, y_pred))

file_names = np.array(test_generator.filenames)

name_nums = np.zeros(file_names.size, dtype = [('names', 'U30'), ('y_pred', int), ('acclasses', int)])
name_nums['names'] = file_names
name_nums['y_pred'] = y_pred
name_nums['acclasses'] = acclasses

np.savetxt('name_pred_acc.csv', name_nums, delimiter = ',', header = "File Name\t\tP. Class\tA. Class", fmt = "%s\t%i\t\t%i")
print("name_pred_acc.csv has been created.")