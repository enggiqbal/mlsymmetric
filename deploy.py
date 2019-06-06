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


#run python3 deploy.py num_class modelnumber checkpoint_dir test_img_dir out_csv
#python3 deploy.py 2 8 outputs/binary/models/ testdataset/imagesWelsh/ testdataset/imagesWelsh/val.csv
#python3 deploy.py 4 8 outputs/multi/models/ testdataset/wild/frominternet/ testdataset/wild/frominternet_multi.csv

start=timer()

#classes=int(sys.argv[1])#2

#i=8#int(sys.argv[1])-1

#wild_path = 'esample/manualtest'
#wild_path =sys.argv[1]

#validation_data_dir =  'esample/valid'
 
#checkpoint_dir="outputs/binary/models/"

#validation_data_dir='binary_data/test/'
#10_InceptionResNetV2_checkpoint.best.hdf5

classes=int(sys.argv[1]) #2
i=int(sys.argv[2]) #8
checkpoint_dir=sys.argv[3]
wild_path= sys.argv[4]
outfile=sys.argv[5]


calculatepercentage=0
input_shape=(200,200,1)
img_width, img_height = 200, 200

batch_size=1
V_batch_size=32


if classes==2:
	
	target_names = ['nonsym', 'sym']
	target_names = ['0', '1']
else:
	target_names = ['V', 'H', 'T', 'R']#[k for k in validation_generator.class_indices]



names=['ResNet50',   'MobileNet', 'MobileNetV2', 'NASNetMobile', 'NASNetLarge', 'VGG16', 'VGG19', 'Xception', 'InceptionResNetV2', 'DenseNet121', 'DenseNet201']


m=[keras_applications.ResNet50, keras_applications.MobileNet,  keras_applications.MobileNetV2, keras_applications.NASNetMobile, keras_applications.NASNetLarge, keras_applications.VGG16, keras_applications.VGG19, keras_applications.Xception, keras_applications.InceptionResNetV2,
keras_applications.DenseNet121,keras_applications.DenseNet201]


modelname= str(i)+"_"+ names[i]
print(modelname)

model=m[i](weights=None, input_shape=input_shape,classes=classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(checkpoint_dir+modelname+'_checkpoint.best.hdf5')
wild_datagen = ImageDataGenerator(rescale = 1. / 255)
'''
validation_generator = wild_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = V_batch_size,  class_mode='categorical',color_mode="grayscale")

'''
wild_generator = wild_datagen.flow_from_directory(wild_path  ,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, classes=None, class_mode='categorical',color_mode="grayscale")


#model.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size, epochs = epochs, validation_data = validation_generator, validation_steps = nb_validation_samples // batch_size )
#import pdb; pdb.set_trace()
steps =  np.ceil(wild_generator.samples/batch_size)
Y_pred = model.predict_generator(wild_generator, steps=steps)

#steps =  np.ceil(wild_generator.samples/V_batch_size)
#steps= wild_generator.samples
#loss, acc = model.evaluate_generator(wild_generator, steps=steps, verbose=0)

#print(loss, acc)
y_pred = np.argmax(Y_pred, axis=1)

#print('Confusion Matrix')

wildclass=[target_names[i] for  i in  y_pred]
print(wildclass)
print(wild_generator.filenames)
print(modelname)

data=open(outfile,"a+")
for i in range(0, len(y_pred)):
	a=wild_generator.filenames[i] + ","+ wildclass[i]+ ","+ str( Y_pred[i][1]) + "\n"
	#print(a)
	data.write(a)

data.close()


def findpercentage(wildclass,o):
    sum=0
    for i in range(0, len(o)):
        if wildclass[i]==o[i]:
            sum=sum+1
            #print ("match",wildclass[i],o[i] )
    return sum/len(o)

if calculatepercentage==1:
	o=[a.split("-")[1] for a in wild_generator.filenames]
	print(" percentage", findpercentage(wildclass,o) )

#oa=zip(wildclass,0)
#print(tuple(oa))
#import pdb; pdb.set_trace()
