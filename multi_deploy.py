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

#run python3 deploy.py num_class modelnumber checkpoint_dir test_img_dir out_csv
#python3 deploy.py 2 8 outputs/binary/models/ testdataset/imagesWelsh/ out.csv
#python3 deploy.py 4 8 outputs/multi/models/ testdataset/wild/frominternet/ testdataset/wild/frominternet_multi.csv
#binary
#python3 deploy.py 2 8 outputs/binary/models/ dataset/esample/randomtest/  prediction.csv

#python3 multi_deploy.py [modelNumber] [expname] [datasetname] [class_list_comma_sep]  

#python3 multi_deploy.py 8 multi hvttraindata H,R,T,V
rootoutput='outputs/'
rootdataset='dataset/'
 

ModelNumber=int(sys.argv[1])-1
#classes=int(sys.argv[2])
expprefix=sys.argv[2]
datasetname=sys.argv[3]
classes_name=sys.argv[4].split(",")
classes=len(classes_name)

test_path=rootdataset + datasetname + "/test/"
checkpoint_dir= rootoutput + expprefix+"/models/" # "outputs/multirefrottarAll/models/"

print("test_path", test_path)
print("checkpoint_dir", checkpoint_dir)
print("classes",classes_name)


start=timer()

#classes=int(sys.argv[1])#2

#i=8#int(sys.argv[1])-1

#wild_path = 'esample/manualtest'
#wild_path =sys.argv[1]

#validation_data_dir =  'esample/valid'
 
#checkpoint_dir="outputs/binary/models/"

#validation_data_dir='binary_data/test/'
#10_InceptionResNetV2_checkpoint.best.hdf5
'''
classes=int(sys.argv[1]) #2
i=int(sys.argv[2]) #8
checkpoint_dir=sys.argv[3]
wild_path= sys.argv[4]
outfile=sys.argv[5]

''' 
'''
classes=3
i=8

wild_path= "dataset/hvttraindata/test/"
outfile="prediction.csv"
classes_name=["nonsym","sym"]
#validation_data_dir="dataset/hvttraindata/valid/"
#test_path= "dataset/hvttraindata/test/"
#test_path= "testdataset/wildnatural/test/"
test_path= "dataset/multi_ref_rot_tra/test/"
 
checkpoint_dir="outputs/multirefrottarAll/models/"


classes=int(sys.argv[1]) #2
modelNumber=int(sys.argv[2]) -1 #8
checkpoint_dir=sys.argv[3]
test_path= sys.argv[4]
outfile=sys.argv[5]


#target_names =  ["ref","rot","tar"]
#target_names =  ["H","R","T","V"]
'''

calculatepercentage=0
input_shape=(200,200,1)
img_width, img_height = 200, 200

#batch_size=1
V_batch_size=32

 
#classes_name#=target_names


names=['ResNet50',   'MobileNet', 'MobileNetV2', 'NASNetMobile', 'NASNetLarge', 'VGG16', 'VGG19', 'Xception', 'InceptionResNetV2', 'DenseNet121', 'DenseNet201']


m=[keras_applications.ResNet50, keras_applications.MobileNet,  keras_applications.MobileNetV2, keras_applications.NASNetMobile, keras_applications.NASNetLarge, keras_applications.VGG16, keras_applications.VGG19, keras_applications.Xception, keras_applications.InceptionResNetV2,
keras_applications.DenseNet121,keras_applications.DenseNet201]


modelname= str(ModelNumber)+"_"+ names[ModelNumber]
print(modelname)

model=m[ModelNumber](weights=None, input_shape=input_shape,classes=classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
weigthfile=checkpoint_dir+modelname+'_checkpoint.best.hdf5'
print("loading model",weigthfile)
model.load_weights(weigthfile)
wild_datagen = ImageDataGenerator(rescale = 1. / 255)

'''
validation_generator = wild_datagen.flow_from_directory(    validation_data_dir,   classes= classes_name, target_size =(img_width, img_height),          batch_size = V_batch_size,  class_mode='categorical',color_mode="grayscale")

validation_generator = wild_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = V_batch_size,  class_mode='categorical',color_mode="grayscale")

'''
test_datagen = ImageDataGenerator(rescale = 1. / 255)
test_generator = test_datagen.flow_from_directory(    test_path,   classes= classes_name, target_size =(img_width, img_height),          batch_size = V_batch_size, shuffle=False, class_mode='categorical',color_mode="grayscale")

'''
wild_generator = wild_datagen.flow_from_directory(wild_path  ,
                              target_size =(img_width, img_height),
                     batch_size = V_batch_size, classes=None, shuffle=False, class_mode='categorical',color_mode="grayscale")
'''

#model.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size, epochs = epochs, validation_data = validation_generator, validation_steps = nb_validation_samples // batch_size )
#import pdb; pdb.set_trace()
steps =  np.ceil(test_generator.samples/V_batch_size)
Y_pred = model.predict_generator(test_generator, steps=steps)

acclasses = test_generator.classes[test_generator.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
print(modelname, "acc percentage", sum(y_pred==acclasses)/len(Y_pred))

print(confusion_matrix(acclasses,y_pred))




np.savetxt('pred.csv',y_pred,delimiter=',')
np.savetxt('acclasses.csv',acclasses,delimiter=',')
#np.savetxt('fname.csv',test_generator.filenames,delimiter=',')

#t=np.column_stack(test_generator.filenames,acclasses,y_pred)
#np.savetxt('t.csv',t,delimiter=',')

'''
y_pred = np.argmax(Y_pred, axis=1)
wildclass=[target_names[i] for  i in  y_pred]

#steps =  np.ceil(wild_generator.samples/V_batch_size)
#steps= wild_gener.samples
#loss, acc = model.evaluate_generator(wild_generator, steps=steps, verbose=0)

#print(loss, acc)


#print('Confusion Matrix')


#print(wildclass)
#print(wild_generator.filenames)
print(modelname)


data=open(wild_path+ modelname +"_"+outfile,"w")
data.write("filename_"+modelname+",predictedclass, " + target_names[0] + "-score," + target_names[1]+"-score\n")
for i in range(0, len(y_pred)):
	a=wild_generator.filenames[i] + ","+ wildclass[i]+ ","+ str( Y_pred[i][0]) + ","+ str( Y_pred[i][1]) + "\n"
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
	#import pdb; pdb.set_trace()
	print(" percentage", findpercentage(wildclass,o) )

#oa=zip(wildclass,0)
#print(tuple(oa))
#import pdb; pdb.set_trace()


'''
