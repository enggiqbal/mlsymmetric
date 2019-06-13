from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing.image import ImageDataGenerator
import keras.applications as  keras_applications

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta
from sklearn.model_selection import train_test_split
def create_model():
    nb_filters = 8
    nb_conv = 5
    image_size=200
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=( image_size, image_size,1) ) )
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta(),  metrics=['mean_squared_error'])
    return model






outdir="angle_output/"
train_path = 'rotationtraindata/'
#test_path = 'hvttraindata/test'
#validation_data_dir =  'hvttraindata/valid'
checkpoint_dir = "angle_models/"
#classes_name=["H","R","T","V"]
modelname="custome"
img_width, img_height=(200,200)
batch_size=32
nb_epoch=20

train_datagen = ImageDataGenerator(rescale = 1. / 255)
test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(train_path,  classes= [""],   target_size =(img_width, img_height), batch_size = 10000, class_mode='sparse',color_mode="grayscale")
X,_=train_generator.next()
y=[int(f.split("-")[5].replace(".png","")) for f in train_generator.filenames]
cv_size = 1000
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=cv_size, random_state=56741)



model=create_model()
#import pdb; pdb.set_trace()
model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, y_valid) )
predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
model.save_weights(checkpoint_dir+modelname+'_model_saved_weight.h5')
#import pdb; pdb.set_trace()


ev=10
tmp_valid=X_valid[0:ev]
tmp_y=y_valid[0:ev]
error = model.evaluate(x=tmp_valid,y=tmp_y,steps=cv_size , verbose=0)
