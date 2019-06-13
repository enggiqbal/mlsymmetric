import pandas as pd
import numpy as np
import os
import shutil
from random import sample
from sklearn.model_selection import train_test_split 
def copyfilesToFolder(fromfolder,files,  toFolder):
   # os.system("rm -rf " + toFolder )
    if not os.path.exists(toFolder):
        os.makedirs(toFolder)
    for x in files:  
        if (os.path.isfile(fromfolder+x)):
            shutil.copy(fromfolder+x, toFolder)

data_class=["sym", "nonsym"]
 


 



tr=8000
tst=1000
val=1000

files_names=   os.listdir('dataset/esample2_sym/')
testandval = 2000
a, X_valid, y_train, y_valid = train_test_split(files_names, files_names, test_size=testandval, random_state=56741)
testandval = 1000
b, X_valid, y_train, c = train_test_split(X_valid,X_valid, test_size=testandval, random_state=56741)
copyfilesToFolder('dataset/esample2_sym/',b,  'dataset/esample/valid/sym/')
copyfilesToFolder( 'dataset/esample2_sym/',c,  'dataset/esample/test/')
copyfilesToFolder('dataset/esample2_sym/',a,  'dataset/esample/train/sym/')




tr=8000
tst=1000
val=1000

files_names=   os.listdir('dataset/esample2_nonsym/')
testandval = 2000
a, X_valid, y_train, y_valid = train_test_split(files_names, files_names, test_size=testandval, random_state=56741)
a=sample(a,tr)
testandval = 1000
b, X_valid, y_train, c = train_test_split(X_valid,X_valid, test_size=testandval, random_state=56741)
copyfilesToFolder('dataset/esample2_nonsym/',b,  'dataset/esample/valid/nonsym/')
copyfilesToFolder( 'dataset/esample2_nonsym/',c,  'dataset/esample/test/')
copyfilesToFolder('dataset/esample2_nonsym/',a,  'dataset/esample/train/nonsym/')

