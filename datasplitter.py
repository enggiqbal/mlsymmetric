import pandas as pd
import numpy as np
import os
import shutil
from random import sample

def copyfilesToFolder(fromFolder,  toFolder, fraction):

    os.system("rm -rf " + toFolder )
    if not os.path.exists(toFolder):
        os.makedirs(toFolder)

    files_names=   os.listdir(fromFolder)
    files_names=sample(files_names, int(fraction * len(files_names)))

    for x in files_names:
        full_file_name = os.path.join(fromFolder, x)
        if (os.path.isfile(full_file_name)):
            shutil.move(full_file_name, toFolder)

data_class=["H", "R","T","V"]
for x in data_class:
    fromFolder="hvttraindata/train/" + x
    toFolder="hvttraindata/valid/" +x
    fraction=0.2
    files_names= os.listdir(fromFolder)
    copyfilesToFolder(fromFolder,  toFolder, fraction)


    fromFolder="hvttraindata/train/" + x
    toFolder="hvttraindata/test/" +x
    fraction=0.2
    files_names= os.listdir(fromFolder)
    copyfilesToFolder(fromFolder,  toFolder, fraction)
