# Symmetry Detection and Classification in Drawings of Graphs

This source depends on  `keras`, `sklearn`,`pandas`, `numpy`,`matplotlib` python packages. The following [docker] (https://www.docker.com/) image includes all the installation and environment. 

```console
$ docker pull hossain/gdocker
```

If you use singularity in high performance computing system (HPC) at the University of Arizona you can load and pull singularity image as follows:

```shell
$ module load singularity
$ singularity pull --name gdocker.simg docker://hossain/gdocker
```
Note that if all of the required packages are installed in your local computer then you don't need docker and singularity. To know more about singularity please visit (https://singularity.lbl.gov/quickstart).
# folder structure 
 
    .
    ├── dataset
    │   ├── datasetname 
    │   │  ├──train
    │   │  │    ├──class1
    │   │  │    ├──class2
    		....
    │   │  ├──valid
    │   │  │    ├──class1
    │   │  │    ├──class2
    		....
    │   │  ├──test
    │   │  │    ├──class1
    │   │  │    ├──class2
    		....            

    .
    ├──outputs
    │    ├── expname 
    │    │   ├──drawing                 # for chart/learning curve processing
    │    │   ├──models                  # for storing best train model
    │    │   ├──output                  # training log
    │    │   ├──st_out                  # standard console output



# supported models 
```console
1 'ResNet50', 
2 'MobileNet', 
3 'MobileNetV2', 
4 'NASNetMobile',
5 'NASNetLarge', 
6 'VGG16', 
7 'VGG19', 
8 'Xception', 
9 'InceptionResNetV2', 
10 'DenseNet121',
11 'DenseNet201'
```

# Training 

```console
$ python3 multi_application_train.py [modelNumber] [expname] [datasetname][list_of_classes_comma_sep] 
$ python3 multi_application_train.py 1 binary esample nonsym,sym

```

# prediction 
 
```console
$ python3 multi_deploy.py [modelNumber] [expname] [datasetname] [class_list_comma_sep] 
$ python3 multi_deploy.py 8 multi hvttraindata H,R,T,V

```
```console
$ python multi_deploy.py --help
usage: multi_deploy.py [-h] [-ver] [-nocol] [-q | -v]
                       MODEL_NUMBER EXP_NAME DATASET_NAME CLASSES

mlsymmetric

positional arguments:
  MODEL_NUMBER       Model number
  EXP_NAME           Name of experiment directory
  DATASET_NAME       Dataset directory
  CLASSES            List of classes

optional arguments:
  -h, --help         show this help message and exit
  -ver, --version    Display version information and dependencies.
  -nocol, --nocolor  Disables color in terminal
  -q, --quiet        Print quiet
  -v, --verbose      Print verbose
```

## Team
Iqbal Hossain, 
Felice De Luca, and 
Stephen Kobourov
University of Arizona


### citation 
```
@article{de2019symmetry,
  title={Symmetry Detection and Classification in Drawings of Graphs},
  author={De Luca, Felice and Hossain, Md Iqbal and Kobourov, Stephen},
  journal={arXiv preprint arXiv:1907.01004},
  year={2019}
}
```