# mlsymmetric
 

### folder structure 
 
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
    │    │   ├──drawing                 # for chart/learning curive processing
    │    │   ├──models                  # for storing best train model
    │    │   ├──output                  # traing log
    │    │   ├──st_out                  # standard console output



# models number 
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

#prediction 
 
```console
$ python3 multi_deploy.py [modelNumber] [expname] [datasetname] [class_list_comma_sep] 
$ python3 multi_deploy.py 8 multi hvttraindata H,R,T,V
```



## Team
Iqbal Hossain, 
Felice De Luca, and 
Stephen Kobourov
University of Arizona
