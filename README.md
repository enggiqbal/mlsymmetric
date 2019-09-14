# mlsymmetric

#folder structure 
dataset\
	\datasetname
		\train\class1
		\train\class2
		\valid\class1
		\valid\class2
		\test\class1
		\test\class2

\outputs\
	\expname
		drawing/   
		models/    
		output/    
		st_out/ 
			


# models number 
0 'ResNet50', 
1 'MobileNet', 
3 'MobileNetV2', 
4 'NASNetMobile',
5 'NASNetLarge', 
6 'VGG16', 
7 'VGG19', 
8 'Xception', 
9 'InceptionResNetV2', 
10 'DenseNet121',
11 'DenseNet201'

#folder structure 

```console
$python3 multi_application_train.py [modelNumber] [expname] [datasetname][list_of_classes_comma_sep] 
$python3 multi_application_train.py 0 binary esample nonsym,sym

```




## Team
Iqbal Hossain, 
Felice De Luca, and 
Stephen Kobourov
University of Arizona
