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
import keras.applications as  keras_applications

theader='''
     \\begin{table}[h!]
     \\begin{center}
     \\begin{tabular}{ | p {1.5cm}  |c |c| }
     \\hline
'''
tfooter='''
      \\end{tabular}
      \\caption{ Analysis}
      \\label{tbl:pic}
      \\end{center}
      \\end{table}
'''


names=['ResNet50',   'MobileNet', 'MobileNetV2', 'NASNetMobile', 'NASNetLarge', 'VGG16', 'VGG19', 'Xception', 'InceptionResNetV2', 'DenseNet121', 'DenseNet201']


m=[keras_applications.ResNet50, keras_applications.MobileNet,  keras_applications.MobileNetV2, keras_applications.NASNetMobile, keras_applications.NASNetLarge, keras_applications.VGG16, keras_applications.VGG19, keras_applications.Xception, keras_applications.InceptionResNetV2,
keras_applications.DenseNet121,keras_applications.DenseNet201]


def getParameterdetails(logdir,outdir, cls):
    global names
    global m
    global tfooter
    input_shape=(200,200,1)
    if cls=="binary":
        classes=2
    else:
        classes=4

    theader='''
         \\begin{table}[h!]
         \\begin{center}
         \\begin{tabular}{ | l  |c |c|  c  |c |c| c  |c |c|}
         \\hline
    '''
    header=    '''
       \\multicolumn{3}{|c}{Model} & \\multicolumn{3}{|c|}{binary classification} &    \\multicolumn{3}{c|}{multi classification}  \\\   \\hline \n
           Name & parameters & layers & train accuracy & valid accuracy & training time & train accuracy & valid accuracy & training time\\\ \\hline \n
         '''
    data=theader + header
    for i in range( 0, len(names)):

        modelname= str(i)+"_"+ names[i]
        print(modelname)
        b_history=pd.read_csv("output/"+ modelname+ 'log.csv', sep=";")
        b_val_acc=round(max(b_history['val_acc']),2)
        b_tr_acc=round(max(b_history['acc']),2)
        m_history=pd.read_csv("multi_output/"+ modelname+ 'log.csv', sep=";")
        m_val_acc=round(max(m_history['val_acc']),2)
        m_tr_acc=round(max(m_history['acc']),2)

        #data=data + str(tr_acc) + " & " +str(val_acc) + " \\ \n"
        model=m[i](weights=None, input_shape=input_shape,classes=classes)

        p=model.count_params()
        p=str(round(p / 1000000,2)) + 'M'

        l=len(model.layers)
        b_trtime=""
        m_trtime=""

        data=data +  names[i] + " & " + str(p) + " & " + str(l) + " & " + str(b_tr_acc) + " & " +str(b_val_acc) +  " & " +str(b_trtime)+ " & " + str(m_tr_acc) + " & " +str(m_val_acc) +  " & " +str(m_trtime) + "\\\ \\hline \n"
    data=data+ tfooter
    print(data )
    f=open(cls+"_summary.txt","w")
    f.write(data)
    f.close()

    return ""




def draw(logdir,outdir, cls):
    global theader
    global tfooter
    global names


    latexcode=open(outdir+   cls+ "_results.tex","w")
    tbody=theader
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.set_size_inches(8,11)
    #fig.ylabel('accuracy')
    #fig.xlabel('epoch')

    for i in range( 0, len(names)):
        modelname= str(i)+"_"+ names[i]
        print(modelname)

        history=pd.read_csv(logdir+ modelname+ 'log.csv', sep=";")

        #fig = plt.figure()
        plt.subplot( 5,3, i+1)

        acc,=plt.plot(history['acc'])
        val_acc,=plt.plot(history['val_acc'])
        plt.title(names[i])
        #plt.title(names[i]+' model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        f1= modelname+"_accuracy.png"
        #plt.savefig(outdir+f1)

        f2=modelname+"_los.png"
        '''
        # summarize history for loss
        fig = plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title(names[i]+' model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.savefig(outdir+f2)
        '''

        rec= names[i] +" &  \\includegraphics[width=0.4\\textwidth]{figures/"+cls+"/"+f1+"} &  \\includegraphics[width=0.4\\textwidth]{figures/"+cls+"/"+f2+"} \\\ \hline "
        tbody=tbody+ rec +"\n"
    fig.legend([acc, val_acc],['train accuracy', 'validation accuracy'], loc='lower right')
    plt.savefig(outdir+cls+"_total.png")
    latexcode.write(tbody)
    latexcode.write(tfooter)
 
logdir="outputs/binary/output/"
outdir="outputs/binary/drawing/"
draw(logdir,outdir, "binary")


logdir="outputs/binarydense/output/"
outdir="outputs/binarydense/drawing/"
draw(logdir,outdir,  "binarydense")



logdir="outputs/multi/output/"
outdir="outputs/multi/drawing/"
draw(logdir,outdir,  "multi")
 


'''
logdir="output/"
outdir="drawing/"
getParameterdetails(logdir,outdir, "binary")
'''

#logdir="multi_output/"
#outdir="multi_drawing/"
#getParameterdetails(logdir,outdir, "multi")
