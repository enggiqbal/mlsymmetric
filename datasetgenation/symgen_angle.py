
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.lines as mlines
import math 
def slope(a,b):
    return (a[1]-b[1])/(a[0]-b[0])
filename="{}-X-XX-1-0-{}.png"
rootpath='../dataset/angledataset/'

for i in range(0,100):
    a=np.random.randint(3,9) /10 
    b=np.random.randint(3,9) /10 
    ppoint=np.array([[0.1,0.1],[a,b]])
    plt.figure(figsize=(2,2))
    plt.axis('off')
    plt.plot(np.transpose(ppoint)[0],np.transpose(ppoint)[1], c='black', marker = 'o')
    plt.ylim(0, 1)
    plt.xlim(0,1)
    s=int( math.degrees(  math.atan( slope(ppoint[0],ppoint[1]))))
    fname=filename.format(i,s)
    plt.savefig(rootpath+fname)