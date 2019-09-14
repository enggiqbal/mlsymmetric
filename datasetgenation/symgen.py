
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.lines as mlines

def sym(n,e):       
    G = nx.erdos_renyi_graph(int(n/2),e)
    pos=nx.spring_layout(G)
    gg=G.copy()
    shifting=0.5
    numberofparalleledge=int(n * 0.1)
    crossingpair=int(n * 0.1)

    for x in G.nodes():
        gg.add_node('m'+str(x))
        pos['m'+str(x)]=[ shifting+ (-1)* pos[x][0],pos[x][1]]

    for e in G.edges():
        gg.add_edge('m'+str(e[0]),'m'+str(e[1]))  
    for i in range(0, numberofparalleledge):
        x=random.randint(0,int(n/2)-1)
        gg.add_edge(x,'m'+str(x))

    for i in range(0, crossingpair):
        x=random.randint(0,int(n/2)-1)
        y=random.randint(0,int(n/2)-1)
        if x!=y:
            gg.add_edge(x,'m'+str(y))
            gg.add_edge(y,'m'+str(x))
             

    ppoint=[]
    N=list(gg.nodes())
    for x in range(0,len(N)):
        for y in range(x+1, len(N)):
            ppoint.append([(pos[N[x]][0]+pos[N[y]][0])/2, (pos[N[x]][1]+pos[N[y]][1])/2])

    np.transpose(ppoint)[0]
    drawing(gg,pos,ppoint)
'''
    plt.subplot(121)
    nx.draw(gg, pos=pos, with_labels=True, font_weight='bold')
    plt.subplot(122)
    plt.scatter(np.transpose(ppoint)[0],np.transpose(ppoint)[1],s=10, alpha=0.3)
    plt.show()
'''
def nonsym(n,e):
    gg = nx.erdos_renyi_graph(n,e)
    pos=nx.spring_layout(gg)
    ppoint=[]
    N=list(gg.nodes())
    for x in range(0,len(N)):
        for y in range(x+1, len(N)):
            ppoint.append([(pos[N[x]][0]+pos[N[y]][0])/2, (pos[N[x]][1]+pos[N[y]][1])/2])

    np.transpose(ppoint)[0]
    drawing(gg,pos,ppoint)
def slope(a,b):
    return (a[1]-b[1])/(a[0]-b[0])

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def drawing(gg,pos,ppoint):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
#    nx.draw(gg, pos=pos, with_labels=True, font_weight='bold')
    nx.draw(gg, pos=pos, node_size=25,with_labels=False, font_weight='bold')
    plt.subplot(122)
    plt.scatter(np.transpose(ppoint)[0],np.transpose(ppoint)[1],s=10, alpha=0.1)


    plt.show()
    return 0

    E=list(gg.edges())
    for i in range(0, len(E)):
        for j in range(i+1, len(E)):

            a=pos[E[i][0]] 
            b=pos[E[i][1]] 
            c=pos[E[j][0]] 
            d=pos[E[j][1]] 
            s1=slope(a,b)
            s2=slope(c,d)
            s,t=line_intersection((a,b),(c,d))

            m=(s1+s2)/2
            print(E[i], E[j],  m, (s,t))


            x1, y1 = [0, t-m*s] 
            x2, y2 = [(s-t/m), 0]
            newline([x1,y1],[x2,y2])
#            plt.xlim(-1, 1), plt.ylim(-1, 1)
#            plt.plot(x1, y1, x2, y2, marker = 'o')

            



        
            


     
    plt.show()


sym(120,0.2)
nonsym(120,0.1)


#sym(10,0.2)
#nonsym(10,0.2)

#sym(20,0.5)
#nonsym(20,0.5)