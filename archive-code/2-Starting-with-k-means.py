"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time
import random

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = '../clustering-benchmark-master/src/main/resources/datasets/artificial/'
name="cluto-t5-8k.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])



# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")

kkk=[]
k=6


score_silh=[]


dav_bould=[]


cal_has=[]


max=0
max1=0
min=4000
while (k<7):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    random.seed(10)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_

    # if (np.abs(inertie-inertieavant)<1): 
    #     boll=False
    #     print("nbre ideal=", k-1)
    centroids = model.cluster_centers_
        
    plt.figure(figsize=(6, 6))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
    plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
    #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    sc=metrics.silhouette_score(datanp,labels)
    score_silh.append(sc)

    db=metrics.davies_bouldin_score(datanp,labels)
    dav_bould.append(db)

    ch=metrics.calinski_harabasz_score(datanp,labels)
    cal_has.append(ch)

    if (max<ch):
      max=ch
    
    if (max1<sc):
      max1=sc

    if (min >db):
      min=db
    kkk.append(k)
    print(round((tps2-tps1)*1000,2))
    k+=1


print("max CAH=",max)
print("max SS=",max1)
print("min DB=",min)

plt.plot(kkk,cal_has,"o-")
plt.xlabel("min_cluster_size")
plt.ylabel("Indice de Calisnki-Harabasz")
plt.title("Calinski-Harabasz: ")
plt.show()


plt.plot(kkk,dav_bould,"o-")
plt.xlabel("min_cluster_size")
plt.ylabel("Indice de Davies-Bouldin")
plt.title("Davies-Bouldin: ")
plt.show()

plt.plot(kkk,score_silh,"o-")
plt.xlabel("min_cluster_size")
plt.ylabel("Indice de Silhouette-Score")
plt.title("Silhouette-Score: ")
plt.show()
