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
name="sizes5.arff"

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
tps1 = time.time()
inert=[]
kkk=[]
k=2
boll= True 
inertieavant=0

score_silh=[]
sc=0

dav_bould=[]
db=0

cal_has=[]
ch=0
while(boll and k<10):
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    random.seed(10)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    print("inertie = ",inertie)
    print("inertie avant=",inertieavant)
    # if (np.abs(inertie-inertieavant)<1): 
    #     boll=False
    #     print("nbre ideal=", k-1)
    centroids = model.cluster_centers_
    
    #plt.figure(figsize=(6, 6))
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
    print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
    #print("labels", labels)

    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(centroids)
    print(dists)
    kkk.append(k)
    inert.append(inertie)
    inertieavant=inertie
    k+=1

plt.plot(kkk,inert,'o-')
plt.show()

plt.plot(kkk,score_silh,"o-")
plt.show()

plt.plot(kkk,dav_bould,"o-")
plt.show()

plt.plot(kkk,cal_has,"o-")
plt.show()