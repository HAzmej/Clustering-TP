import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = '../clustering-benchmark-master/src/main/resources/datasets/artificial/'
name="cluto-t5-8k.arff"

#jain.arff

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


# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")

epsilon=7 #2  # 4
min_pts= 6 #10   # 10

xx=[]
db=[]
ch=[]
ss=[]
max=0
max1=-1
min=4000
while (epsilon<9):
    tps1 = time.time()
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)
    print(epsilon,": eps")
    print(metrics.calinski_harabasz_score(datanp,labels))
    print( round((tps2-tps1)*1000,2), "ms") 
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Données après clustering DBSCAN (1) - Epsilon= "+str(epsilon)+" MinPts= "+str(min_pts))
    plt.show()
    d=metrics.davies_bouldin_score(datanp,labels)
    c=metrics.calinski_harabasz_score(datanp,labels)
    s=metrics.silhouette_score(datanp,labels)

    db.append(metrics.davies_bouldin_score(datanp,labels))
    xx.append(epsilon)
    ch.append(metrics.calinski_harabasz_score(datanp,labels))
    ss.append(metrics.silhouette_score(datanp,labels))

    if (max<c):
      max=c
    
    if (max1<s):
      max1=s

    if (min >d):
      min=d
    epsilon+=1

print("max CAH=",max)
print("max SS=",max1)
print("min DB=",min)

plt.plot(xx,ch,"o-")
plt.xlabel("min_cluster_size")
plt.ylabel("Indice de Calisnki-Harabasz")
plt.title("Calinski-Harabasz: ")
plt.show()


plt.plot(xx,db,"o-")
plt.xlabel("min_cluster_size")
plt.ylabel("Indice de Davies-Bouldin")
plt.title("Davies-Bouldin: ")
plt.show()

plt.plot(xx,ss,"o-")
plt.xlabel("min_cluster_size")
plt.ylabel("Indice de Silhouette-Score")
plt.title("Silhouette-Score: ")
plt.show()

  #  datanp2=np.copy(datanp)
    # neigh=NearestNeighbors(n_neighbors=min_pts)
    # neigh.fit(datanp2)
    # dist, i=neigh.kneighbors(datanp2)

    # newdist= np.asarray([np.average(dist[j][1:]) for j in range(0,dist.shape[0])])
     
    # d=np.sort(newdist)

    # plt.title("Plus Proche voisins"+str(min_pts))
    # plt.plot(d)
    # plt.show

    ####################################################
# Standardisation des donnees

# scaler = preprocessing.StandardScaler().fit(datanp)
# data_scaled = scaler.transform(datanp)
# print("Affichage données standardisées            ")
# f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
# f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

# #plt.figure(figsize=(10, 10))
# plt.scatter(f0_scaled, f1_scaled, s=8)
# plt.title("Donnees standardisées")
# plt.show()


# print("------------------------------------------------------")
# print("Appel DBSCAN (2) sur données standardisees ... ")
# tps1 = time.time()
# epsilon=0.05 #0.05
# min_pts=5 # 10
# model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
# model.fit(data_scaled)

# tps2 = time.time()
# labels = model.labels_
# # Number of clusters in labels, ignoring noise if present.
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise = list(labels).count(-1)
# print('Number of clusters: %d' % n_clusters)
# print('Number of noise points: %d' % n_noise)

# plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
# plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))




