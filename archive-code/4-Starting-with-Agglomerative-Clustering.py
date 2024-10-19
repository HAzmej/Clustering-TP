import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = '../clustering-benchmark-master/src/main/resources/datasets/artificial/'
name="complex9.arff"

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

plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)#
plt.show()

link=['ward']

### FIXER la distance
# 
for x in link:
    xx=[]
    ch=[]
    db=[]
    ss=[]

    seuil_dist=14

    tr=[]
    max=0
    max1=-1
    min=4000
    
    while (seuil_dist<15):
        tps1 = time.time()
        
        model = cluster.AgglomerativeClustering(linkage=x, n_clusters=3)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        # Nb iteration of this method
        #iteration = model.n_iter_
        k = model.n_clusters_
        leaves=model.n_leaves_
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Clustering agglomératif ("+x+", distance_treshold= "+str(seuil_dist)+") "+str(name))
        plt.show()

        d=metrics.davies_bouldin_score(datanp,labels)
        c=metrics.calinski_harabasz_score(datanp,labels)
        s=metrics.silhouette_score(datanp,labels)

        db.append(metrics.davies_bouldin_score(datanp,labels))
        ch.append(metrics.calinski_harabasz_score(datanp,labels))
        ss.append(metrics.silhouette_score(datanp,labels))

        if (max<c):
            max=c
        
        if (max1<s):
            max1=s

        if (min >d):
            min=d

        print("nb clusters =",k,", nb feuilles = ", leaves," runtime = ", round((tps2 - tps1)*1000,2),"ms : ",seuil_dist)
        
        tr.append(round((tps2-tps1)*100,2))
        # print("ch=",ch)
        xx.append(seuil_dist)

        ###
        # FIXER le nombre de clusters
        ###
       
        
        seuil_dist+=1
  

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



#  for k in range(2,12):
#     tps1 = time.time()
#     model = cluster.AgglomerativeClustering(linkage=x, n_clusters=k)
#     model = model.fit(datanp)
#     tps2 = time.time()
#     labels = model.labels_
#             # Nb iteration of this method
#             #iteration = model.n_iter_
#     kres = model.n_clusters_
#     leaves=model.n_leaves_
#             #print(labels)
#             #print(kres)

#     ch=metrics.davies_bouldin_score(datanp,labels)
#     cal_has.append(ch)
#             # print("ch=",ch)
#     xx.append(seuil_dist)
            
#             # plt.scatter(f0, f1, c=labels, s=8)
#             # plt.title("Clustering agglomératif ("+x+", n_cluster= "+str(k)+") "+str(name))
#             # plt.show()
#             # print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")
# #######################################################################