import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering



path = '../clustering-benchmark-master/src/main/resources/datasets/artificial/'
name="R15.arff"

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



### FIXER la distance
# 

tps1 = time.time()
seuil_dist=range(1, 10)
indice_Calinski_Harabasz = []
for s in seuil_dist : 
    model = cluster.AgglomerativeClustering(distance_threshold=s, linkage='single', n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    k = model.n_clusters_
    leaves=model.n_leaves_
    hh=metrics.calinski_harabasz_score(datanp, labels)
    indice_Calinski_Harabasz.append(hh)
    
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Clustering agglomératif (average, distance_treshold= "+str(s)+") "+str(name))
    plt.show()
    print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")
    
plt.plot(seuil_dist, indice_Calinski_Harabasz, marker='o')
plt.xlabel("Seuil Distance")
plt.ylabel("indice_Calinski_Harabasz")
plt.title("Courbe de l'indice_Calinski_Harabasz en fonction du seuil de la distance ")
plt.show()