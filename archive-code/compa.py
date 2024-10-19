import matplotlib.pyplot as plt
import numpy as np

# Algorithmes et leurs scores (exemple)
algorithms = ['k-means', 'Agglomerative_Single','Agglomerative_Complete','Agglomerative_Average','Agglomerative_Ward', 'DBSCAN', 'HDBSCAN']
silhouette_scores = [0.55, -0.59,0.43,0.54,0.54,-0.36, 0.48]
davies_bouldin_scores = [0.6, 1.38, 0.7, 0.9,0.6,1.7,2]
calinski_harabasz_scores = [4.538015,0.00258,3.0396,4.392248,4.421781,0.0914,1.2163]

# Création d'un bar chart comparatif
barWidth = 0.25
r1 = np.arange(len(silhouette_scores))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, silhouette_scores, color='b', width=barWidth, edgecolor='grey', label='Silhouette')
plt.bar(r2, davies_bouldin_scores, color='r', width=barWidth, edgecolor='grey', label='Davies-Bouldin')
plt.bar(r3, calinski_harabasz_scores, color='g', width=barWidth, edgecolor='grey', label='Calinski-Harabasz')

plt.xlabel('Algorithm', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(silhouette_scores))], algorithms)
plt.legend()
plt.show()
