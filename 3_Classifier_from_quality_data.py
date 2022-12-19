import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import itertools
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

## Import the data
df_data = pd.read_csv('RbTnSeq_data/other_data/fit_logratios_good.tab',sep='\t')
XT = df_data.iloc[:,4:].to_numpy()

## Scale the data using Standard Scaler
# A state is represented by values concatenated across all the experiments
# So standardization is done across genes
scaler = StandardScaler()
XTs = scaler.fit_transform(XT)

## Principal Component Analysis
PCA0 = PCA(n_components=XTs.shape[1])
ZTs = PCA0.fit_transform(XTs)
print('Dimension of the PCA transformed dataset : ', np.shape(ZTs))
S = PCA0.singular_values_**2
S = np.concatenate([np.array([0]),S],axis=0)
##
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(S)/np.sum(S))
plt.plot([50,50], [0,1])
plt.xlabel('Number of principal components')
plt.ylabel('Energy captured in covariance matrix')
plt.ylim([0,1])
# plt.plot(range(1,len(S)+1),S)
# plt.xscale('log')
plt.show()
##
fig = plt.figure(figsize = (15,8))
ax = fig.add_subplot(1,2,1,projection='3d')
ax.scatter(ZTs[:,0], ZTs[:,1], ZTs[:,2], '.')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
# plt.show()

PCA_disp2d = PCA(n_components=2)
ZTs_2d = PCA_disp2d.fit_transform(XTs)
ax2 = fig.add_subplot(1,2,2)
ax2.plot(ZTs[:,0],ZTs[:,1], '.')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
plt.show()

## PCA with fixed components
PCA1 = PCA(n_components=100)
ZTs = PCA1.fit_transform(XTs)



## Clustering: Elbow plot method
k_range = list(range(1,10,1))*10
SSE = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(ZTs)
    SSE.append(km.inertia_)
plt.plot(k_range, SSE,'.')
plt.xticks(k_range)
plt.show()




## Clustering
# Clustering parameters
n_clusters = 3

# K means clustering
km1 = KMeans(n_clusters=n_clusters)
y_predicted = km1.fit_predict(ZTs)

# Separate each cluster into a different dataframe
dict_cluster_data = {}
for i in range(n_clusters):
    dict_cluster_data[i] = ZTs[y_predicted == i,:]
np_centers = km1.cluster_centers_
# ----------------------------
# Plot Results
# ----------------------------
# Plot1: 3D Scatter Plot
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
for i in range(n_clusters):
    ax1.scatter(dict_cluster_data[i][:,0], dict_cluster_data[i][:,1], dict_cluster_data[i][:,2],'.', color=ls_colors[i], alpha=0.1)
    ax1.scatter(np_centers[i,0],np_centers[i,1],np_centers[i,2],'*', color=ls_colors[i], linewidths=5)
plt.show()
#
# Plot2: 2D Scatter Plot
plt.figure()
for i in range(n_clusters):
    plt.plot(dict_cluster_data[i][:, 0], dict_cluster_data[i][:, 1],'.', alpha=0.1)
    plt.plot(np_centers[i,0],np_centers[i,1],'*', color=ls_colors[i], markersize=10)
    # km1.cluster_centers_  - plot these as the centers of the plot
plt.show()


# Plot3: 2D Contour Plot
PCA_2pc = PCA(n_components=2)
PCA_2pc.fit(XTs)
xc = np.linspace(np.min(ZTs[:,0]), np.max(ZTs[:,0]), 100)
yc= np.linspace(np.min(ZTs[:,1]), np.max(ZTs[:,1]), 10)
Xc,Yc = np.meshgrid(xc,yc)

Zcr_2PC = np.concatenate([Xc.reshape(-1,1), Yc.reshape(-1,1)], axis=1)
Zcr = PCA1.transform(PCA_2pc.inverse_transform(Zcr_2PC))
Zpr = km1.predict(Zcr)

Zc = Zpr.reshape(Xc.shape)

plt.figure()
plt.contourf(Xc,Yc,Zc)
plt.show()

##
silhouette_score()




