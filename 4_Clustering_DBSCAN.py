import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import itertools
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import scanpy as sc
from itertools import product

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



## Import the data
df_data = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/other_data/fit_logratios_good.tab',sep='\t')
XT = df_data.iloc[:,4:].to_numpy()

## Scale the data using Standard Scaler
# A state is represented by values concatenated across all the experiments
# So standardization is done across genes
scaler = StandardScaler()
XTs = scaler.fit_transform(XT)

## PCA with fixed components
PCA1 = PCA(n_components=50)
ZTs = PCA1.fit_transform(XTs)


##
# DBSCAN with one set of parameters
# Fit 1 : eps=50, min_samples=40, n_components=50
# dict_results = {}
for e_i in np.arange(30,80,10):
    print('epsilon : ', e_i)
    # dict_results[e_i]= {}
    ls_n = []
    ls_nout = []
    for n_i in np.arange(10,60,5):
        dbscan1 = DBSCAN(eps=e_i, min_samples=n_i)
        labels = dbscan1.fit_predict(ZTs)
        ls_n.append(n_i)
        ls_nout.append(np.sum(labels==-1))
    plt.plot(ls_n, ls_nout)
plt.show()
##
# dbscan1 = DBSCAN(eps=60, min_samples=5) # for 31 genes
# dbscan1 = DBSCAN(eps=50, min_samples=5) # for 98 genes
dbscan1 = DBSCAN(eps=30, min_samples=15) # for 313 genes
# dbscan1 = DBSCAN(eps=70, min_samples=5) # for 11 genes
# dbscan1 = DBSCAN(eps=50, min_samples=50) # for 124 genes
# dbscan1 = DBSCAN(eps=20, min_samples=12) # for 595 genes
labels = dbscan1.fit_predict(ZTs)
print('Cluster labels obtained :', np.unique(labels))
print('Number of elements classified as noise :', np.sum(labels!=np.median(labels)))
#
# Plot2: 2D Scatter Plot
plt.figure()
for i in np.unique(labels):
    # Get the data corresponding to the cluster
    np_i = ZTs[labels==i,:]
    # Plot the data corresponding to the cluster
    plt.plot(np_i[:, 0], np_i[:, 1],'.')

plt.show()
##
# Plot: 3D Scatter Plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1,projection='3d')

# Main cluster
np_i = ZTs[labels==np.median(labels),:]
ax.scatter(np_i[:, 0], np_i[:, 1],np_i[:, 2],'.', color = '#888888', s=30)
# Outliers
np_i = ZTs[labels!=np.median(labels),:]
ax.scatter(np_i[:, 0], np_i[:, 1],np_i[:, 2],'*', color = ls_colors[2], s=50) #'#00ab41'
# ax.view_init(0,90)
l = ax.legend(['Similar', 'Differential'], title = 'Fitness comparison of gene knockout\nmutant to that of wild type' ,loc='upper center',bbox_to_anchor=[0.5, 1.2], ncol=2)
ax.set_xticks([-50,200])
ax.set_yticks([-50, 50])
ax.set_zticks([-20,60])
ax.set_xlabel('$PC_1$')
ax.set_ylabel('$PC_2$')
ax.set_zlabel('$PC_3$')
for key, spine in ax.spines.items():
    spine.set_visible(False)
ax.grid(which='major', color='#332462')
plt.setp(l.get_title(), multialignment='center')
plt.show()
fig.savefig('DBSCAN_classifier_1.png', transparent=True)

# # Plot: 3D Scatter Plot
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(1,1,1,projection='3d')
# for i in np.unique(labels):
#     # Get the data corresponding to the cluster
#     np_i = ZTs[labels==i,:]
#     # Plot the data corresponding to the cluster
#     ax.scatter(np_i[:, 0], np_i[:, 1],np_i[:, 2],'.')
# # ax.view_init(0,90)
# ax.legend(['Outliers', 'Cluster'])
# # ax.set_xlabel('PC1')
# # ax.set_ylabel('PC2')
# # ax.set_zlabel('PC3')
# plt.show()


## Save the correlation matrix

df_data.loc[labels!=np.median(labels),'locusId'].to_csv(path_or_buf='/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_' + str(np.sum(labels!=np.median(labels))) + '.csv')



##
# ----------------------------------------------------------------------------------------------------------------------
# Analysis done by Aqib - Using his knowledge on single cell RNASeq, wanted to see how scanpy package can be used to
#  learn something about the variables at play across the experiments
# ----------------------------------------------------------------------------------------------------------------------
# dataset is df_data_final
adata = sc.AnnData(XTs,obs=list(df_data.locusId))

## PCA
sc.pp.pca(adata,n_comps=50)
##
# k nearest neighbors
sc.pp.neighbors(adata,n_neighbors=30)
# clustering
sc.tl.leiden(adata, resolution=0.2)
# UMAP
sc.tl.umap(adata)
#
sc.pl.pca(adata,color='leiden')
# ----------------------------------------------------------------------------------------------------------------------
sc.pl.umap(adata, color='leiden')
# ##
# adata.obsp['connectivities'].toarray()

#

n_genes_of_interest = adata.obs['leiden'].shape[0] - np.sum(adata.obs['leiden']=='0')
print('Number of classified genes : ', n_genes_of_interest)
# np.sum(adata.obs['leiden']=='0')/adata.obs['leiden'].shape[0]

df_data.loc[list(adata.obs['leiden'] !='0'),'locusId'].to_csv(path_or_buf='/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/LEIDEN_gene_predictions.csv')


# ----------------------------------------------------------------------------------------------------------------------
## NEW METHOD
# ----------------------------------------------------------------------------------------------------------------------

# Reduce the space of data to its principal components
adata = sc.AnnData(XTs,obs=list(df_data.locusId))
sc.pp.pca(adata,n_comps=50)
ZTs = adata.obsm['X_pca']

# Compute the distances across all the points and plot their distribution
