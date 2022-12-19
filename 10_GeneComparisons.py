import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22


##
DBSCAN = True
# DBSCAN = False


# import genes from dbscan
# dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_11.csv').locusId.to_list()
# dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_31.csv').locusId.to_list()
# dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_98.csv').locusId.to_list()
# dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_125.csv').locusId.to_list()
# dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_313.csv').locusId.to_list()
dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_595.csv').locusId.to_list()
leiden_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/LEIDEN_gene_predictions.csv').locusId.to_list()
if DBSCAN == True:
    ls_crucial_genes = dbscan_genes
else:
    ls_crucial_genes = leiden_genes
# ls_crucial_genes = dbscan_genes.intersection(leiden_genes)
print('Number of genes from clustering algorithm : ', len(ls_crucial_genes))
## Import the genes from all the conditions

folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/'
ls_out = []
dict_genes = {}
for folder_name_i in os.listdir(folder_name):
    if 'DBSCAN' in folder_name_i:
        continue
    elif 'LEIDEN' in folder_name_i:
        continue
    else:
        if 'compound' in folder_name_i:
            condition_i = folder_name_i[9:]
        elif 'pH' in folder_name_i:
            condition_i = 'pH_'+folder_name_i[3:]
        elif 'Temperature' in folder_name_i:
            condition_i = 'T_'+folder_name_i[12:]
        else:
            condition_i = folder_name_i
        dict_genes[condition_i] = pd.read_csv(folder_name + '/' + folder_name_i, sep='\t').iloc[:,-1].to_list()

## Find the intersection genes
dict_intersection_genes={}
for condition_i in dict_genes.keys():
    dict_intersection_genes[condition_i] = list(set(dict_genes[condition_i]).intersection(set(ls_crucial_genes)))

## Barplot of the gene statistics
ls_exhaustive_genes = []
for condition_i in dict_genes.keys():
    ls_exhaustive_genes += dict_intersection_genes[condition_i]

dict_gene_count = {gene:ls_exhaustive_genes.count(gene) for gene in list(set(ls_exhaustive_genes))}
dict_gene_count = {k: v for k, v in sorted(dict_gene_count.items(), key=lambda item: item[1], reverse=True)}

##
plt.figure(figsize=(10,8))
plt.bar(list(dict_gene_count.keys())[0:], list(dict_gene_count.values())[0:])
plt.xticks(rotation=90)
plt.xticks([])
plt.xlabel('Gene LocusId in order of their ranking')
plt.ylabel('Number of occurances out of \n' + str(len(dict_genes.values())) + ' growth conditions')
plt.show()

## Functionalities of the top 10 genes
df_gene_info = pd.read_csv('/Users/shara/Desktop/Mission_Sakura/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/genes',sep='\t')
df_gene_info.index = df_gene_info.locusId
df_gene_info = df_gene_info.loc[list(dict_gene_count.keys())[100:120], ['desc']]
# df_gene_info = df_gene_info.loc[list(dict_gene_count.keys())[20:40], ['desc']]

# df_gene_info = df_gene_info.loc[df_gene_info.locusId.isin(list(dict_gene_count.keys())[0:20]), ['locusId', 'desc']]
df_gene_info['SignificantConditions'] =pd.Series(dict_gene_count)

df_gene_info.to_csv('RankedGenes.csv')

## Plot of the various selected genes


df_fit = pd.read_csv('/Users/shara/Desktop/Mission_Sakura/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/other_data/fit_logratios_good.tab',sep='\t')

f, ax = plt.subplots(4,5, sharex=True, sharey=True, figsize=(20,12))
ax = ax.reshape(-1)
for i in range(20):
    df_i = df_fit.loc[df_fit.locusId==list(df_gene_info.index)[i],:].iloc[:,4:]
    ax[i].plot([0,len(df_i.to_numpy()[0])],[0,0], color='#000000', linewidth=3)
    ax[i].plot(df_i.to_numpy()[0],'.', alpha=0.5, color='#AAAAAA')
    ax[i].set_xticks([0,500,1000])
    ax[i].set_title(list(df_gene_info.index)[i])
    ax[i].set_yticks([-15,0,15])


plt.show()

## Plot of the number of intersection genes and the total number of

ls_n_total_genes = []
ls_n_intersection_genes = []
for condition_i in dict_genes.keys():
    ls_n_total_genes.append(len(dict_genes[condition_i]))
    ls_n_intersection_genes.append(len(dict_intersection_genes[condition_i]))
ls_x = list(range(len(ls_n_total_genes)))
#
plt.figure(figsize=(10,8))
plt.bar(ls_x, ls_n_total_genes)
plt.bar(ls_x, ls_n_intersection_genes)
plt.ylim([0,140])
plt.xlabel('Condition number')
plt.ylabel('Total number of genes')
plt.legend(['Co-fit genes', 'Clustered genes'], ncol=2,loc='upper center')
plt.show()


##
ls_y = np.array(ls_n_intersection_genes)/np.array(ls_n_total_genes)*100
plt.figure(figsize=(10,8))
plt.bar(ls_x, ls_y, color='#AAAAAA')
# plt.bar(ls_x, ls_n_intersection_genes)
plt.ylim([0,120])
plt.xlabel('Condition number')
plt.ylabel('% of gene intersections')
plt.show()










