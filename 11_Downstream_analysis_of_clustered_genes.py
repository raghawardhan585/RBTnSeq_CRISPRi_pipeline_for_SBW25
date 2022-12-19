import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.sparse.csgraph import reverse_cuthill_mckee as CM
from scipy.sparse import csr_matrix
import re

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Import the essential genes
dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_125.csv').locusId.to_list()

# Objective
# Use correlation analysis to map genes which have similar functionality.
# So we are looking for very strong correlation values of approx >0.9
# So, we apply a mask of thresholding to classify genes as having -1,0,+1 effect
##
# Import the log fit data
df_data_all_genes = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/other_data/fit_logratios_good.tab',sep='\t')
df_data = df_data_all_genes.loc[df_data_all_genes.locusId.isin(dbscan_genes), :]
df_data.index = df_data.locusId

## Creating the masked correlation plot

CORR_THRESHOLD = 0.8
df_corr = df_data.iloc[:,4:].T.corr()
# df_corr[df_corr>=np.abs(CORR_THRESHOLD)]= 1
# df_corr[df_corr<=-np.abs(CORR_THRESHOLD)]= -1
df_corr[(df_corr<np.abs(CORR_THRESHOLD)) & (df_corr>-np.abs(CORR_THRESHOLD))] = 0
# Converting into sparse matrix form using reverse cuthill mckee algorithm (graph theory)
graph = csr_matrix(df_corr)
ls1 = CM(graph,symmetric_mode=True)
df_corr = df_corr.iloc[ls1,ls1]
plt.figure(figsize=(15,15))
sb.heatmap(df_corr, vmin=0, vmax=1,cmap='BuGn')
plt.xlabel('locusId', fontsize=48)
plt.ylabel('locusId', fontsize=48)
plt.savefig('DBSCAN_Correlation_1.png', transparent=True)
plt.show()
#'PRGn', 'PRGn_r', 'YlGn'
## Cluster information
# Counting the number of clusters
n_clusters = 0
ls_genes_all = list(df_corr.index)
ls_clusters = []
start_index = 0
end_index = 0
for i in range(len(df_corr)-1): # Initiation and Propagation
    print(i, (df_corr.iloc[i,i+1:]==0).all() , (df_corr.iloc[i+1:,i]==0).all())
    if ((df_corr.iloc[i,i+1:]==0).all() & (df_corr.iloc[i+1:,i]==0).all()):
        end_index = i + 1
        # print(start_index, end_index)
        # print(ls_genes_all[start_index:end_index])
        ls_clusters.append(ls_genes_all[start_index:end_index])
        start_index = i + 1
# Termination
end_index = len(df_corr) + 1
ls_clusters.append(ls_genes_all[start_index:end_index])
# Printing the stats:
print('The various clusters are as follows : ')
for i in range(len(ls_clusters)):
    print(i, ' : ', ls_clusters[i])
print('Number of clusters : ', len(ls_clusters))

# Get the information about the genes that were clustered together
df_genes = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/genes', sep='\t')
dict_gene_info = {}
cluster_no = 0
for cluster in ls_clusters:
    # if len(cluster)>1:
    dict_gene_info[cluster_no] = df_genes[df_genes.locusId.isin(cluster)].loc[:,['locusId','desc']]
    print('Cluster Number : ', cluster_no)
    print(dict_gene_info[cluster_no])
    cluster_no+=1

## Analyzing a specific cluster
## Fitness for a selected cluster is plotted and the optimal growth conditions are presented
cluster_no = 0 # 0,10
df_genes_i = dict_gene_info[cluster_no]
ls_genes = list(df_genes_i.locusId)
# Plot the data of those genes
plt.figure(figsize=(20,6))
for i in range(len(ls_genes)):
    gene = ls_genes[i]
    df_i = df_data_all_genes.loc[df_data_all_genes.locusId==gene,:].iloc[:,4:]
    # plt.plot(df_i.to_numpy().reshape(-1),'.', color=ls_colors[i], label=gene)
    plt.plot(df_i.to_numpy().reshape(-1), '.', label=gene)
plt.legend(ncol=5, loc='upper center')
plt.ylim([-10,20])
plt.show()

df_gene_data = df_data_all_genes.loc[df_data_all_genes.locusId.isin(ls_genes),:]
df_gene_data.index = df_gene_data.locusId
df_gene_data = df_gene_data.iloc[:,4:]
condition_scores = df_gene_data.mean(axis=0)
dict_cond_scores = df_gene_data.mean(axis=0).to_dict()
dict_cond_sorted = {k: v for k, v in sorted(dict_cond_scores.items(), key=lambda item: item[1])}
ls_keys_sorted = list(dict_cond_sorted.keys())
N_conditions =20
print('-----------------------------------------------------------------------------------------------')
print(df_genes_i)
print('-----------------------------------------------------------------------------------------------')
print('The top 10 conditions for negative fitness : ')
for i in range(N_conditions):
    print(i+1, ' - ', ls_keys_sorted[i] , '[ ',np.round(dict_cond_sorted[ls_keys_sorted[i]],2), ' ]')
print('-----------------------------------------------------------------------------------------------')

print('The top 10 conditions for positive fitness : ')
for i in range(N_conditions):
    print(i+1, ' - ', ls_keys_sorted[-i-1] , '[ ',np.round(dict_cond_sorted[ls_keys_sorted[-i-1]],2), ' ]')
print('-----------------------------------------------------------------------------------------------')

## Identify the top 5 unique fitness conditions

# Importing the metadata
df_metadata = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/exps', sep='\t')
df_metadata['SetNumber'] = df_metadata.apply(lambda row: re.split('_',row['SetName'])[-1], axis=1)
df_metadata['PrimaryKey'] = df_metadata['SetNumber']+ df_metadata['Index'] #+ ' ' + df_metadata['Description']
df_metadata.fillna(9999, inplace=True)
ls_unique_columns = set(df_metadata.columns[12:-10])
df_gene_data.columns = [re.split(' ', i)[0] for i in df_gene_data.columns]


#
ls_conditions_accounted_for = []
dict_fitness_conditions = {'positive':{}, 'negative':{}}
for criteria in dict_fitness_conditions.keys():
    ls_conditions_all = list(dict_cond_sorted.keys())
    if criteria == 'positive':
        ls_conditions_all = list(dict_cond_sorted.keys())[::-1]
    elif criteria == 'negative':
        ls_conditions_all = list(dict_cond_sorted.keys())
    # Get the top 5 fitness condition names
    ls_fitness_conditions = []
    dict_conditions_for_criteria = {}
    for set_condition in ls_conditions_all:
        set_name = re.split(' ', set_condition)[0]
        condition = set_condition.replace(set_name + ' ','')
        if set_name not in ls_conditions_accounted_for:
            # Condition Name
            current_metadata_condition = df_metadata[df_metadata['PrimaryKey'] == re.split(' ', set_condition)[0]].iloc[0,:]
            if 'Soil=PNNL_Prosser_PlotA_B_20191220' in current_metadata_condition['Description']:
                continue
            condition_name = ''
            for i in range(1,5): # Iterating through input conditions in metadata [Condition_1...4]
                if current_metadata_condition['Condition_' + str(i)] != 9999:
                    condition_name += '; ' + current_metadata_condition['Condition_' + str(i)] + ' ' + str(
                        current_metadata_condition['Concentration_' + str(i)]) + ' ' + str(
                        current_metadata_condition['Units_' + str(i)])
            condition_name = condition_name [2:]
            # Find the replicates' set names
            df_similar_dataframes = df_metadata.loc[(df_metadata[ls_unique_columns] == (
            current_metadata_condition[ls_unique_columns])).all(axis=1), :]
            ls_set_names_curr = list(set(df_similar_dataframes['PrimaryKey']).intersection(set(df_gene_data.columns)))
            ls_conditions_accounted_for += ls_set_names_curr
            # Store the result
            dict_conditions_for_criteria[condition_name] = ls_set_names_curr
            if len(dict_conditions_for_criteria) == 5:
                break
    # Store the result
    dict_fitness_conditions[criteria] = dict_conditions_for_criteria
dict_fitness_conditions

##

# dict_fitness_conditions={'negative':{'Glucose 10mM; Ammonium chloride 10mM; 1x MOPS': ['set13IT001', 'set13IT002', 'set13IT003', 'set13IT004']}}
dict_fitness_conditions={'negative':{'Sucrose 5mM; Ammonium chloride 10mM': ['set22IT088', 'set22IT089', 'set22IT090']}}

# Compute the max min and mean
f,ax = plt.subplots(2,5, sharex=False, sharey=True, figsize=(20,16))

for criteria in dict_fitness_conditions.keys():
    if criteria == 'positive':
        ax_r = 0
        color = '#3DBFC4' #ls_colors[3]
    elif criteria == 'negative':
        ax_r = 1
        color = '#F7756E' #ls_colors[0]
    ax_c=0
    for condition in dict_fitness_conditions[criteria].keys():
        df_data_i = df_gene_data.loc[:, dict_fitness_conditions[criteria][condition]]
        df_data_i['mean'] = df_data_i.mean(axis=1)
        df_data_i['std'] = df_data_i.std(axis=1)
        df_data_i['max'] = df_data_i.max(axis=1)
        df_data_i['min'] = df_data_i.min(axis=1)

        if 'p-Coumaric (C) and Ammonium chloride (N); with MOPS' in condition:
            df_i = df_data_i
            print(df_data_i)
            break
        df_data_i = df_data_i[['mean', 'std', 'max', 'min']]
        # Plot the data
        x_pos = np.arange(df_data_i.shape[0])
        ax[ax_r, ax_c].bar(x_pos, df_data_i['mean'], yerr = df_data_i['std'],  color=color)
        ax[ax_r, ax_c].set_xticks(x_pos)
        ax[ax_r, ax_c].set_xticklabels(list(df_data_i.index), rotation=90)

        # title
        SINGLE_LINE_LIMIT=21
        # ls_title = re.split(' ', condition)
        title = ''
        newline = ''
        for title_i in re.split(' ', condition):
            if title_i == '3-(N-morpholino)propanesulfonic':
                title_i = 'MOPS'
            if len(newline + ' ' + title_i)>SINGLE_LINE_LIMIT:
                title += newline + '\n'
                newline = title_i + ' '
            else:
                newline+= title_i + ' '
        title += newline

        ax[ax_r, ax_c].set_title(title)

        ax_c+=1
    #     break
    # break
ax[0,0].set_ylabel('Positive Fitness Conditions', fontsize=36)
ax[1,0].set_ylabel('Negative Fitness Conditions', fontsize=36)
f.show()












##
# Extract the genes which we suppose have high positive and negative fitness values
ls_genes = []
for i in range(len(ls_clusters)):
    # if 'PFLU_RS10765' in ls_clusters[i]:
    if 'PFLU_RS08675' in ls_clusters[i]:
    # if 'PFLU_RS01900' in ls_clusters[i]:
        ls_genes = ls_clusters[i]
        break
# Plot the data of those genes
plt.figure(figsize=(15,6))
for i in range(len(ls_genes)):
    gene = ls_genes[i]
    df_i = df_data_all_genes.loc[df_data_all_genes.locusId==gene,:].iloc[:,4:]
    # plt.plot(df_i.to_numpy().reshape(-1),'.', color=ls_colors[i], label=gene)
    plt.plot(df_i.to_numpy().reshape(-1), '.', label=gene)
plt.legend(ncol=4, loc='upper center')
plt.ylim([-15,15])
plt.show()


## Extract the required conditions with high positive and high negative fitness condition

df_gene_data = df_data_all_genes.loc[df_data_all_genes.locusId.isin(ls_genes),:]
df_gene_data.index = df_gene_data.locusId
df_gene_data = df_gene_data.iloc[:,4:]
condition_scores = df_gene_data.mean(axis=0)
dict_cond_scores = df_gene_data.mean(axis=0).to_dict()
dict_cond_sorted = {k: v for k, v in sorted(dict_cond_scores.items(), key=lambda item: item[1])}
ls_keys_sorted = list(dict_cond_sorted.keys())
print('-----------------------------------------------------------------------------------------------')
print('The top 10 conditions for negative fitness : ')
for i in range(10):
    print(i+1, ' - ', ls_keys_sorted[i] , '[ ',np.round(dict_cond_sorted[ls_keys_sorted[i]],2), ' ]')
print('-----------------------------------------------------------------------------------------------')

print('The top 10 conditions for positive fitness : ')
for i in range(10):
    print(i+1, ' - ', ls_keys_sorted[-i-1] , '[ ',np.round(dict_cond_sorted[ls_keys_sorted[-i-1]],2), ' ]')
print('-----------------------------------------------------------------------------------------------')


# Spit out the results as a dictionary
dict_out_plus = {}
dict_out_minus = {}
for i in range(10):
    dict_out_plus[i + 1] = {}
    dict_out_plus[i + 1]['Fitness Condition (+)'] = ls_keys_sorted[-i-1]
    for gene in ls_genes:
        dict_out_plus[i + 1][gene] = np.round(df_gene_data.loc[gene,ls_keys_sorted[-i-1]],2)
    # dict_out_plus[i + 1]['Fitness Score (+)'] = np.round(dict_cond_sorted[ls_keys_sorted[-i-1]],2)
    # --------
    dict_out_minus[i + 1] = {}
    dict_out_minus[i + 1]['Fitness Condition (-)'] = ls_keys_sorted[i]
    for gene in ls_genes:
        dict_out_minus[i + 1][gene] = np.round(df_gene_data.loc[gene,ls_keys_sorted[i]],2)
    # dict_out_minus[i + 1]['Fitness Score (-)'] = np.round(dict_cond_sorted[ls_keys_sorted[i]],2)

pd.DataFrame(dict_out_minus).T.to_csv('Negative_OptimalFitness_For_Selected_Strains.csv')
pd.DataFrame(dict_out_plus).T.to_csv('Positive_OptimalFitness_For_Selected_Strains.csv')


## Looking for functional realtionships across genes

df_genes = pd.read_csv('RbTnSeq_data/genes', sep='\t')

df_genes_cluster_info = df_genes.loc[df_genes.locusId.isin(ls_genes)].loc[:,['locusId', 'desc']]

pd.DataFrame(df_genes_cluster_info).to_csv('Info_For_Selected_Genes.csv')


##

