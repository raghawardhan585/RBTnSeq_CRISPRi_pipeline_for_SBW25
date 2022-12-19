# Objective
# For a given gene of interest in the RbTnSeq data, we find how to compute the


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from copy import deepcopy
import re
import sys



plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22


##
# Import the essential genes
dbscan_genes = pd.read_csv('/Users/shara/Box/YeungLabUCSBShare/Shara/_Project_Naruto/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/gene_predictions/DBSCAN_gene_predictions_125.csv').locusId.to_list()

# Import the strain information
df_data_raw = pd.read_csv('RbTnSeq_data/all.poolcount', sep='\t')

##
gene_of_interest = 'PFLU_RS28535'
# gene_of_interest = 'PFLU_RS25640'

# Get the data only pertaining to the gene of interest
df_data_gene_strains_all = df_data_raw[df_data_raw.locusId == gene_of_interest]
# Discard strains with
df_data_gene_strains_all  = df_data_gene_strains_all [(df_data_gene_strains_all.f>0.1) & (df_data_gene_strains_all.f<0.9)]

ls_exhaustive_strains = df_data_gene_strains_all.pos.to_list()
dict_strain_count_all = {gene:ls_exhaustive_strains.count(gene) for gene in list(set(ls_exhaustive_strains))}

# Discard strains that have less than 3 replicates
dict_strain_count =deepcopy(dict_strain_count_all)
for key in dict_strain_count_all.keys():
    if dict_strain_count_all[key]<3:
        dict_strain_count.pop(key)
# Plot the frequency of the strains
plt.figure(figsize=(12,8))
plt.bar(['strain_' + str(i) for i in range(1,1+len(dict_strain_count))], dict_strain_count.values())
plt.ylabel('# Biological Replicates')
plt.yticks(list(range(0,22,2)))
plt.xticks(rotation=90)
plt.show()
# Get the data of the filtered strains
df_data_strain_raw = df_data_gene_strains_all.loc[df_data_gene_strains_all.pos.isin(list(dict_strain_count.keys())),:]
## Zero statistics of the dataset
n_zeros = np.sum(np.sum(df_data_strain_raw.iloc[:,7:]==0))
n_tot = np.sum(np.product(df_data_strain_raw.iloc[:,7:].shape))
print('Number of total zeros : ', n_zeros)
print('Number of entries: ', n_tot)
print('% of zeros : ', n_zeros/n_tot*100)

## Percentage of zeros in each strain

ls_strain_zeros = []
for strain in dict_strain_count.keys():
    df_i = df_data_strain_raw.loc[df_data_strain_raw.pos == strain,:].iloc[:,7:]
    ls_strain_zeros.append( np.sum(np.sum(df_i==0)) / np.product(np.shape(df_i))*100 )

plt.figure(figsize=(12,8))
plt.bar(['strain_' + str(i) for i in range(1,1+len(ls_strain_zeros))], ls_strain_zeros)
plt.ylabel('% of zeros in the data')
plt.yticks(list(range(0,110,20)))
plt.xticks(rotation=90)
plt.show()


##

# Importing the metadata
df_metadata_raw = pd.read_csv('RbTnSeq_data/exps', sep='\t')
# Assign the condition number as 0 where it is nan
df_metadata_raw.loc[df_metadata_raw['Condition Number'].isna(), 'Condition Number'] = 0

# Process the Time 0 samples from data from across all experiment sets
# Get the samples of Time0 - assign nan to 0 values and compute nan mean and only keep 1 time point
df_md_T0 = df_metadata_raw[df_metadata_raw['Condition Number']==0]
ls_T0_samples = list(df_md_T0.SetName + '.' +df_md_T0.Index)
ls_all_samples = list(df_metadata_raw.SetName + '.' + df_metadata_raw.Index)
ls_Tf_samples = list(set(ls_all_samples)-set(ls_T0_samples))

df_T0 = df_data_strain_raw.loc[:,ls_T0_samples]
print('% of 0s in Time0 before nan_mean : ', np.sum(np.sum(df_T0.iloc[:,7:]==0))/np.product(df_T0.iloc[:,7:].shape)*100)
df_T0[df_T0==0] = np.nan
df_T0['mean_T0'] = df_T0.mean(axis=1)
df_T0 = df_T0.loc[:,['mean_T0']]
print('% of 0s in Time0 after nan_mean : ', np.sum(np.sum(df_T0.isna()))/np.product(df_T0.shape)*100)

# Discard the strains that do not have any coverage at Time0
df_data_Tf_raw = df_data_strain_raw.loc[~df_T0['mean_T0'].isna(),ls_Tf_samples]
df_T0 = df_T0.loc[~df_T0['mean_T0'].isna(),:]
df_md_Tf_raw = df_metadata_raw[df_metadata_raw['Condition Number']!=0]
df_data_strain_raw = df_data_strain_raw.loc[df_T0.index,:]

# Average the data across measurement replicates
print('% of zeros in the data before nanmean: ', np.sum(np.sum(df_data_Tf_raw ==0))/np.product(df_data_Tf_raw.shape)*100)
df_data_Tf_raw[df_data_Tf_raw==0]=np.nan
df_md_Tf = deepcopy(df_md_Tf_raw)
df_data_Tf = deepcopy(df_data_Tf_raw)
ls_drop_data_cols = []
ls_drop_metadata_indices = []
for setname in df_md_Tf['SetName'].unique():
    for cond_no in df_md_Tf[df_md_Tf['SetName'] == setname].loc[:, 'Condition Number'].unique():
        # print(setname, cond_no)
        # sys.stdout.write("\033[F")
        ls_reqd_indices = df_md_Tf[(df_md_Tf['SetName'] == setname) & (df_md_Tf['Condition Number'] == cond_no)].loc[:, 'Index'].unique()
        ls_reqd_samples = setname + '.' + ls_reqd_indices
        df_data_Tf.loc[:, ls_reqd_samples[0]] = np.nanmean(df_data_Tf.loc[:, ls_reqd_samples], axis=1)
        # Drop the excess columns in data
        ls_drop_data_cols += list(ls_reqd_samples[1:])
        # Drop the excess rows in metadata
        ls_indices = df_md_Tf[(df_md_Tf['SetName'] == setname) & (df_md_Tf['Condition Number'] == cond_no)].index
        ls_drop_metadata_indices += list(ls_indices[1:])
print('% of zeros in the data : ', np.sum(np.sum(df_data_Tf_raw.isna()))/np.product(df_data_Tf_raw.shape)*100)
df_data_Tf.drop(labels=ls_drop_data_cols, axis=1, inplace=True)
df_md_Tf.drop(labels=ls_drop_metadata_indices, axis=0, inplace=True)
print('% of zeros in the data after nanm: ', np.sum(np.sum(df_data_Tf.isna()))/np.product(df_data_Tf.shape)*100)
## Average the data across the strains (biological replicates)
ls_strainID = list(df_data_strain_raw.pos.unique())
for strainID in ls_strainID:
    df_data_Tf.loc[strainID,:] = df_data_Tf.loc[df_data_strain_raw[df_data_strain_raw['pos'] == strainID].index, :].mean(axis=0)
    df_T0.loc[strainID,:] = df_T0.loc[df_data_strain_raw[df_data_strain_raw['pos'] == strainID].index, :].mean(axis=0)
df_data_Tf = df_data_Tf.loc[ls_strainID,:]
df_T0 = df_T0.loc[ls_strainID,:]
print('% of zeros in the data after strain averaging: ', np.sum(np.sum(df_data_Tf.isna()))/np.product(df_data_Tf.shape)*100)
##
df_LFN = df_data_Tf.divide(df_T0.to_numpy(),axis=1)
sb.heatmap(df_LFN)
plt.show()
##
plt.figure(figsize=(12,8))
plt.ylabel('Euclidean distance from the median trend')
plt.bar( ['strain_' + str(i) for i in range(1,1+len(ls_strain_zeros))],(df_LFN - df_LFN.median(axis=0)).pow(2).sum(axis=1).pow(0.5))
plt.xticks(rotation=90)
plt.show()
##
ls_labels = ['strain_' + str(i) for i in range(1,1+len(ls_strain_zeros))]
sb.heatmap(pd.DataFrame(np.array(df_LFN.T.corr()), index= ls_labels, columns=ls_labels), cmap='RdBu' )
plt.show()
##

# def metadata_preprocessing_base_level(df_metadata_in):
#     # Metadata Preprocessing
#     # ------------------------------------------------------------------------------------------------------
#     # Assumptions
#     # Samples have to be liquid, aerobic and under a defined shaking speed
#     # Samples have a Time0 measurement as only the log2 fitness ratio is used in the algorithm
#     # [REVISIT] Samples with temperature shift (as Group) are discarded
#     # [REVISIT] The volume effects across samples is ignored for now
#     # Parameters currently used in the learning process   -   Temperature, pH, Shaking speed, various conditions
#     # ------------------------------------------------------------------------------------------------------
#     # Copying the metadata so that it can be edited
#     df_metadata_filtered = deepcopy(df_metadata_in)
#     # Drop the set of experiments without Time0 samples ( because they are probably used for differential fitness and cannot fit into our current analysis)
#     ls_set_Time0 = []
#     for set_i in list(df_metadata_filtered.SetName.unique()):
#         df_i = df_metadata_filtered[df_metadata_filtered.SetName == set_i]
#         if 'Time0' in list(df_i['Group'].unique()):
#             ls_set_Time0.append(set_i)
#     df_metadata_filtered = df_metadata_filtered[df_metadata_filtered.SetName.isin(ls_set_Time0)]
#     # Discard experiments where temperature shifts occur
#     df_metadata_filtered = df_metadata_filtered[df_metadata_filtered.Group != 'temperature shift']
#     # Drop samples where Shaking is nan and Group is not
#     df_metadata_filtered = df_metadata_filtered[~df_metadata_filtered.Shaking.isna()]
#     # [Became redundant] Drop samples where Growth Method is nan
#     # [Became redundant] Drop samples where Liquid v. solid is nan [only Liquid samples are considered]
#     # [Became redundant] Drop samples where Aerobic_v_Anaerobic is nan [only Aerobic samples are considered]
#     # Drop samples where MediaStrength is none
#     df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.Shaking.isna())]
#     # Drop samples where Media is NaN
#     df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.Media.isna())]
#     # Drop samples where MediaStrength is NaN
#     df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.MediaStrength.isna())]
#     # Drop samples where pH is NaN
#     df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.pH.isna())]
#     # Drop sets where Time0 is the only available sample
#     ls_set_MoreThanTime0 = []
#     for set_i in list(df_metadata_filtered.SetName.unique()):
#         df_i = df_metadata_filtered[df_metadata_filtered.SetName == set_i]
#         if list(df_i.Group.unique()) !=['Time0']:
#             ls_set_MoreThanTime0.append(set_i)
#     df_metadata_filtered = df_metadata_filtered[df_metadata_filtered.SetName.isin(ls_set_MoreThanTime0)]
#     df_metadata_final = deepcopy(df_metadata_filtered)
#     # Introduce a column for the set number
#     df_metadata_final['SetNumber'] = df_metadata_final.apply(lambda row: re.split('_',row['SetName'])[-1], axis=1)
#     # Assign the condition number as 0 where it is nan
#     df_metadata_final.loc[df_metadata_final['Condition Number'].isna(), 'Condition Number'] = 0
#     return df_metadata_final
#
#
# def data_preprocessing_base_level(df_data, df_metadata):
#     # Compute a dictionary with the list of index numbers for time0 samples for each experiment set ( key: experiment set)
#     dict_ls_time0 = {}
#     for setname in df_metadata['SetName'].unique():
#         dict_ls_time0[setname] = df_metadata.loc[(df_metadata.Group == 'Time0') & (df_metadata.SetName == setname), 'Index'].to_list()
#     # Discard all strains which do not map to a gene locus ID
#     ls_reqd_samples = (df_metadata.SetName + '.' + df_metadata.Index).to_list()
#     # ['barcode', 'rcbarcode', 'scaffold', 'strand', 'pos', 'locusId', 'f'] - the full set of non numerical entries in the dataframe
#     ls_reqd_samples_with_general_columns = ['locusId', 'f'] + ls_reqd_samples
#     df_data_out = deepcopy(df_data)
#     df_data_out = df_data_out.loc[~df_data_out.locusId.isna(), ls_reqd_samples_with_general_columns]
#     # Assign nan values to replace all 0 values because these are measurements with highest error and should be treated as such
#     df_data_out[df_data_out == 0] = np.nan
#     return df_data_out
#
# def data_average_across_samples(df_data, df_metadata):
#     print('_____________________________________________________________________________________')
#     print('Zero statistics before doing nanmean across measurement samples')
#     print('_____________________________________________________________________________________')
#     print('Total number of nan values : ', np.sum(np.sum(df_data.isna())))
#     print('% of nan values : ', np.sum(np.sum(df_data.isna())) / np.product(df_data.iloc[:, 2:].shape) * 100)
#     print('-------------------------------------------------------------------------------------')
#     print('Now we apply the nan mean across samples')
#     df_metadata_out = deepcopy(df_metadata)
#     df_data_out = deepcopy(df_data)
#     ls_drop_data_cols = []
#     ls_drop_metadata_indices = []
#     for setname in df_metadata_out['SetName'].unique():
#         for cond_no in df_metadata_out[df_metadata_out['SetName'] == setname].loc[:,'Condition Number'].unique():
#             print(setname, cond_no)
#             sys.stdout.write("\033[F")
#             ls_reqd_indices = df_metadata_out[(df_metadata_out['SetName'] == setname) & (df_metadata_out['Condition Number'] == cond_no)].loc[:, 'Index'].unique()
#             ls_reqd_samples = setname + '.' + ls_reqd_indices
#             df_data_out.loc[:, ls_reqd_samples[0:1]] = np.nanmean(df_data_out.loc[:, ls_reqd_samples], axis=1)
#             # Drop the excess columns in data
#             ls_drop_data_cols += list(ls_reqd_samples[1:])
#             # Drop the excess rows in metadata
#             ls_indices = df_metadata_out[(df_metadata_out['SetName'] == setname) & (df_metadata_out['Condition Number'] == cond_no)].index
#             ls_drop_metadata_indices += list(ls_indices[1:])
#     df_data_out.drop(labels=ls_drop_data_cols, axis=1, inplace=True)
#     df_metadata_out.drop(labels=ls_drop_metadata_indices, axis=0, inplace=True)
#     print('_____________________________________________________________________________________')
#     print('Zero statistics after doing nanmean across measurement samples')
#     print('_____________________________________________________________________________________')
#     print('Total number of nan values : ', np.sum(np.sum(df_data_out.isna())))
#     print('% of nan values : ', np.sum(np.sum(df_data_out.isna())) / np.product(df_data_out.iloc[:, 2:].shape) * 100)
#     print('-------------------------------------------------------------------------------------')
#     return df_data_out, df_metadata_out
#
#
# def data_display_stats_Time0_zeros(df_data, df_metadata):
#     df_data_in = deepcopy(df_data)
#     df_metadata_in = deepcopy(df_metadata)
#
#     # Analyzing the percentage of nan values in  Time0
#     df_md_Time0 = df_metadata_in[df_metadata_in['Condition Number'] == 0]
#     ls_Time0_samples = list(df_md_Time0.SetName + '.' + df_md_Time0.Index)
#     df_Time0 = df_data_in.loc[:, ls_Time0_samples]
#     df_Time0.index = df_data_in.locusId
#
#     print(' The percentage of 0 reads in each experimental set :')
#     print(np.sum(df_Time0.isna(), axis=0) / df_Time0.shape[0] * 100)
#     print('[WARNING] Since set 11 has 87% of the reads unknown at Time0, we discard this experiment entirely')
#     ls_set11_samples = list(df_metadata_in[df_metadata_in.SetNumber == 'set11'].SetName + '.' + df_metadata_in[
#         df_metadata_in.SetNumber == 'set11'].Index)
#     df_metadata_in = df_metadata_in[df_metadata_in.SetNumber != 'set11']
#     df_data_in.drop(labels=ls_set11_samples, axis=1, inplace=True)
#     print('_____________________________________________________________________________________')
#     print('Zero statistics after doing nanmean across measurement samples')
#     print('_____________________________________________________________________________________')
#     print('Total number of nan values : ', np.sum(np.sum(df_data_in.isna())))
#     print('% of nan values : ', np.sum(np.sum(df_data_in.isna())) / np.product(df_data_in.iloc[:, 2:].shape) * 100)
#     print('-------------------------------------------------------------------------------------')
#     print(np.sum(df_data.loc[:, ls_set11_samples].isna()))
#     return df_data_in, df_metadata_in
