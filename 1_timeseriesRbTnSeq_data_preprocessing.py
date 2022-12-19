import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import itertools
import copy
import sys

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22


def metadata_preprocessing_base_level(df_metadata_in):
    # Metadata Preprocessing
    # ------------------------------------------------------------------------------------------------------
    # Assumptions
    # Samples have to be liquid, aerobic and under a defined shaking speed
    # Samples have a Time0 measurement as only the log2 fitness ratio is used in the algorithm
    # [REVISIT] Samples with temperature shift (as Group) are discarded
    # [REVISIT] The volume effects across samples is ignored for now
    # Parameters currently used in the learning process   -   Temperature, pH, Shaking speed, various conditions
    # ------------------------------------------------------------------------------------------------------
    # Copying the metadata so that it can be edited
    df_metadata_filtered = copy.deepcopy(df_metadata_in)
    # Drop the set of experiments without Time0 samples ( because they are probably used for differential fitness and cannot fit into our current analysis)
    ls_set_Time0 = []
    for set_i in list(df_metadata_filtered.SetName.unique()):
        df_i = df_metadata_filtered[df_metadata_filtered.SetName == set_i]
        if 'Time0' in list(df_i['Group'].unique()):
            ls_set_Time0.append(set_i)
    df_metadata_filtered = df_metadata_filtered[df_metadata_filtered.SetName.isin(ls_set_Time0)]
    # Discard experiments where temperature shifts occur
    df_metadata_filtered = df_metadata_filtered[df_metadata_filtered.Group != 'temperature shift']
    # Drop samples where Shaking is nan and Group is not
    df_metadata_filtered = df_metadata_filtered[~df_metadata_filtered.Shaking.isna()]
    # [Became redundant] Drop samples where Growth Method is nan
    # [Became redundant] Drop samples where Liquid v. solid is nan [only Liquid samples are considered]
    # [Became redundant] Drop samples where Aerobic_v_Anaerobic is nan [only Aerobic samples are considered]
    # Drop samples where MediaStrength is none
    df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.Shaking.isna())]
    # Drop samples where Media is NaN
    df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.Media.isna())]
    # Drop samples where MediaStrength is NaN
    df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.MediaStrength.isna())]
    # Drop samples where pH is NaN
    df_metadata_filtered = df_metadata_filtered[~((df_metadata_filtered.Group !='Time0') & df_metadata_filtered.pH.isna())]
    # Drop sets where Time0 is the only available sample
    ls_set_MoreThanTime0 = []
    for set_i in list(df_metadata_filtered.SetName.unique()):
        df_i = df_metadata_filtered[df_metadata_filtered.SetName == set_i]
        if list(df_i.Group.unique()) !=['Time0']:
            ls_set_MoreThanTime0.append(set_i)
    df_metadata_filtered = df_metadata_filtered[df_metadata_filtered.SetName.isin(ls_set_MoreThanTime0)]
    df_metadata_final = copy.deepcopy(df_metadata_filtered)
    # Introduce a column for the set number
    df_metadata_final['SetNumber'] = df_metadata_final.apply(lambda row: re.split('_',row['SetName'])[-1], axis=1)
    # Assign the condition number as 0 where it is nan
    df_metadata_final.loc[df_metadata_final['Condition Number'].isna(), 'Condition Number'] = 0
    return df_metadata_final


def data_preprocessing_base_level(df_data, df_metadata):
    # Compute a dictionary with the list of index numbers for time0 samples for each experiment set ( key: experiment set)
    dict_ls_time0 = {}
    for setname in df_metadata['SetName'].unique():
        dict_ls_time0[setname] = df_metadata.loc[(df_metadata.Group == 'Time0') & (df_metadata.SetName == setname), 'Index'].to_list()
    # Discard all strains which do not map to a gene locus ID
    ls_reqd_samples = (df_metadata.SetName + '.' + df_metadata.Index).to_list()
    # ['barcode', 'rcbarcode', 'scaffold', 'strand', 'pos', 'locusId', 'f'] - the full set of non numerical entries in the dataframe
    ls_reqd_samples_with_general_columns = ['locusId', 'f'] + ls_reqd_samples
    df_data_out = copy.deepcopy(df_data)
    df_data_out = df_data_out.loc[~df_data_out.locusId.isna(), ls_reqd_samples_with_general_columns]
    # Assign nan values to replace all 0 values because these are measurements with highest error and should be treated as such
    df_data_out[df_data_out == 0] = np.nan
    return df_data_out

def data_average_across_samples(df_data, df_metadata):
    print('_____________________________________________________________________________________')
    print('Zero statistics before doing nanmean across measurement samples')
    print('_____________________________________________________________________________________')
    print('Total number of nan values : ', np.sum(np.sum(df_data.isna())))
    print('% of nan values : ', np.sum(np.sum(df_data.isna())) / np.product(df_data.iloc[:, 2:].shape) * 100)
    print('-------------------------------------------------------------------------------------')
    print('Now we apply the nan mean across samples')
    df_metadata_out = copy.deepcopy(df_metadata)
    df_data_out = copy.deepcopy(df_data)
    ls_drop_data_cols = []
    ls_drop_metadata_indices = []
    for setname in df_metadata_out['SetName'].unique():
        for cond_no in df_metadata_out[df_metadata_out['SetName'] == setname].loc[:,'Condition Number'].unique():
            print(setname, cond_no)
            sys.stdout.write("\033[F")
            ls_reqd_indices = df_metadata_out[(df_metadata_out['SetName'] == setname) & (df_metadata_out['Condition Number'] == cond_no)].loc[:, 'Index'].unique()
            ls_reqd_samples = setname + '.' + ls_reqd_indices
            df_data_out.loc[:, ls_reqd_samples[0:1]] = np.nanmean(df_data_out.loc[:, ls_reqd_samples], axis=1)
            # Drop the excess columns in data
            ls_drop_data_cols += list(ls_reqd_samples[1:])
            # Drop the excess rows in metadata
            ls_indices = df_metadata_out[(df_metadata_out['SetName'] == setname) & (df_metadata_out['Condition Number'] == cond_no)].index
            ls_drop_metadata_indices += list(ls_indices[1:])
    df_data_out.drop(labels=ls_drop_data_cols, axis=1, inplace=True)
    df_metadata_out.drop(labels=ls_drop_metadata_indices, axis=0, inplace=True)
    print('_____________________________________________________________________________________')
    print('Zero statistics after doing nanmean across measurement samples')
    print('_____________________________________________________________________________________')
    print('Total number of nan values : ', np.sum(np.sum(df_data_out.isna())))
    print('% of nan values : ', np.sum(np.sum(df_data_out.isna())) / np.product(df_data_out.iloc[:, 2:].shape) * 100)
    print('-------------------------------------------------------------------------------------')
    return df_data_out, df_metadata_out


def data_display_stats_Time0_zeros(df_data, df_metadata):
    df_data_in = copy.deepcopy(df_data)
    df_metadata_in = copy.deepcopy(df_metadata)

    # Analyzing the percentage of nan values in  Time0
    df_md_Time0 = df_metadata_in[df_metadata_in['Condition Number'] == 0]
    ls_Time0_samples = list(df_md_Time0.SetName + '.' + df_md_Time0.Index)
    df_Time0 = df_data_in.loc[:, ls_Time0_samples]
    df_Time0.index = df_data_in.locusId

    print(' The percentage of 0 reads in each experimental set :')
    print(np.sum(df_Time0.isna(), axis=0) / df_Time0.shape[0] * 100)
    print('[WARNING] Since set 11 has 87% of the reads unknown at Time0, we discard this experiment entirely')
    ls_set11_samples = list(df_metadata_in[df_metadata_in.SetNumber == 'set11'].SetName + '.' + df_metadata_in[
        df_metadata_in.SetNumber == 'set11'].Index)
    df_metadata_in = df_metadata_in[df_metadata_in.SetNumber != 'set11']
    df_data_in.drop(labels=ls_set11_samples, axis=1, inplace=True)
    print('_____________________________________________________________________________________')
    print('Zero statistics after doing nanmean across measurement samples')
    print('_____________________________________________________________________________________')
    print('Total number of nan values : ', np.sum(np.sum(df_data_in.isna())))
    print('% of nan values : ', np.sum(np.sum(df_data_in.isna())) / np.product(df_data_in.iloc[:, 2:].shape) * 100)
    print('-------------------------------------------------------------------------------------')
    print(np.sum(df_data.loc[:, ls_set11_samples].isna()))
    return df_data_in, df_metadata_in

##
# Importing the metadata
# df_metadata_raw = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/exps', sep='\t')
df_metadata_raw = pd.read_csv('RbTnSeq_data/exps', sep='\t')

# df_md = metadata_preprocessing_base_level(df_metadata_raw)
##
# Fitness data processing
# Import the fitness data
df_data_raw = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/all.poolcount', sep='\t')
df_data = data_preprocessing_base_level(df_data_raw, df_md)

df_data_1, df_md_1 = data_average_across_samples(df_data=df_data, df_metadata=df_md)

##
df_data_2, df_md_2 = data_display_stats_Time0_zeros(df_data=df_data_1, df_metadata=df_md_1)

##
df_d2 = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/other_data/fit_logratios_good.tab',sep='\t')
df_quality = pd.read_csv('Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/other_data/fit_quality.tab',sep='\t')

##
# todo collect only Time 0 points and see the zero stats of each strain, across the sets








## Number of genes with 10% of genes missing

# print('Number of strains with missing values : ', df_data_nanmean.isna().any(axis=0))

# plt.hist(np.sum(df_data_nanmean.isna(),axis=1))
# plt.show()

#Strains with
np.sum(np.sum(df_data_nanmean.iloc[:,2:].isna(),axis=1)/df_data_nanmean.iloc[:,2:].shape[1]>0.2)
# print('Number of strains with >10% missing values :' )









## Eliminate the mutant strains even if they have a single zero value
n_strains_initial = df_data_final.shape[0]
n_genes_initial = len(df_data_final.locusId.unique())
print(n_strains_initial)

print(n_genes_initial)
df_data_final.dropna(axis=0, inplace=True) # todo Very crucial
n_strains_final = df_data_final.shape[0]
n_genes_final = len(df_data_final.locusId.unique())

print('Number of strains in data')
print(' Initial : ', n_strains_initial, '   |   Final : ', n_strains_final)
print('% of strains lost : ', (n_strains_final-n_strains_initial)/n_strains_initial*100)
print('Number of genes in data')
print(' Initial : ', n_genes_initial, '   |   Final : ', n_genes_final)
print('% of genes lost : ', (n_genes_final-n_genes_initial)/n_genes_initial*100)




## Compute the fitness for each set
for setname in df_metadata_final.SetName.unique():
    print(setname)
    ls_T0 = [setname + '.' + item for item in dict_ls_time0[setname]]
    print(ls_T0)
    # print(np.nanmean(df_data_final.loc[:,ls_T0], axis=1))
    countT0 =np.nanmean(df_data_final.loc[:,ls_T0], axis=1)
    break














##
df_md_filt = df_metadata_raw.iloc[['set11' in i for i in list(df_metadata_raw.SetName)], :]
df_md_filt = df_md_filt.iloc[list(df_md_filt ['Condition Number'].isna()),:]
df_data_filt = df_data_raw.loc[:,list(df_md_filt.SetName + '.' + df_md_filt.Index)]





##
# ----------------------------------------------------------------------------------------------------------------------
# Analysis done by Aqib - Using his knowledge on single cell RNASeq, wanted to see how scanpy package can be used to
#  learn something about the variables at play across the experiments
# ----------------------------------------------------------------------------------------------------------------------
# import scanpy as sc
# # dataset is df_data_final
# adata = sc.AnnData(np.array(df_data_final).T,obs=df_metadata_final)
#
# ##
# # pca
# sc.pp.pca(adata,n_comps=50)
# ##
# # neighbors (kNN)
# sc.pp.neighbors(adata,n_neighbors=25)
# ##
# # clustering
# sc.tl.leiden(adata)
# ##
# # UMAP
# sc.tl.umap(adata)
#
# ##
# sc.pl.pca(adata,color='leiden')
# ----------------------------------------------------------------------------------------------------------------------

