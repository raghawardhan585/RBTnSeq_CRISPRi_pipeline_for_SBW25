import pandas as pd
# pd.read_csv('RbTnSEq_data/other_data/fit.image')

import pyreadr
result = pyreadr.read_r('RbTnSEq_data/other_data/fit.image') # also works for Rds
##
 #pd.read_csv('RbTnSEq_data/other_data/expsUsed.csv')

df_expsUsed = df_expsUsed_all[df_expsUsed_all.Date_pool_expt_started == 'September 1, 2021']

## Get all the conditions

set_conditions = set(df_expsUsed.Condition_1.unique()).union(set(df_expsUsed.Condition_2.unique())).union(set(df_expsUsed.Condition_3.unique()))
ls_conditions = list(set_conditions - {''})

ls_pH = list(df_expsUsed.pH[~df_expsUsed.pH.isna()].unique())
ls_temp = list(df_expsUsed.Temperature[~df_expsUsed.Temperature.isna()].unique())

##

import pandas as pd
import pathlib
r_files = ['.RData', '.RHistory', '.RProfile']
for rf in r_files:
    p = pathlib.Path.cwd() / rf
    try:
        p.rename(p.with_suffix('.ignore'))
    except FileNotFoundError:
        pass
import rpy2
from rpy2 import robjects
for rf in r_files:
    p = pathlib.Path.cwd() / rf
    p = p.with_suffix('.ignore')
    try:
        p.rename(p.with_suffix(''))
    except FileNotFoundError:
        pass
robj = robjects.r['load']('/Users/shara/Desktop/Mission_Sakura/Phase_I_SBW25_RbTnSeq_FirstStrainRecommendations/RbTnSeq_data/other_data/fit.image')

df_expsUsed_all = pd.DataFrame(robjects.r['expsUsed']).T
print('ajfdhafsdlk')
df_genes = pd.DataFrame(robjects.r['genes']).T
print('ajfdhdfasfasdgdsg')
df_fit = pd.DataFrame(robjects.r['fit']).T
print('fassdajfdgagsrgrtewhafsdlk')

##
d = {key: robj.rx2(key)[0] for key in robj.names}
# df_genes = robj['genes']
# df_fit = robj['fit']
# df_expsUsed_all = robj['expsUsed']




