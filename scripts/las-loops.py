import glob
import os
import sys
import numpy as np
import pandas as pd
from bioinfokit.analys import norm
from pybedtools import BedTool
import seaborn as sns
import matplotlib.pyplot as plt

from statannot import add_stat_annotation

sys.path.append('~/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import pValue_featureOccurenceInLoop

# lncRNA & las-loops
Timing = [0, 8]
for time in Timing:
    pValue_featureOccurenceInLoop(file_features='data/genome/dicty_rosengarten_lnRNA.bed', 
                                time=time,
                                mode='whole',
                                N_shuffle=1000,
                                file_loops='data/las_loops/%sAB_las_loops_4kb.sorted.bed',
                                name='las_loops/ncRNA_las_loops_%sh' % time)

# save las loop
Timing = [0, 8]
list_ofncRNA_andFeatures = ["data/ncRNA_rosengarten/ncRNA_allRosengarten.bed",
                            "data/genome/dicty_rosengarten_lnRNA.bed"]

ncRNA_andFeatures = BedTool(list_ofncRNA_andFeatures[1])
for time in Timing:
    name = 'data/las_loops/%sAB_las_loops_4kb.sorted.bed' % time
    #load las-loops
    las_loops = BedTool(name)
    # create list of lncRNA intersecting las-loops
    ncRNA_andFeatures.intersect(las_loops).saveas('data/ncRNA_rosengarten/lncRNA_interLasLoops_%sh.bed' % time)

# load expression files
files = glob.glob('data/ncRNA_rosengarten/GSE90829_RAW/*raw.txt')
files.sort()
ncRNA_expr = pd.read_table(files[0], header = None)
ncRNA_expr.columns = ['name', Path(files[0]).stem.split('_', 2)[2]]
times = []
for file in files[1:]:
    tmp = pd.read_table(file, header = None)
    tmp.columns = ['name', Path(file).stem.split('_', 2)[2]]
    ncRNA_expr = ncRNA_expr.merge(tmp, on='name')
    times.append(Path(file).stem.split('_', 4)[3])
    
for time in set(times):
    ncRNA_expr[time] = ncRNA_expr.filter(regex=time).mean(axis=1)
ncRNA_expr = ncRNA_expr[ncRNA_expr.columns.drop(list(ncRNA_expr.filter(regex='rep')))]
ncRNA_expr = ncRNA_expr.reindex(sorted(ncRNA_expr.columns), axis=1)
#compute number of points with expr greater than 0
ncRNA_expr['total'] = ncRNA_expr.select_dtypes(np.number).gt(0).sum(axis=1)

# create vector with correct names
a = ['A']*4 + ['B']*4
b = ['0','2','5','8'] * 2
expr_names = [m+str(n) for m,n in zip(b,a)]
expr_names.sort()
vec = ['Length']
vec.extend(expr_names)

rosengarten_lnRNA_expr = pd.read_table("results/dicty_genes+rosengarten_lnRNA.featureCounts.txt",comment='#')
rosengarten_lnRNA_expr[['Gene', 'Coord']] = rosengarten_lnRNA_expr['Geneid'].str.split('.', expand=True, n=1)
for feature in ["Geneid", "Coord", "Chr", "Start", "End", 'Strand']:
    del rosengarten_lnRNA_expr[feature] 
rosengarten_lnRNA_expr = rosengarten_lnRNA_expr.set_index('Gene')
rosengarten_lnRNA_expr.columns = vec

#tpm-normalized
nm = norm()
nm.tpm(df=rosengarten_lnRNA_expr, gl='Length')
# get TPM normalized dataframe
tpm_df = nm.tpm_norm
for time in set(['0','2','5','8']):
    tpm_df[time + 'h'] = tpm_df.filter(regex=time).mean(axis=1)

#lncRNA at las-loops
for time in Timing:
    tmp = pd.read_table("data/ncRNA_rosengarten/lncRNA_interLasLoops_%sh.bed" % time, header = None)
    lncRNA_lasLoops = tmp.iloc[:,8]
    my_list = np.unique(lncRNA_lasLoops)
    lncRNA_tpm = tpm_df.loc[tpm_df.index.isin(lncRNA_names)]

    lncRNA_tpm['status'] = np.nan #'NotLasLoops'
    lncRNA_tpm.loc[lncRNA_tpm.index.isin(my_list), 'status'] = 'LasLoops'

    lncRNA_tpm['delta_8h_0h'] = lncRNA_tpm['8h'] - lncRNA_tpm['0h']
    lncRNA_tpm['delta_0h_8h'] = lncRNA_tpm['0h'] - lncRNA_tpm['8h']

    number_lncRNA = len(lncRNA_tpm)
    NotLasLoops_values_delta_0h_8h = []
    NotLasLoops_values_delta_8h_0h = []
    for i in range(1000):
        np.random.seed(i)
        n_rand = np.random.choice(lncRNA_tpm.index, size=len(lncRNA_tpm[lncRNA_tpm['status'] == 'LasLoops']))
        #lncRNA_tpm.loc[lncRNA_tpm.index.isin(n_rand), 'status'] = 'NotLasLoops'
        NotLasLoops_values_delta_0h_8h.append(np.median(lncRNA_tpm.loc[lncRNA_tpm.index.isin(n_rand), "delta_0h_8h"]))
        NotLasLoops_values_delta_8h_0h.append(np.median(lncRNA_tpm.loc[lncRNA_tpm.index.isin(n_rand), "delta_8h_0h"]))


    df_final = pd.DataFrame({'status': 'NotLasLoops', 
                            'delta_0h_8h':NotLasLoops_values_delta_0h_8h,
                            'delta_8h_0h':NotLasLoops_values_delta_8h_0h})
    lncRNA_tpm_fin = lncRNA_tpm.loc[lncRNA_tpm['status'] == 'LasLoops',:].filter(['status', 'delta_0h_8h', 'delta_8h_0h'])

    df_final = df_final.append(lncRNA_tpm_fin, ignore_index=True)
    ax = sns.boxplot(x="status", y="delta_8h_0h", data=df_final) 
    ax.set(ylim=(-3, 3))
    add_stat_annotation(ax, data=df_final, x="status", y="delta_8h_0h", 
                        box_pairs=[("LasLoops", "NotLasLoops")],
                        test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

    # plt.savefig("results/pics/paper/las_loops/las_loops_lncRNAexpr_%sh.pdf" % time, bbox_inches='tight')
    plt.show()
    plt.clf()
