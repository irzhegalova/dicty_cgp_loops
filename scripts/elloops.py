# import standard python libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import fnmatch
import multiprocess

# import libraries for biological data analysis
import cooler
import bioframe

import cooltools
import cooltools.api.expected as ce
from cooltools.api import snipping
import cooltools.api.snipping as clsnip
from cooltools.lib.numutils import LazyToeplitz

import subprocess
import sys
sys.path.append('~/projects/dicty/hic_loop_study/scripts/functions/modules/')
from elscore import compute_fc_flames, compute_flames
import elscore

from functions import dataframe_difference, prepare_view_df

os.chdir('~/projects/dicty/hic_loop_study/')

# %%
print(cooler.__version__)
print(cooltools.__version__)
print(bioframe.__version__)

# %%
cooler_dir = "../dicty_hic_distiller/subsampled"
Timing = 0
clr = cooler.Cooler(os.path.join(cooler_dir, str(Timing) + "AB.filtered.mcool::/resolutions/2000"))
df_chromsizes = prepare_view_df(clr)
supports = df_chromsizes[['chrom', 'start', 'end']].values

resolution = clr.binsize
paired_sites = pd.read_csv("data/loops_quantifyChromosight/" + str(Timing) +"AB_chromosight_quantifyMarkedGood.bedpe", header=None, sep='\t')
paired_sites.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

# %%
flank = 20000
resolution = 2000
q = 0.2
# plot potential las loops

expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000)
oe_stack = cooltools.pileup(clr, paired_sites, view_df=df_chromsizes,
                                expected_df=expected, flank=flank)
fc_flames = compute_fc_flames(oe_stack, flank=flank, resolution=resolution, pad=3)

# create fc column
paired_sites['FC'] = fc_flames.tolist()
loops_rightFlames = paired_sites.loc[paired_sites['FC'] < np.nanquantile(fc_flames, q), :]
loops_leftFlames = paired_sites.loc[paired_sites['FC'] > np.nanquantile(fc_flames, 1 - q), :]
# %%
fc_thre = np.nanquantile(fc_flames, [q, 1 - q])
df_elloops = paired_sites.query('FC > @fc_thre[1] or FC < @fc_thre[0]')
df_elloops['strand'] = '+'
df_elloops.loc[df_elloops.FC > fc_thre[1], 'strand'] = '-'

df_elloops.drop(['end1', 'start2'], axis = 1)[['chrom1', 'start1', 'end2', 'chrom2', 'FC', 'strand']].to_csv('results/long_loops/0AB_elloops.bed', index=False, header=None, sep = '\t')

# %%
resolution=2000
# df_bedpe = bioframe.read_table('results/long_loops/0AB_loops_rightFlames0.2.bed', schema='bed3')
df_bedpe = bioframe.read_table('results/long_loops/0AB_loops_leftFlames0.8.bed', schema='bed3')
paired_sites = df_bedpe[df_bedpe['end'] - df_bedpe['start'] > 2 * resolution][df_bedpe['end'] - df_bedpe['start'] < np.quantile(df_bedpe['end'] - df_bedpe['start'], 0.99)]
median_size = np.median(paired_sites.end - paired_sites.start)
# %%
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
clr = cooler.Cooler(cooler_path + str(resolution))
# Prepare view_df dataframe
df_chromsizes = prepare_view_df(clr)

expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=2) #, median=True

# %%
flank=40000
pup = coolpup.pileup(clr, paired_sites,
rescale=True, rescale_size=int(1+flank * 2// resolution),  rescale_flank=1.5,
local=True,
features_format='bed', view_df=df_chromsizes, nproc=4, 
expected_df=expected, 
flank=flank, min_diag=2)
# %%
from coolpuppy import coolpup
from coolpuppy.lib import numutils
from coolpuppy.lib.puputils import divide_pups
from coolpuppy import plotpup
import cooler
plotpup.plot(pup,
             score=False, 
             cmap='coolwarm', #'viridis', #'coolwarm', #
             scale='log',
             vmax=1.8, vmin=0.6,
             height=5)
plt.savefig('results/pics/paper/dicty.leftFlame.2k.pdf', dpi=100, bbox_inches='tight')
plt.show()