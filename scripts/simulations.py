import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from packaging import version

# import libraries for biological data analysis
from coolpuppy import coolpup
from coolpuppy.lib import numutils
from coolpuppy.lib.puputils import divide_pups
from coolpuppy import plotpup
import cooler
import cooltools
from cooltools import insulation
import pyBigWig
import bioframe

# custom functions
import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
import functions
from functions import *

from coolpuppy_func import *

conv_pairs_TPM = pd.read_csv('data/conv_pairs_TPM.txt', sep='\t')

# %%
resolution = 500
clr = cooler.Cooler('data/whole-genome_L120_default.lef_slow.250.PK_0.05.polstep_1.lefstep_0.1.headon_0.codir_0.col-factor_codir_1.0.col-factor-headon_1.0.cutoff_2.0.mcool::/resolutions/'+str(resolution))

# Prepare view_df dataframe
df_chromsizes = prepare_view_df(clr)

# Compute expected cis interactions
expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000)

# %% 
selected_CGP = conv_pairs_TPM.query('anchor_group == "loop anchors"').loc[:,['chrom', 'end+', 'start-']]
selected_CGP.columns = ['chrom', 'start', 'end']
create_pileup_and_save_plot(clr, selected_CGP, df_chromsizes, expected, '', 
                                file_name="simulations.LA.intergenicCenter.plusGeneEnd.ByDistance" % group_length)
# %%
pValue_featureJaccardLoop(
    file_loops="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed",
    time=None,
    mode='whole',
    N_shuffle=1000,
    file_features='data/0AB_chromosight_sim.bed',
    name="realLoop_vs_simLoop.jaccard.whole", 
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000)