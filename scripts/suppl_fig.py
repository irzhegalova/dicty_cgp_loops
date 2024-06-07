# %%
# https://www.pnas.org/doi/epdf/10.1073/pnas.76.10.5269
from math import pow
import bioframe
import networkx as nx
import os
import pickle as pkl


import pandas as pd
from statannot import add_stat_annotation
from scipy.stats import zscore
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# DE
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import register_cmap
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize

import seaborn as sns
from seaborn.matrix import dendrogram, heatmap
import string
# rotate lineplot
from matplotlib import pyplot, transforms
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection

import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')

from custom_functions import *
from sum_expr_around_anchors \
    import compute_argminDerative, compute_LoopExprAndStrength, compute_Derivative, compute_exprSumAroundBedAnchor, create_distancesToNextLoop

# fonts
from textwrap import wrap
from matplotlib import font_manager

font_dirs = ["/usr/share/fonts/arial/"]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    

# colors
# Create an array with the colors you want to use
susielu = ["#b84873", "#6dbc5f","#5a3789","#bdab3d","#6d80d8","#bd7635","#bf68b8","#46c19a","#ba4b41","#71883b"] # Set your custom color palette
# susielu_pal = sns.set_palette(sns.color_palette(susielu))
susielu_greyscale = ["#636363", "#a4a4a4","#444444", "#a7a7a7", "#828282",] # Set your custom color palette
# susielu_greyscale_pal = sns.set_palette(sns.color_palette(susielu_greyscale))
susielu_accent = ["#636363", "#b84873", "#a4a4a4","#444444", "#a7a7a7", "#828282",] 
sns.set_theme(context="paper", style='whitegrid', palette=susielu, font="Arial")
cm = 1/2.54  # centimeters in inches

# sns.diverging_palette(350,5, 86, 40, center='light', as_cmap=True)
# sns.palplot()

PALETTES = {
    "coolwarm_colorbrew": np.array(
        (
            (178,24,43),
            (239,138,98),
            (253,219,199),
            (247,247,247),
            (209,229,240),
            (103,169,207),
            (33,102,172),
        )
    )
    / 255,
    "coolwarm_pallete": np.array(
        (   
            (109,128,216),
            (247,247,247),
            (186,75,65),
        )
    )
    / 255,
}

_register_cmaps()
# %%
fasta_file = bioframe.load_fasta("/home/fox/projects/dicty/hic_loop_study/data/genome/dicty.fa")
# a = (g/2)^G+C × ((1-g)/2)^A+T,
# where:
# a = probability
# g = G+C content of the target genome = 0.22
# C+G = number of G and C in the stretch = 2
# A+T = number of A and T in the stretch. = 2
# dicty 0.00184041
# human 0.05550736
g=0.22
GC_n=int(2)
AT_n=int(2)
a = pow(g/2, GC_n) * pow((1-g)/2, AT_n)
print(a)


# %% create S1
# a, Size distribution of DpnII restriction fragments in D. discoideum, S. cerevisiae, D. melanogaster and H. sapiens genomes. GC content of the genome is shown below the boxplots. 
# b, Clustering of Hi-C replicates. 
# c, Cis/trans ratio in Dcity compared to other organisms. #TODO delete
# d, Whole-genome Hi-C maps at the Dicty development stages. e, Eigenvector profiles across Dicty chromosomes. f, Network of las-loops identified in free-living cells and late aggregates.
from Bio.Seq import Seq
from Bio.Restriction import DpnII
from Bio import SeqIO

fig = plt.figure(layout="constrained", figsize=(18*cm, 16*cm))


# row0
subfigs = fig.subfigures(3, 1, wspace=0.07, height_ratios=[1.4, 1.2, 1.5])

# Create subplots for the boxplot and legend.
axs0 = subfigs[0].subplots(1, 2, width_ratios=[2.5, 1.5])

# Add a text annotation to each subplot.
for n in range(2):
    ax = axs0[n]
    ax.text(-0.1, 1.1, string.ascii_lowercase[n], transform=ax.transAxes, size=15, weight='bold')

# Define the paths to the genomes.
# The genomes are used to calculate the restriction fragment size distribution.
genomes_dic = {
    'yeast': 'data/genome_yeast/genome.fa',
    'dicty': "data/genome/dicty.fa",
    'dro': "data/hic_zoo/Drosophila_melanogaster.BDGP5.75.dna.toplevel.fa"
}

# Calculate the restriction fragment size distribution for each organism.
# The fragment size distribution is calculated by splitting the DNA sequence into restriction fragments
# using the DpnII enzyme. The length of each fragment is recorded and stored in a list.
# The fragment lengths are then grouped by organism and logged.
frag_len = []
for organism in genomes_dic.keys():
    fasta_path = genomes_dic[organism]
    for seq_record in SeqIO.parse(fasta_path, "fasta"):
        fragments = DpnII.catalyse(seq_record.seq)
        for seq in fragments:
            frag_len.append([organism, len(seq)])

# Convert the fragment lengths to a DataFrame and calculate the logged fragment lengths.
df = pd.DataFrame(frag_len, columns =['name', 'RF_size'])
df['log10_RF'] = np.log10(df.RF_size + 0.01)

# Save the fragment lengths to a file.
df.to_csv('results/RF_size.tsv')

# Read the human fragment lengths from a file.
df_human = pd.read_csv('results/human_RF_size_sampledNotMedian.tsv')

# Concatenate the fragment lengths from all organisms into a single DataFrame.
df = pd.read_csv('results/RF_size.tsv')
df_tmp = pd.concat([df, df_human], axis=0)
df_tmp.to_csv('results/all_RF_size.tsv')

# Read the concatenated fragment lengths from a file.
df = pd.read_csv('results/all_RF_size.tsv')

# Calculate the logged fragment lengths for the concatenated DataFrame.
df['log10_RF'] = np.log10(df.RF_size + 0.01)

# Create a boxplot of the logged fragment lengths for each organism.
sns.boxplot(data=df, x='name', y='log10_RF', ax=axs0[0], palette=susielu_accent)

# Set the title, ylabel, and xlabel of the boxplot.
axs0[0].set(title="\n".join(wrap("Restriction fragment size distribution", 25)),
            ylabel="log10(RF size)",
            xlabel="")

# hic cor
import fnmatch
cooler_names=[]
Timing_cooler_dir = "../dicty_hic_distiller/REPLICAS/filtered/"

# select only files in the directory
for fname in sorted(os.listdir(Timing_cooler_dir)):
    cooler_path = os.path.join(Timing_cooler_dir, fname) #+ '::/resolutions/2000'
    if os.path.isdir(cooler_path):
        # skip directories
        continue
    else:
        if fnmatch.fnmatch(fname, '*2kb.cool'):
            cooler_names.append(fname.split('.')[0])
cor_spread = pd.read_csv("results/hic_cor.tsv", sep='\t')

# place filenames in row and column names
cor_spread.index = cooler_names
# cor_spread.set_axis(cooler_names, axis='columns') #, inplace=True
cor_spread.columns = cooler_names
sns.heatmap(data=cor_spread, annot=True, fmt='.2f', cmap='viridis', ax=axs0[1], vmin=0.5, vmax=1, annot_kws={"size": 7, 'color':'black'}, linewidths=1, linecolor='black', cbar_kws={'label': 'HiCRep coefficient'})
axs0[1].set(title="\n".join(wrap("HiCRep correlation between samples", 25)))
axs0[1].tick_params(axis='y',rotation=45)

import sys
# sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/')
# import cis_trans_ratio as ctr
# Time = 0
# # Parse chromosome info
# zoo_path = {
#     'sac_cer': [6400, f'data/hic_zoo/GSM4585145_23C-45min.mcool::/resolutions/', 'data/hic_zoo/genome_renamed.fa.sizes'],
#     'dicty': [8000, f'../dicty_hic_distiller/subsampled/{Time}AB.filtered.mcool::/resolutions/', 'data/genome/dicty.chrom.sizes'],
#     'drosophila': [5000, f'data/hic_zoo/Bam_merged.1000.mcool::/resolutions/', 'data/hic_zoo/dm3.chrom.sizes'],
#     #'human': [res, '', 'data/hic_zoo/hg38.chrom.sizes']
#     }
# cis_trans_array =  np.empty(shape=[0, 3])
# dicty_chrom_path = 'data/genome/dicty.chrom.sizes'
# stages = [0, 2, 5, 8]
# chrom_sizes_ser = pd.read_table(dicty_chrom_path, sep='\t', names=['size'], index_col=0)['size']
# chrom_names = chrom_sizes_ser.index
# for stage in stages:
#     clr = cooler.Cooler(f'../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::resolutions/100000' % stage)
#     # Compute cis-trans ratio
#     df = ctr.cis_trans_per_chrom_array(chrom_names, '../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::resolutions/8000' % stage, str(stage)+'h')
#     cis_trans_array = np.append(cis_trans_array, df, axis=0)
# cis_trans_df = pd.DataFrame(cis_trans_array)
# cis_trans_df.columns = ['chr', 'cis-trans ratio', 'stage']
# cis_trans_df.to_csv('results/dicty_cistrans.tsv')
    
    
# for sp in zoo_path.keys():
#     chrom_sizes_ser = pd.read_table(zoo_path[sp][2], sep='\t', names=['size'], index_col=0)['size']
#     chrom_names = chrom_sizes_ser.index

#     # Compute cis-trans ratio
#     df = ctr.cis_trans_per_chrom_array(chrom_names, zoo_path[sp][1] + str(zoo_path[sp][0]), sp)
#     cis_trans_array = np.append(cis_trans_array, df, axis=0)
# cis_trans_df = pd.DataFrame(cis_trans_array)
# cis_trans_df.columns = ['chr', 'cis-trans ratio', 'organism']
# cis_trans_df.to_csv('results/cistrans.tsv')
# cis-trans ratio for merged data in dicty developmental stages
# cis_trans_df = pd.read_csv('results/dicty_cistrans.tsv')
# # cis_trans_human_df = pd.read_csv('results/human_cistrans.tsv')

# # cis_trans_df = pd.concat([cis_trans_df, cis_trans_human_df])
# # cis_trans_df.to_csv('results/cistrans_all.tsv')

# cis_trans_df = cis_trans_df.astype({'cis-trans ratio': 'float64'})

# sns.boxplot(cis_trans_df, x='stage', y='cis-trans ratio',ax=axs0[2])
# axs0[2].set(title="\n".join(wrap("Cis-trans ratio betweeen developmental stages", 25)), xlabel="", ylabel="Cis-trans ratio")

axs1 = subfigs[1].subplots(1, 5, gridspec_kw={'width_ratios':[1,1,1,1,0.1]})
subfigs[1].text(-0.1, 1.1, string.ascii_lowercase[2], transform=axs1[0].transAxes,size=15, weight='bold')
subfigs[1].suptitle('Hi-C maps for developmental stages', fontsize=10)

# Fig S1d
import cooler
import cooltools.lib.plotting
from matplotlib.colors import LogNorm
norm = LogNorm(vmax=0.3)
norm_raw = LogNorm(vmin=1, vmax=10_000)

plt_width=4
# f, axs = plt.subplots(
#     figsize=( plt_width+plt_width+2, plt_width+plt_width+1),
#     ncols=2,
#     gridspec_kw={'height_ratios':[4,],"wspace":0.01,'width_ratios':[1,.05]},
#     constrained_layout=True)
Time = [0,2,5,8]
for i in range(4):
    ### to load a cooler with a specific resolution use the following syntax:
    clr = cooler.Cooler(f'../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::resolutions/100000' % Time[i])
    ax = axs1[i]
    im = ax.matshow(
        clr.matrix(balance=True)[:], 
        norm=norm,  
        cmap='fall',
        aspect='auto'
    ); 
    # ax.xaxis.set_visible(False)
    ### to make a list of chromosome start/ends in bins:
    chromstarts = []
    for j in clr.chromnames:
        print(f'{j} : {clr.extent(j)}')
        chromstarts.append(clr.extent(j)[0])
    ax.set_xticks(chromstarts)
    ax.set_xticklabels(clr.chromnames)
    ax.tick_params(axis='x',rotation=45)
    if i > 0:
        ax.yaxis.set_ticks_position('none') 
        ax.set_yticks([])
    else:
        ax.set_yticks(chromstarts)
        ax.set_yticklabels(clr.chromnames)
    ax.xaxis.tick_bottom()

cax = axs1[4]
plt.colorbar(im, cax=cax, label='corrected frequencies')
axs2 = subfigs[2].subplots(1, 3, gridspec_kw={'width_ratios':[0.5,1,1]})
for n in range(2):
    ax = axs2[n]
    ax.text(-0.1, 1.1, string.ascii_lowercase[n+3], transform=ax.transAxes, size=15, weight='bold')

chromsizes = bioframe.read_chromsizes('data/genome/dicty.chrom.sizes')
#chromosomes = chromsizes['chr'].values
bins = cooler.binnify(chromsizes, 100_000)
fasta_records = bioframe.load_fasta('data/genome/dicty.fa')
bins = bioframe.frac_gc(bins, fasta_records)
# obtain first 3 eigenvectors
cis_eigs = cooltools.eigs_cis(
                        clr,
                        bins,
                        n_eigs=3,
                        #phasing_track='GC',
                        #sort_metric='var_explained'
                        )

# cis_eigs[0] #returns eigenvalues, here we focus on eigenvectors
eigenvector_track = cis_eigs[1][['chrom','start','end','E1']]

from mpl_toolkits.axes_grid1 import make_axes_locatable
ax = axs2[0]
# ax.matshow(
#     clr.matrix()[:],
#     norm=norm,
#     cmap='fall'
# );
# plt.axis([0,bins.shape[0],bins.shape[0],0])

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax, label='corrected frequencies');

ax.set_xticks(chromstarts)
ax.set_xticklabels(clr.chromnames)
# ax.set_yticks(chromstarts)
# ax.set_yticklabels(clr.chromnames)
# ax.xaxis.tick_bottom()
# ax.set_xticks([0, 20, 40])
# ax.set_xticklabels([0, 200, 400])

# ax1 = divider.append_axes("top", size="90%", pad=0.25, sharex=ax)
weights = clr.bins()[:]['weight'].values
ax.plot([0,bins.shape[0]],[0,0],'k',lw=0.25) #.query("chrom == 'chr1'")
ax.plot(eigenvector_track['E1'].values, label='E1') #query("chrom == 'chr1'")

ax.set(ylabel='E1', xlabel='chromosome coordinates, Mb')
# ax.set_xticks([]);
axs2[0].set(title="\n".join(wrap("Eigenvalues whole genome", 15)))


def plot_stages_networks(axs2):
    """
    Plot the networks of loops common anchors for the two development stages.
    
    Parameters
    ----------
    axs2 : matplotlib.axes._subplots.AxesTuple
        The tuple of axes to plot the networks.
    """
    
    # Read the data
    dots_4kb = pd.read_excel('/home/fox/projects/dicty/hic_loop_study/data/Long_loops_02-06.mod.xlsx',
                        names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'stage'])
    
    # Lowercase chromosome names
    dots_4kb.loc[:, 'chrom1'] = dots_4kb.loc[:, 'chrom1'].str.lower()
    dots_4kb.loc[:, 'chrom2'] = dots_4kb.loc[:, 'chrom2'].str.lower()

    # Create the dictionaries of dataframes for each stage
    dots_4kb_h = {
        stage: dots_4kb.loc[dots_4kb['stage'] == f'{stage}h',
                            ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']]
        for stage in [0, 8]
    }

    # Add columns with region name in ucsc format
    for stage in [0, 8]:
        dots_4kb_h[stage]['ucsc1'] = dots_4kb_h[stage]['chrom1'] + ':' + dots_4kb_h[stage]['start1'].astype(str) + '-' + \
        dots_4kb_h[stage]['end1'].astype(str)
        dots_4kb_h[stage]['ucsc2'] = dots_4kb_h[stage]['chrom2'] + ':' + dots_4kb_h[stage]['start2'].astype(str) + '-' + \
        dots_4kb_h[stage]['end2'].astype(str)

    # Create the graphs
    G = {stage: nx.from_pandas_edgelist(dots_4kb_h[stage], source='ucsc1', target='ucsc2')
        for stage in [0, 8]}

    # Set the node positions
    pos = nx.spring_layout(G[0])

    # Plot the graphs
    nx.draw(G[0], pos, ax=axs2[1], font_size=6, node_size=100, with_labels=True, node_color="#5a3789")
    axs2[1].set_xticks([]);
    axs2[1].set_yticks([]);
    axs2[1].set(title="\n".join(wrap("0h stage", 25)))

    pos = nx.spring_layout(G[8])
    nx.draw(G[8], pos, ax=axs2[2], font_size=6, node_size=100, with_labels=True, node_color="#5a3789")
    axs2[2].set_xticks([]);
    axs2[2].set_yticks([]);
    axs2[2].set(title="\n".join(wrap("8h stage", 25)))



plot_stages_networks(axs2)



plt.savefig("results/pics/paper/FigS1_v4.pdf", format="pdf", dpi=300)
plt.show()
# for i in np.where(np.diff( (cis_eigs[1]['E1']>0).astype(int)))[0]:
#     ax.plot([0, bins.shape[0]],[i,i],'k',lw=0.5)
#     ax.plot([i,i],[0, bins.shape[0]],'k',lw=0.5)

# %% Fig S2
# Supplementary Figure 2.
# a, loop size distribution for the four development stages and cell cycle phases. #TODO replace with Proportion and size distribution of consecutive and stand-alone loops
# b, IS profiles at various slide insulation squares.
fig = plt.figure(layout="constrained", figsize=(18*cm, 16*cm))

# Define the subfigures with their respective height ratios
subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[1.2, 3])
# row0
axs0 = subfigs[0].subplots(1, 4, width_ratios=[1.2, 0.5, 0.5, 0.5])

# Add subplot labels
for n in range(2):
    axs0[n].text(-0.1, 1.1, string.ascii_lowercase[n], transform=axs0[n].transAxes, size=15, weight='bold')

# Parse files
stages = ['2AB', '5AB', '8AB'] #'0AB', 

# Define a list to keep zoomed trans contact matrices
maps_dicty = [] 

# Loop over the stages and loop types
for stage in stages:
    for type in ['consecutive', 'standalone']: #'all', 
        # Define the loop file paths based on the stage and type
        if type == "all":
            all_loops = {"%s_all" % stage: f'data/loops_quantifyChromosight/{stage}_chromosight_quantifyMarkedGood.bed' for stage in stages}
        elif type == "consecutive":
            all_loops = {"%s_%s" % (stage, type): f'data/loops_quantifyChromosight/{stage}_consecutive_1binsAdded.bed' for stage in stages}
        elif type == "standalone":
            all_loops = {"%s_%s" % (stage, type): f'data/loops_quantifyChromosight/{stage}_standAlone_1binsAdded.bed' for stage in stages}
        
        # Read the loop file and add the loop type and stage information
        mtx_comb = bioframe.read_table(all_loops["%s_%s" % (stage, type)], schema='bed3')
        mtx_comb['type'] = type
        mtx_comb['stage'] = stage
        mtx_comb['length'] = mtx_comb['end'] - mtx_comb['start']
        
        # Add the loop length information to the maps_dicty list
        for loop_len in mtx_comb['length'].tolist():
            maps_dicty.append([stage, type, loop_len])

# Convert the maps_dicty list to a DataFrame
df_loopSize = pd.DataFrame.from_dict(maps_dicty)
df_loopSize.columns = ['stage', 'type', 'length']

# Set the color palette and plot the boxplot
sns.color_palette(susielu)
sns.boxplot(df_loopSize, x='stage', y='length', hue='type', ax=axs0[0])
axs0[0].yaxis.set_label_position("right")
axs0[0].yaxis.tick_right()
axs0[0].legend([],[], frameon=False)
axs0[0].set(xlabel='', ylabel='loop length, kbp', xticklabels=['0h', '2h', '5h', '8h'], yticks=[20000, 40000], yticklabels=['20', '40']) #
subfigs[0].suptitle('Loop length throught development')

# Add pie charts for each stage
for i in range(1,4):
    t = stages[i-1]
    axs0[i].pie(df_loopSize.query('stage == @t')['type'].value_counts().tolist(), autopct='%1.1f%%')
    # axs0[i].legend([],[], frameon=False)

# row 2
axs1 = subfigs[1].subplots(2, 4, width_ratios=[1.5, 1.5, 1.5, 1.5], height_ratios=[1.2, 3.6] )#, sharey='row'
axs1[0,0].text(-0.1, 1.1, string.ascii_lowercase[2], transform=axs1[0,0].transAxes, size=15, weight='bold')
subfigs[1].suptitle('Insulation score with different window size')

import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import *

# Define the settings for the plot
fill = True
mode = 'median'
window = 20000
nbins = 30
how_far_from_edge = 10
ymin = -1.3
ymax = 1.3
vmin = -2.8
vmax = 2.8
resolutions = ['5k', '10k', '20k', '50k']

# Create the figure and subplots
fig, axs1 = plt.subplots(2, 4, figsize=(12, 4))
cbar_ax = fig.add_axes([.99, .1, .05, .35])

# Loop over the resolutions
for i in range(4):
    res = resolutions[i]

    # Plot the data
    df_IS, order = plot_around_loop("bw/dicty_CS.1k.w%s.bw" % res,
                    "test.pdf", bed_list=["data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe"],
                    fill=True, window=20000, nbins=30, how_far_from_edge=10, 
                    mode='median', ymin=-1, ymax=1.0, vmin=-3, vmax=3, return_matrix=True)

    order = order.tolist()
    ax_col_marg = axs1[0, i]  # upper profile
    ax_heatmap = axs1[1, i]  # heatmap

    # Create the heatmap
    sns.heatmap(
        df_IS[0][order, :],
        ax=ax_heatmap,
        cbar=i == 0,
        cbar_ax=None if i else cbar_ax,
        xticklabels=True,
        yticklabels=False,
        cmap="coolwarm_pallete",
        vmin=vmin,
        vmax=vmax)

    # Format the heatmap
    ax_heatmap.xaxis.tick_bottom()
    ax_heatmap.xaxis.set_label_position('bottom')
    ax_heatmap.set(
        xticks=np.arange(0, nbins+1, 30), 
        xticklabels=(np.arange(0, nbins+1, 30)-nbins//2)*window*2/nbins/1000, 
        xlabel="\n".join(wrap('relative position from loop anchors, kbp', 25)))

    # Calculate and plot the median
    if mode == "median":
        upper_median = np.nanmedian(df_IS[0][order, :], axis=0)
        ax_col_marg.plot(upper_median, color=susielu[4])
        ax_col_marg.set(title="window=%sb" % res)

        # Format the upper profile
        if i == 3:
            ax_col_marg.yaxis.set_label_position("right")
            ax_col_marg.yaxis.tick_right()
            ax_col_marg.set(xticklabels=[])
        else:
            ax_col_marg.set(xticklabels=[], yticklabels=[])
        ax_col_marg.set_ylim([ymin, ymax])

        # Calculate and plot the error bars
        to_bootstrap = (df_IS[0][order, :],)
        rng = np.random.default_rng()
        res = bootstrap(
            to_bootstrap,
            np.nanstd,
            n_resamples=1000,
            confidence_level=0.9,  # vectorized=True,
            axis=0,
            random_state=rng,
            method="basic",
        )
        dyfit = res.standard_error
        if fill:
            ax_col_marg.fill_between(
                range(nbins),
                upper_median - dyfit,
                upper_median + dyfit,
                color=susielu[4],
                alpha=0.3,
            )
        cbar_ax.set_ylabel('Convergence Score')

    # Add vertical lines
    ax_col_marg.vlines(
        x=[how_far_from_edge - 1, nbins - how_far_from_edge],
        ls="--",
        ymin=ymin,
        ymax=ymax,
        colors='black',
        alpha=0.5
    )
    ax_col_marg.vlines(
        x=[how_far_from_edge - 1 + 2, nbins - how_far_from_edge - 2],
        ls="--",
        ymin=ymin,
        ymax=ymax,
        colors='black', 
        alpha=0.5
    )

# Save the figure
fig.savefig("results/pics/paper/FigS2_ISzoo_v2.pdf", format="pdf", dpi=300, bbox_inches='tight')
# plt.show()

# %% Plot loop profiles around IS loops from GM12878

resolution = 5000  # resolution of the cool file
organism = 'human'
windows = [25 * resolution]  # list of window sizes in Kb
bed_list = ['data/hic_zoo/GSE63525_GM12878_primary_HiCCUPS_looplist.bed']  # list of bed files
mode = 'median'  # mode used to calculate the loop profiles
df_chromsizes = bioframe.read_chromsizes("data/hic_zoo/hg38.chrom.nochr.sizes", filter_chroms=False)  # chromosome sizes

for window in windows:
    # generate long name for the plot
    long_name = organism + '.IS.' + str(window // 1000) + 'k'
    # plot loop profiles
    plot_around_loop(
        'bw/' + long_name + '.bw',  # bigWig file
        long_name + '.' + mode + ".pdf",  # output file
        chrom_file=df_chromsizes,  # chromosome sizes
        resolution=resolution,  # resolution of the bigWig file
        nbins=75,  # number of bins to plot
        mode=mode,  # mode used to calculate the loop profiles
        window=125000,  # window size in bp
        fill=False,  # whether to fill the area under the curve
        how_far_from_edge=25,  # how far from the edge to plot the loop profiles
        ymin=-0.5, ymax=0.5, vmin=-2, vmax=2,  # limits for the y-axis
        bed_list=bed_list  # list of bed files
    )

# %% Fig S3
# a, Clustering of RNA-seq replicates. 
# b, A heatmap of gene expression at Dicty development stages. 
# c, UMAP of loop k-means clustering according to the coverage of genes with the same developmental trajectories.

# Create a figure with a constrained layout
fig = plt.figure(layout="constrained", figsize=(18*cm, 16*cm))

# Create two subfigures in the figure
# The first subfigure has 1 row and 4 columns with a 0.07 width space between columns
# The second subfigure has 1 row and 1 column
subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[1, 1])

# Create an array of 1 row and 4 columns to store the subplot axes
axs0 = subfigs[0].subplots(1, 4, width_ratios=[2.4, 0.8, 1.4, 0.2])

# Add a suptitle to the first subfigure
# axs0[0].text(-0.1, 1.1, string.ascii_lowercase[0], transform=axs0[0].transAxes, size=15, weight='bold')

# Load the gene expression counts data
counts_df = pd.read_table('data/all_gene_markdup_htseq', index_col=0, header=None).T
counts_df.index = ['0A', '0B','2A', '2B','5A', '5B','8A', '8B']

# Load the metadata for the samples
metadata = pd.DataFrame({'sample': ['0A', '0B','2A', '2B','5A', '5B','8A', '8B'], 'condition': ['0h', '0h', '2h', '2h', '5h', '5h', '8h', '8h']}).set_index('sample')

# create a DeseqDataSet object
inference = DefaultInference(n_cpus=4)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design_factors="condition",
    refit_cooks=True,
    inference=inference)
dds.deseq2()

# Perform PCA on the normalized counts data

X = dds.layers['normed_counts']
x = StandardScaler().fit_transform(X) # Normalize the features
pca_dds = PCA(n_components=2)
principalComponents_dds = pca_dds.fit_transform(x)

# Create a DataFrame to store the principal components and the sample conditions
principal_dds_Df = pd.DataFrame(data = principalComponents_dds
             , columns = ['principal component 1', 'principal component 2'])
principal_dds_Df['condition'] = metadata['condition'].values

# Add a title, labels, and a scatter plot to the first subplot
axs0[0].text(-0.1, 1.1, string.ascii_lowercase[0], transform=axs0[0].transAxes, size=15, weight='bold')
axs0[0].set(title="\n".join(wrap("Principal Component Analysis of RNA-seq data", 25)) , ylabel="PC-2: %s%% of variance" % str(int(np.round(pca_dds.explained_variance_ratio_*100,0)[1])), xlabel='PC-1: %s%% of variance' % str(int(np.round(pca_dds.explained_variance_ratio_*100,0)[0])))
sns.scatterplot(data=principal_dds_Df, x='principal component 1', y='principal component 2', s = 50, hue='condition', edgecolors='black', palette=susielu, ax=axs0[0])

# B
# Create a DataFrame with RNA-seq TPM values and apply z-score normalization
tpm_df = create_rnaseq_df_zscore(need_coord=False)

# Perform hierarchical clustering on the z-score normalized genes
stages = ['0AB', '2AB', '5AB', '8AB']
tmp = tpm_df.loc[:,stages].apply(zscore, 1)#.iloc[0:1000,:]

# Perform hierarchical clustering on the z-score normalized genes
dendrogram_row = dendrogram(tmp, metric="correlation", method='average', label=False, axis=0, ax=axs0[1], rotate=True, linkage=None,tree_kws={"linewidths": .5})
xind = dendrogram_row.reordered_ind
matrix = tmp.iloc[xind, ]

# Create a heatmap of the z-score normalized genes
# with hierarchical clustering applied
heatmap(matrix, cmap='RdYlGn_r', cbar=True, ax=axs0[2], xticklabels=stages, yticklabels=False, cbar_ax=axs0[3])
axs0[1].set(title="\n".join(wrap("RNA-seq heatmap", 25)))  # ylabel="genes", xlabel=""

with open(os.path.join('results/', "dds.pkl"), "wb") as f:
    pkl.dump(dds, f)
with open(os.path.join('results/', "dds.pkl"), "rb") as f:
    dds2 = pkl.load(f)
stat_res = DeseqStats(dds, inference=inference)
stat_res.summary()
stat_res.results_df.to_csv(os.path.join('results/DE/', "results.csv"))

stat_res_B_vs_A = DeseqStats(dds, contrast=["condition", "0h", "2h"], inference=inference)
tmp = stat_res.results_df.query('padj < 0.05').index.tolist()

upset_df = pd.DataFrame({'genes': counts_df.columns})

for t1, t2 in [[0,2], [2,5], [5,8]]:
    if t1 < t2:
        stat_res = DeseqStats(dds, contrast=["condition", "%sh" % t1, "%sh" % t2], inference=inference)
        stat_res.summary()
        tmp = stat_res.results_df.query('padj < 0.05 and (log2FoldChange < 0.5 or log2FoldChange > 1)').index.tolist()
        upset_df["%sh_%sh" % (t1, t2)] = 0
        upset_df.loc[upset_df['genes'].isin(tmp), "%sh_%sh" % (t1, t2)] = 1

from upsetplot import generate_data, plot
from upsetplot import UpSet
upset_plot_df = upset_df.drop(['genes'], axis=1).groupby(['0h_2h', '2h_5h', '5h_8h']).size()
upset_plot = UpSet(upset_plot_df, facecolor='#b84873', shading_color='white', with_lines=True)
upset_plot.style_subsets(edgecolor='black',linewidth=1, min_subset_size=0, min_degree=0)

axs1 = subfigs[1].subplots(2, 2, width_ratios=[0.01, 2.5], height_ratios=[2, 1.2],sharex='col')
axs1[0,0].text(-0.1, 1.1, string.ascii_lowercase[2], transform=axs1[0,0].transAxes, size=15, weight='bold')
upset_plot.plot_shading(axs1[1,1])
upset_plot.plot_matrix(axs1[1,1])
# axs1[1,1].set_xticks([])
axs1[1,1].set_yticks([0,1,2])
# axs1[1,1].set_xticklabels([])
axs1[1,1].set_yticklabels(["2h vs 5h\n(n=5041)", "0h vs 2h\n(n=3930)", "5h vs 8h\n(n=3387)"])
axs1[0,1].set(title='DE genes between stages')
# upset_plot.plot_totals(axs1[1,0])
axs1[1, 0].axis('off')
upset_plot.plot_intersections(axs1[0,1])
axs1[0, 0].axis('off')
axs1[1,1].tick_params(
            axis='both',
            which='both',
            left=True,
            right=False,
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=True)
axs1[1,1].set_frame_on(True)



axs2 = subfigs[1].subplots(1, 2, width_ratios=[0.4, 1.8])
axs2[0].text(-0.1, 1.1, string.ascii_lowercase[2], transform=axs2[0].transAxes, size=15, weight='bold')
import umap
import umap.plot
from sklearn.cluster import KMeans
loops_path = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed"

loops_bed = BedTool(loops_path)
loops_df = pd.read_table(loops_path, header=None)
loops_df.columns = ["chrom", "start", "end"]
for group in ['1', '2', '3', '4', 'Unicellular', 'Multicellular']:
    feature = BedTool("data/genome/pseudogenes_group" + group + ".bed")
    covByPseudogenes = loops_bed.coverage(feature).to_dataframe()
    loops_df["covByGroup"+str(group)]  = covByPseudogenes.loc[:, 'thickStart'].tolist()
loops_dropped = loops_df[['covByGroup1', 'covByGroup2', 'covByGroup3', 'covByGroup4']]

# UMAP
fit = umap.UMAP(n_neighbors=50, min_dist=0.25,
                n_components=2,
                metric="euclidean", 
                random_state=42) #correlation
u = fit.fit_transform(loops_dropped)

kmeans = KMeans(n_clusters=6, random_state=0).fit(loops_dropped)
kmeans.labels_
loops_df['kmeans_cl'] = kmeans.labels_

scatter = axs2[1].scatter(u[:,0], u[:,1],
            c=kmeans.labels_, cmap='viridis', edgecolors='black')
clust = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6']
# plt.legend(labels=['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6'])
axs2[0].legend(handles=scatter.legend_elements()[0], labels=clust)

axs2[0].axis('off')
plt.savefig("results/pics/paper/FigS3_v4.pdf", format="pdf", dpi=300)


# %% Fig S3 -> S4
# a-c, Pileups for IS, Gene coverage, CS for el-loops (3 groups)
# d, Scatter IS vs RNA-seq TPM within a 2-kb bin (delta?).
# g average gene body to S3

# a-d, Pileups for panels IS, Gene coverage, CS for el-loops (3 groups). 
# e, Scatter IS vs RNA-seq TPM within a 2-kb bin (delta?). 
# f, Related to Fig. 4h and Fig. 4i: the same picture for a different set of parameters to demonstrate the robustness of the analysis.
fig = plt.figure(layout="constrained", figsize=(18*cm, 34*cm))

subfigs = fig.subfigures(4, 1, wspace=0.07, height_ratios=[4, 2, 2, 2])
# row0
axs0 = subfigs[0].subplots(4, 4, width_ratios=[1.5, 1.5, 1.5, 1.5], height_ratios=[1.2, 1.2, 1.2, 0.2] )#, sharey='row'

import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from coolpuppy_func import add_heatmap_custom
from custom_functions import *
fill=True
mode='median'
window=20000
nbins=30
how_far_from_edge=10
ymin=-1.3
ymax=1.3
vvalue_list = [[-0.5, 0.5], [0, 1], [0, 1],[-3,3]]
bw_list = ['bw/IS/0AB.20k.IS.cooltools.bw', 'bw/cov_by_minusGenes.bw', 'bw/cov_by_plusGenes.bw', 'bw/0AB_CS.1kb.bw']
names_list = ['Insulation Score', 'Minus strand gene coverage', 'Plus strand gene coverage', 'Convergence Score']

for i in range(len(bw_list)):
    vmin=vvalue_list[i][0]
    vmax=vvalue_list[i][1]
    axs0[0,i].text(-0.1, 1.3, string.ascii_lowercase[i], transform=axs0[0,i].transAxes, size=15, weight='bold')
    bw_path = bw_list[i]
    df_IS, order = plot_around_loop(bw_path,
                    "test.pdf", bed_list=["results/long_loops/0AB_loops_rightFlames0.2.bedpe", "results/long_loops/0AB_regular_loops.bedpe", "results/long_loops/0AB_loops_leftFlames0.8.bedpe"],
                    fill=True, window=20000, nbins=30, how_far_from_edge=10, 
                    mode='median', ymin=-1, ymax=1.0, vmin=-3, vmax=3, return_matrix=True)

    order = order.tolist()
    # ax_col_marg = axs1[0,i]  # upper profile
    ax_bar = axs0[3,i]  # colorbar
    #cbar_ax = axs0.add_axes([.99, .35, .03, .5])
    for j in range(3):
        ax_heatmap = axs0[j,i]  # heatmap
        if j == 0 : 
            ax_heatmap.set_title("\n".join(wrap(names_list[i], 15)))
        if i == 0 and j == 0:
            ax_heatmap.set(ylabel="3' el-loops")
            for_order = np.nanmedian(
                df_IS[j][:, 19:22],
                axis=1,
            )
            order = np.argsort(for_order)[::1]
        if i == 0 and j == 1:
            ax_heatmap.set(ylabel="Ordinary loops")
            for_order = np.nanmedian(
                df_IS[j][:, [9,10,11,19,20,21]],
                axis=1,
            )
            order = np.argsort(for_order)[::1]
        if i == 0 and j == 2:
            ax_heatmap.set(ylabel="5' el-loops")
            for_order = np.nanmedian(
                df_IS[j][
                    :, 9:12],
                axis=1,
            )
            order = np.argsort(for_order)[::1]
        sns.heatmap(
            df_IS[j][order, :],
            ax=ax_heatmap,
            # cbar=False, # cbar_ax=ax_bar,
            cbar=j == 0,
            #linewidths=0.01,
            #linecolor='black',
            cbar_ax=ax_bar,
            cbar_kws = {'orientation':'horizontal'},
            yticklabels=False,
            xticklabels=False,
            cmap='coolwarm_pallete',
            vmin=vmin,
            vmax=vmax)
        ax_heatmap.axvline(10, 0, 1, color='black', linestyle='--', linewidth=1)
        ax_heatmap.axvline(20, 0, 1, color='black', linestyle='--', linewidth=1)
        # Drawing the frame 
        for spine in ax_heatmap.spines.items(): 
            spine[1].set(visible=True, lw=.8, edgecolor="black")
        ax_bar.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
fig.savefig("results/pics/paper/FigS4_v2_IS.sorted.pdf", format="pdf", dpi=300, bbox_inches='tight')

#new row 1
df_bedpe = create_rnaseq_df(need_coord=True)
df_bedpe['length'] = df_bedpe['end'] - df_bedpe['start']
df_bedpe = df_bedpe.query('length > 2000')
df_bedpe['expr_group'] = df_bedpe.apply(
    lambda x: 'high' if x['0AB']>8 else 
    'average' if x['0AB']>2 and x['0AB']<8 else
    'low', axis=1
)
df_bedpe['length_group'] = df_bedpe.apply(
    lambda x: #'extra-long' if x.length>=12000 else 
    'long' if x.length>=8000 and x.length<20000 else
    'average' if x.length>4000 and x.length<8000 else
    'short', axis=1
)

flank=10_000
resolution=500
time = 0
cooler_path="../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::/resolutions/" % (time)
resolution=500
clr = cooler.Cooler(cooler_path + str(resolution))
df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes[
"end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes['end'] = clr.chromsizes.tolist()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
feature_length = int(df_bedpe.length.median() // resolution)
# pup = coolpup.pileup(clr, df_bedpe, groupby=['expr_group1'], 
# rescale=True, rescale_size=int(1+flank * 2// resolution),  rescale_flank=2,
# local=True,
# features_format='bed', view_df=df_chromsizes, nproc=4, 
# #by_strand=True, 
# flip_negative_strand=True, 
# #expected_df=expected, 
# flank=flank, min_diag=2)

from coolpuppy import plotpup
# row 1
axs1 = subfigs[1].subplots(1, 1)
subfigs[1].text(-0.1, 1.1, string.ascii_lowercase[4], transform=axs1.transAxes, size=15, weight='bold')
subfigs[1].suptitle("\n".join(wrap('Correlation of gene expression and insulation changes', 80)), fontsize=15)
# Load RNA-seq 8h and 0h z-scores
rnaseq8h_binned_2kb = bioframe.read_table('bw/8_merged.TPM_log2_binned.2kb.tsv', schema='bedGraph', comment='#').sort_values(by=['chrom', 'start'])
# RNA-seq 8h z-scores
rnaseq8h_binned_2kb['z_value'] = (rnaseq8h_binned_2kb['value'] - np.nanmean(rnaseq8h_binned_2kb['value'])) / np.nanstd(rnaseq8h_binned_2kb['value'])

rnaseq_binned_2kb = bioframe.read_table('bw/0_merged.TPM_log2_binned.2kb.tsv', schema='bedGraph', comment='#').sort_values(by=['chrom', 'start'])
# RNA-seq 0h z-scores
rnaseq_binned_2kb['z_value'] = (rnaseq_binned_2kb['value'] - np.nanmean(rnaseq_binned_2kb['value'])) / np.nanstd(rnaseq_binned_2kb['value'])


# Load insulation vector data
is_vec_black = bioframe.read_table('bw/IS/0AB.20k.IS.cooltools.bedgraph',schema='bedGraph').sort_values(by=['chrom', 'start'])
# Insulation vector data for 0h
is_vec_8h = bioframe.read_table('bw/IS/8AB.20k.IS.cooltools.bedgraph',schema='bedGraph').sort_values(by=['chrom', 'start'])  # dicty.10k.binsize1.bedgraph #dicty_conv.10k.bg
# Insulation vector data for 8h
is_vec_8h['z_value'] = (is_vec_8h['value'] - np.nanmean(is_vec_8h['value'])) / np.nanstd(is_vec_8h['value'])
is_vec_black['log_value'] = np.log2(is_vec_black['value'])
# Logarithm transform of insulation vector data for 0h
is_vec_black['z_value'] = (is_vec_black['value'] - np.nanmean(is_vec_black['value'][np.isfinite(is_vec_black['value'])])) / np.nanstd(is_vec_black['value'][np.isfinite(is_vec_black['value'])])


# Merge insulation vector data and RNA-seq z-scores
tmp1 = is_vec_black.merge(is_vec_8h, on=['chrom', 'start', 'end'], how='outer', suffixes=('_0h', '_8h'))

# Merge RNA-seq z-scores
tmp2 = rnaseq_binned_2kb.merge(rnaseq8h_binned_2kb, on=['chrom', 'start', 'end'], how='outer', suffixes=('_0h', '_8h'))

# Create dataframe with merged data
rna_is_df = tmp1.merge(tmp2, on=['chrom', 'start', 'end'], how='inner', suffixes=('_is', '_rna'))

# Calculate delta RNA z-score and delta RNA
rna_is_df['delta_rna_zscore'] = (rna_is_df['z_value_8h_rna'] / rna_is_df['z_value_0h_rna'])
rna_is_df['delta_rna'] = rna_is_df['value_8h_rna'] - rna_is_df['value_0h_rna']
rna_is_df['z_value_delta_rna'] = (rna_is_df['delta_rna'] -
                                  np.nanmean(rna_is_df['delta_rna'][np.isfinite(rna_is_df['delta_rna'])]) / \
                                  np.nanstd(rna_is_df['delta_rna'][np.isfinite(rna_is_df['delta_rna'])])
                                  )

# Calculate delta IS and delta IS z-score
rna_is_df['delta_is'] = rna_is_df['value_8h_is'] - rna_is_df['value_0h_is']
rna_is_df['delta_is_zscore'] = (rna_is_df['z_value_8h_is'] / rna_is_df['z_value_0h_is'])
rna_is_df['z_value_delta_is'] = (rna_is_df['delta_is'] -
                                 np.nanmean(rna_is_df['delta_is'][np.isfinite(rna_is_df['delta_is'])]) / \
                                 np.nanstd(rna_is_df['delta_is'][np.isfinite(rna_is_df['delta_is'])])
                                 )
rna_is_df = rna_is_df.sort_values(by=['chrom', 'start'])
rna_is_df = rna_is_df.sort_values(by=['chrom', 'start'])


from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(rna_is_df['delta_rna'][np.isfinite(rna_is_df['delta_is'])], rna_is_df['delta_is'][np.isfinite(rna_is_df['delta_is'])])

g = sns.regplot(x="delta_rna", y="delta_is", data=rna_is_df,
                scatter_kws={'alpha': 0.1},
                line_kws={'label': "Сor={0:.2f}, p-value < 0.01".format(r_value, p_value)}, ax=axs1)
# plot legend
g.legend()
g.set(ylabel="IS at 8h - IS at 0h", xlabel="TPM at 8h - TPM at 0h", xticks=[-5, 0, 5], yticks=[-0.5, 0, 0.5])


subfigs[1].suptitle("\n".join(wrap('Average Hi-C map of genes with different expression levels',80)), fontsize=15)
# row0

from matplotlib.colors import LogNorm, Normalize
from coolpuppy.plotpup import add_heatmap
pupsdf = pup.copy()

max_coordinates = [
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "flank"],
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "resolution"],
    ]

i = 0
for expr_group in ['low', 'average', 'high']:
    add_heatmap_custom(
        pupsdf.loc[pupsdf.expr_group1 == expr_group, 'data'],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "flank"],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "rescale"],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "rescale_flank"],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "n"],
        max_coordinates=max_coordinates,
        cmap='viridis',
        plot_ticks=True,
        height=1,
        aspect="auto",
        color=None,
        norm=LogNorm(0.0015, 0.025),
        stripe=False,
        font_scale=1,
        ax=axs1[i]
    )
    for spine in axs1[i].spines.items(): 
        spine[1].set(visible=True, lw=.8, edgecolor="black")
    if i > 0:
        axs1[i].set_yticks([]) 
        if i == 1:
            axs1[i].set(title="\n".join(wrap("2-8 TPM", 25)), xticks=[-20, -5, 5, 20], yticks=[], xticklabels=[-10, 'TSS', 'TTS', 10],xlabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)))
        if i == 2:
            axs1[i].set(title="\n".join(wrap("> 8 TPM", 25)), xticks=[-20, -5, 5, 20], yticks=[], xticklabels=[-10, 'TSS', 'TTS', 10], xlabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)))
    else:
        axs1[i].set(title="\n".join(wrap("< 2 TPM", 35)), xticks=[-20, -5, 5, 20], xticklabels=[-10, 'TSS', 'TTS', 10], yticklabels=[10, '', -10], yticks=[-20, 0, 20], xlabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)), ylabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)))
        #, xticks=[0, 14.5, 25.5, 40], yticks=[0, 14.5, 25.5, 40], 
    axs1[i].axvline(-5, 0, 1, color='black', linestyle='--', linewidth=1)
    axs1[i].axhline(5, 0, 1, color='black', linestyle='--', linewidth=1)
    i += 1

cax = axs1[3]
import matplotlib.cm as mpl_cm
cb = plt.colorbar(
    mpl_cm.ScalarMappable(LogNorm(0.0015, 0.025), 'viridis'), ticks=[0.0015, 0.025], label='log2 mean observed', cax=cax, format=ticker.FuncFormatter(lambda x, pos: f"{x:.2g}"),
)
cax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
cax.minorticks_off()

# row2

axs2 = subfigs[2].subplots(1, 4, width_ratios=[1.5, 1.5, 1.5, 0.1]) #, sharey='row', ,sharex='all'
subfigs[2].text(-0.5, 1.3, string.ascii_lowercase[5], transform=axs2[0].transAxes, size=15, weight='bold')
subfigs[2].suptitle('Chromatin and expression change 2AB to 8AB', fontsize=15)

# DE genes chromatin change
thres_high = 6
thres_low = 4
tpm_df = create_rnaseq_df(need_coord=True)
for time in [0,2,5,8]:
    tpm_df["%sAB_expr_group" % time] = "average"
    tpm_df.loc[tpm_df["%sAB" % time] > thres_high, "%sAB_expr_group" % time] = "high"
    tpm_df.loc[tpm_df["%sAB" % time] < thres_low, "%sAB_expr_group" % time] = "low"


tpm_df = tpm_df.astype({'start': 'int64', 'end': 'int64'})
paired_sites = tpm_df[tpm_df['end'] - tpm_df['start'] > 2000]
paired_sites = paired_sites[paired_sites['end'] - paired_sites['start'] < np.quantile(paired_sites['end'] - paired_sites['start'], 0.99)]
median_size = np.median(paired_sites.end - paired_sites.start)

pup = {}
for time in [2,8]:
    cooler_path="../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::/resolutions/" % (time)
    resolution=500
    clr = cooler.Cooler(cooler_path + str(resolution))
    df_chromsizes = prepare_view_df(clr)
    expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=0) #, median=True

    flank=10_000

    pup['%sAB' % time] = coolpup.pileup(clr, paired_sites, groupby=['2AB_expr_group1', '8AB_expr_group2'], #['group1'], #'length_group2'
    rescale=True, rescale_size=int(1+flank * 2// resolution), rescale_flank=2,
    local=True, features_format='bed', view_df=df_chromsizes, nproc=4, flip_negative_strand=True,  
    expected_df=expected, 
    flank=flank, min_diag=2)

pup['2AB']['cooler'] = '8AB.filtered'
from coolpuppy.lib.puputils import divide_pups
from coolpuppy_func import add_heatmap_custom, subtract_pups
pup_divide = subtract_pups(pup['2AB'], pup['8AB'])

ymin=-0.35
ymax=0.35

axs2[0].imshow(pup_divide['data'][4], cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), interpolation="none",vmin=ymin, vmax=ymax)
axs2[1].imshow(pup_divide['data'][6], cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), interpolation="none",vmin=ymin, vmax=ymax)
axs2[2].imshow(pup_divide['data'][0], cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), interpolation="none",vmin=ymin, vmax=ymax)

cax = axs2[3]
cbar = plt.colorbar(
    mpl_cm.ScalarMappable(Normalize(ymin, ymax), sns.diverging_palette(145, 300, s=60, as_cmap=True)), label='log2 mean obs/exp', cax=cax, format=ticker.FuncFormatter(lambda x, pos: f"{x:.2g}"), ticks=[ymin, ymax])#ticks=, 
cax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
cax.minorticks_off() 

ticks_pixels = np.linspace(0, flank*2//resolution,5)
ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)
for i in range(3):
    if i == 0:
        axs2[i].set(xticks=[0, 14.5, 25.5, 40], yticks=[0, 14.5, 25.5, 40], xticklabels=[-10, 'TSS', 'TTS', 10], yticklabels=[-10, 'TSS', 'TTS', 10], ylabel="\n".join(wrap('relative position from TSS/TTS, kbp', 20)), title="\n".join(wrap('Genes down-regulated in 2AB', 20)))
    if i == 1:
        axs2[i].set(xticks=[0, 14.5, 25.5, 40], yticks=[], xticklabels=[-10, 'TSS', 'TTS', 10], title="\n".join(wrap('Genes up-regulated in 2AB', 20)))
    if i == 2:
        axs2[i].set(xticks=[0, 14.5, 25.5, 40], yticks=[], xticklabels=[-10, 'TSS', 'TTS', 10], title="\n".join(wrap('Genes with unchanged expression levels', 20)))
    for spine in axs2[i].spines.items(): 
        spine[1].set(visible=True, lw=.8, edgecolor="black")
    axs2[i].axvline(14.5, 0, 1, color='black', linestyle='--', linewidth=1)
    axs2[i].axhline(14.5, 0, 1, color='black', linestyle='--', linewidth=1)



fig.savefig("results/pics/paper/FigS4_v2.pdf", format="pdf", dpi=300, bbox_inches='tight')
# %% Fig S4
# Supplementary Figure 4. a, Related to Fig. 4a: the same picture for another set of parameters to demonstrate the robustness of the analysis

# # preprocessing
# middle_pixel_dict = {}
# for Time in [0, 8]:  #0, 2, 5, 8
#     middle_pixel_dict[str(Time) + 'AB_dev'] = compute_avMiddlePixels(
#         resolution=2000,
#         flank_bins=1,
#         cell_cycle=False,
#         Timing=Time,
#         use_bed='provided_bed',
#         bed_file="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",)  #"data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.filtered.bedpe")
#         # "results/long_loops/0AB_regular_loops.bedpe")
#         #'one_for_all' 'individual'


cm = 1/2.54  # centimeters in inches

values = [
        ['two_strands_per_anchor', 'right', 'sum_only', 0],
        ['two_strands_per_anchor', 'right', 'sum_only', 3],
        # ['two_strands_per_anchor', 'right', 'sum_only', 4],
        #['two_strands_per_anchor', 'right', 'sum_and_subtraction', 0],
        #['two_strands_per_anchor', 'right', 'sum_and_subtraction', 3],
        #['one_strand_per_anchor', 'opposite', '', 0],
        #['one_strand_per_anchor', 'opposite', '', 3.0],
        #['one_strand_per_anchor', 'fellow', '', 3.0]
        ]

fig = plt.figure(layout="constrained", figsize=(18*cm, 12*len(values)*cm))
subfigs = fig.subfigures(len(values), 1, wspace=0.01, hspace=0.01)

for prefix in ['0AB',]:  # '2AB', '5AB', '8AB'
    for i in range(len(values)):
        mode, submode, twostrands_mode, threshold = values[i]
        tmp_genes = compute_LoopExprAndStrength(
            mode=mode,
            submode=submode,
            prefix=prefix,
            twostrands_mode=twostrands_mode,
            loop_df_path="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe"
        )  # two_strands_per_anchor #one_strand_per_anchor
        tmp_genes['0AB_dev'] = middle_pixel_dict['0AB_dev']
        #tmp_genes['2AB_dev'] = middle_pixel_dict['2AB_dev']
        #tmp_genes['5AB_dev'] = middle_pixel_dict['5AB_dev']
        tmp_genes['8AB_dev'] = middle_pixel_dict['8AB_dev']
        tmp_genes['delta_strength'] = tmp_genes['8AB_dev'] - tmp_genes['0AB_dev']

        if mode == 'two_strands_per_anchor':
            column_prefix = prefix + "_sum"
        else:
            column_prefix = prefix
        x_der_argmin = compute_argminDerative(x=column_prefix+'_left', df_genes=tmp_genes)
        y_der_argmin = compute_argminDerative(x=column_prefix+'_right', df_genes=tmp_genes)


        # create groups
        tmp_genes['group'] = 0
        tmp_genes.loc[tmp_genes[column_prefix + '_right'] > y_der_argmin + threshold, 'group'] = 1
        tmp_genes.loc[tmp_genes[column_prefix + '_left'] > x_der_argmin + threshold, 'group'] = 2
        tmp_genes.loc[(tmp_genes[column_prefix + '_left'] > x_der_argmin + threshold) & (tmp_genes[column_prefix + '_right'] > y_der_argmin + threshold), 'group'] = 3
        tmp_genes['group'] = tmp_genes['group'].astype(str).astype("category")
        tmp_genes['group'].cat.reorder_categories(['0', '1', '2', '3'])

        # print(tmp_genes['group'].value_counts())

        # plotting
        axs0 = subfigs[i].subplot_mosaic( #subfigs[0, 0]
            """
            A..F
            B..F
            CDEF
            """,
            #sharex=True,
            # set the height ratios between the rows
            height_ratios=[12, 12, 24, ],
            # set the width ratios between the columns
            width_ratios=[12, 4, 4, 8],
            gridspec_kw={'wspace': 0.001, 'hspace': 0.001,},
        )
        axs0['A'].text(-0.1, 1.1, string.ascii_lowercase[i], transform=axs0['A'].transAxes, size=15, weight='bold')
        # ['two_strands_per_anchor', 'right', 'sum_and_subtraction', 0],
        # ['two_strands_per_anchor', 'right', 'sum_and_subtraction', 7],
        # ['one_strand_per_anchor', 'opposite', '', 0],
        # ['one_strand_per_anchor', 'opposite', '', 7.0]
        if i == 0:
            subfigs[i].suptitle("\n".join(wrap('Convergence gene expression around anchors, threshold=first derivative + 3', 60)))
        if i == 1:
            subfigs[i].suptitle("\n".join(wrap('Convergence gene expression around anchors, threshold=first derivative + 4', 60)))
        if i == 2:
            subfigs[i].suptitle("\n".join(wrap('Convergence gene expression - divergent gene expression around anchors, threshold=first derivative', 60)))
        if i == 3:
            subfigs[i].suptitle("\n".join(wrap('Convergence gene expression - divergent gene expression around anchors, threshold=first derivative+3', 60)))
        # r'$e(t_0)$ [s]'
        
        g = sns.scatterplot(data=tmp_genes,
                            x=column_prefix+"_left", #FC_8_
                            y=column_prefix+"_right",
                            alpha=0.5,
                            hue=prefix+"_dev", #group
                            # hue="delta_elscore",
                            palette="viridis",
                            legend=False,
                            ax=axs0['C']
                            )
        # axs0['C'].spines["outline"].set(visible=True, lw=.8, edgecolor="black")
        # ax.vlines(x=threshold, ls='--', ymin=-4, ymax=11)
        # ax.hlines(y=threshold, ls='--', xmin=-4, xmax=11)

        norm = plt.Normalize(tmp_genes['0AB_dev'].min(), tmp_genes['0AB_dev'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        
        # Remove the legend and add a colorbar
        # axs0['C'].get_legend().remove()

        cbar_ax = subfigs[i].add_axes([.39, .16, .01, .1])
        g.figure.colorbar(sm, cax=cbar_ax)

        # derivative argmin
        axs0['C'].set(xticks=[-5, 0, 5, 10], yticks=[-5, 0, 5, 10], xticklabels=[-5, 0, 5, 10], yticklabels=[-5, 0, 5, 10], xlabel='RNA-seq log2(TPM) at 5\' anchor', ylabel="\n".join(wrap('RNA-seq log2(TPM) at 3\' anchor', 12)))
        axs0['C'].text(tmp_genes[column_prefix+'_left'].max()-1, tmp_genes[column_prefix+'_right'].max()-1, "Group 4 \n (n=%s)" % tmp_genes['group'].value_counts()['3'], horizontalalignment='left', verticalalignment='bottom', fontsize=8)
        axs0['C'].text(tmp_genes[column_prefix+'_left'].max()-1, tmp_genes[column_prefix+'_right'].min()+1, "Group 3 \n (n=%s)" % tmp_genes['group'].value_counts()['2'], horizontalalignment='left', verticalalignment='top', fontsize=8)
        axs0['C'].text(tmp_genes[column_prefix+'_left'].min()+1, tmp_genes[column_prefix+'_right'].max()-1, "Group 2 \n (n=%s)" % tmp_genes['group'].value_counts()['1'], horizontalalignment='right', verticalalignment='bottom', fontsize=8)
        axs0['C'].text(tmp_genes[column_prefix+'_left'].min()+1, tmp_genes[column_prefix+'_right'].min()+1, "Group 1 \n (n=%s)" % tmp_genes['group'].value_counts()['0'], horizontalalignment='right', verticalalignment='top', fontsize=8)

        ax_box = sns.boxplot(x="group",
                            #y="delta_strength",
                            y=prefix + "_dev",
                            data=tmp_genes, ax=axs0['F'])
        from scipy.stats import mannwhitneyu
        for g1, g2 in [("0", "3"), ("1", "3"), ("2", "3")]:
            U1, p_value = mannwhitneyu(tmp_genes.query('group == @g1')['0AB_dev'].tolist(), tmp_genes.query('group == @g2')['0AB_dev'].tolist())
            print('for %s vs. %s: Mann-Whitney p = %s' % (g1, g2, p_value))
        # add_stat_annotation(ax_box, data=tmp_genes, x='group', y=prefix + '_dev', box_pairs=[("1", "3"), ("2", "3"), ("0", "3")], test='Mann-Whitney-ls', text_format='star', loc='inside', verbose=2)
        axs0['F'].set(xticklabels=['1', '2', '3', '4'], xlabel='Group', ylabel='Loop strength')

        # plot density & derivative
        for x in [column_prefix+'_left', column_prefix+'_right']:
            der, X_plot, log_dens = compute_Derivative(x=x, df_genes=tmp_genes)
            x_der_argmin = compute_argminDerative(x=x, df_genes=tmp_genes)
            df = pd.DataFrame({'x': X_plot[:, 0].tolist(), 'exp': np.exp(log_dens).tolist(), 'der': der.tolist()})
            if x == column_prefix+'_left':
                g.vlines(x=x_der_argmin+threshold, ls='--', ymin=tmp_genes[x].min()-2, ymax=tmp_genes[x].max()+2, colors='black')
                axs0['A'].fill(X_plot[:, 0], np.exp(log_dens), fc="#b84873")
                axs0['A'].plot(X_plot[:, 0], np.exp(log_dens), lw=1, color='black')
                ax=axs0['B']
                ax.plot(X_plot[:, 0], der, lw=1, path_effects=[pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()]) #, alpha=0.5
                
                axs0['A'].set(ylabel='Freq.', xlabel=None, xlim=(tmp_genes[x].min()-2, tmp_genes[x].max()+2), xticklabels=[])
                axs0['B'].set(ylabel='Slope', xlabel=None, xlim=(tmp_genes[x].min()-2, tmp_genes[x].max()+2), xticklabels=[])
                # axs0['C'].yaxis.set_tick_params(which='both', labelbottom=True)
                # plt.setp(axs0['A'].get_xticklabels(), visible=False)
                # plt.setp(axs0['B'].get_xticklabels(), visible=False)
                # plt.setp(axs0['C'].get_xticklabels(), visible=True)
                
                axs0['B'].hlines(y=0, ls='--', xmin=tmp_genes[x].min()-2, xmax=tmp_genes[x].max()+2, colors='black')
                axs0['B'].vlines(x=x_der_argmin+threshold, ls='--', ymin=-0.08, ymax=0.09, colors='black')
                axs0['A'].yaxis.set_major_locator(ticker.FixedLocator([0, 0.1]))
                axs0['B'].yaxis.set_major_locator(ticker.FixedLocator([0]))
                # axs0['B'].set_path_effects([pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
                
            if x == column_prefix+'_right':
                g.hlines(y=x_der_argmin+threshold, ls='--', xmin=tmp_genes[x].min()-2, xmax=tmp_genes[x].max()+2, colors='black')
                axs0['E'].fill_between(X_plot[:, 0], np.exp(log_dens), fc="#b84873")
                lineplot_plusplus(x='x',y='exp',data=df, orientation = "vertical", ax=axs0['E'], lw=1, color='black')
                # axs0['E'].set_ylim([0, 0.15])
                ax=axs0['D']
                lineplot_plusplus(x='x',y='der',data=df, orientation = "vertical", ax=ax, lw=1, path_effects=[pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()])
                # ax.plot(X_plot[:, 0], der, alpha=0.5)
                axs0['D'].vlines(x=0, ls='--', ymin=tmp_genes[x].min()-2, ymax=tmp_genes[x].max()+2, colors='black')
                axs0['D'].hlines(y=x_der_argmin+threshold, ls='--', xmin=-0.08, xmax=0.09, colors='black')
                axs0['D'].xaxis.set_major_locator(ticker.FixedLocator([0]))
                #axs0['D'].sharey(axs0['C'])
                #axs0['E'].sharey(axs0['C'])
                axs0['E'].set(xlabel='Freq.', ylabel=None, ylim=(tmp_genes[x].min()-2, tmp_genes[x].max()+2), yticklabels=[])
                axs0['D'].set(xlabel='Slope', ylabel=None, ylim=(tmp_genes[x].min()-2, tmp_genes[x].max()+2), yticklabels=[])
                # plt.setp(axs0['D'].get_yticklabels(), visible=False)
                # plt.setp(axs0['E'].get_yticklabels(), visible=False)
                # plt.setp(axs0['C'].get_yticklabels(), visible=False)

# fig.savefig("results/pics/paper/FigS4_v2_add.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()

# %% fix to Fig 2b
ConvPair = bioframe.read_table('data/genome/genesConvPairs.byExpr.byAnchors.thr2.bed', schema='bed')
df_loops = load_BedInMode("data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed", '0', 'anchors_3bins', resolution=2000)
N_loopAnchor = bioframe.overlap(ConvPair, df_loops, how='inner')[['chrom', 'start', 'end']].drop_duplicates().shape[0]

df_loops = load_BedInMode("data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed", '0', 'inside', resolution=2000)
N_loopInside = bioframe.overlap(ConvPair, df_loops, how='inner')[['chrom', 'start', 'end']].drop_duplicates().shape[0]

# data = [N_loopAnchor, N_loopInside, ConvPair.shape[0] - N_loopInside - N_loopAnchor]
labels = ['Loop anchors', 'Inside loops','Outside loops']
# using bedtools
# bedtools intersect -a data/genome/genesConvPairs.byExpr.byAnchors.thr2.bed -b data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bedpe -wa | uniq | wc -l
# bedtools intersect -a data/genome/genesConvPairs.byExpr.byAnchors.thr2.bed -b data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bedpe -wa -v | uniq | bedtools intersect -a stdin -b data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed -wa | uniq | wc -l
data =[1501, 1728, 1079]
def get_abs(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d})"

plt.pie(data, labels = labels, autopct=lambda pct: get_abs(pct, data), wedgeprops = {"edgecolor" : "black", 'linewidth': 1, 'antialiased': True})
plt.savefig("results/pics/paper/Fig2b_fix.pdf", format="pdf", dpi=300, bbox_inches='tight')

# %% Fig S5 
# a, CGP enrichment at loop anchors, inside and outside loops. 
# b, Size distribution of ELEs. 
# c, Representative examples of ELEs. H3K27ac, H3K4me1, ATAC-seq profiles and gene positions are shown. 
# d, Distributions of expression level of genes which contain and do not contain ELEs. p-value in the Mann-Whitney U test. 
# e, Averaged interaction between DGPs separated by different genomic distances.

# Supplementary Figure 5. a, Related to Fig. 5b: CGP enrichment within loops and outside loops. b, Averaged Hi-C maps of divergent CGPs.
# 
fig = plt.figure(layout="constrained", figsize=(18*cm, 15*cm))

subfigs = fig.subfigures(3, 1, wspace=0.07, height_ratios=[1.5, 1.2,1])
subfigs[0].suptitle('Convergence Gene Pairs within loop anchor, loop interior and outside loops', fontsize=10)
axs0 = subfigs[0].subplots(1, 3)
# axs0[0].set(title="\n".join(wrap('CGPs within and outside loops', 25)))

def get_abs(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d})"

N_shuffle = 1000
i = 0
for mode in ['anchors_3bins', 'inside-1bin']:
    axs0[i].text(-0.1, 1.1, string.ascii_lowercase[i], transform=axs0[i].transAxes, size=15, weight='bold')
    inter_shuffle_df, inter_lasLoops = pValue_featureOccurenceInLoop(
        file_loops="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed",
        time=None,
        mode=mode,
        N_shuffle=N_shuffle,
        file_features='results/convPairs.Mar14.bed', 
        name="ConvPairsAll75_%s." % mode, 
        genome_file="data/genome/dicty.chrom.sizes",
        pic_path="results/pics/paper",
        resolution=2000, fraction=0.75, return_df=True)
    sns.histplot(
            data=inter_shuffle_df, x="Shuffle", kde=True, stat="percent", binwidth=1, ax=axs0[i])
    axs0[i].axvline(inter_lasLoops.shape[0], color="red", lw=3)
    p_value = np.round(np.min(
            [len(inter_shuffle_df[inter_shuffle_df["Shuffle"] > inter_lasLoops.shape[0]]) / N_shuffle,
            len(inter_shuffle_df[inter_shuffle_df["Shuffle"] < inter_lasLoops.shape[0]]) / N_shuffle,]), 3,)
    if p_value == 0:
        p_value = 1 / N_shuffle
        axs0[i].annotate(
            "p-value < " + str(p_value),
            xy=(0.75, 0.75),
            xycoords="axes fraction",
            #bbox=dict(facecolor="pink", alpha=0.3),
            horizontalalignment="right",
            fontsize=8,
        )
    else:
        axs0[i].annotate(
            "p-value = " + str(p_value),
            xy=(0.75, 0.75),
            xycoords="axes fraction",
            #bbox=dict(facecolor="pink", alpha=0.3),
            horizontalalignment="right",
            fontsize=8,
        )
    i += 1
axs0[2].set(title="\n".join(wrap('ELE length distribution', 25)))
axs0[2].text(-0.1, 1.1, string.ascii_lowercase[2], transform=axs0[2].transAxes, size=15, weight='bold')
# subfigs[1].suptitle('Average divergent gene pairs', fontsize=10)
V_enh = bioframe.read_table('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed')#.query('score == "none"')
V_enh['length'] = V_enh.end - V_enh.start
sns.boxplot(data=V_enh, y='length', ax=axs0[2], log_scale=True, showfliers=False)
# axs0[0].pie(data, labels = labels, autopct=lambda pct: get_abs(pct, data), wedgeprops = {"edgecolor" : "black", 'linewidth': 1, 'antialiased': True}, colors=susielu, textprops={'fontsize': 10}) #colors = colors, 

# histplot RNA-seq
# axs0[1].text(-0.1, 1.1, string.ascii_lowercase[1], transform=axs0[1].transAxes, size=15, weight='bold')
# axs0[1].set(title="\n".join(wrap('Gene expression distribution', 45)))
# tpm_df = create_rnaseq_df(need_coord=True)
# tpm_df['log2_0AB'] = np.log2(tpm_df['0AB']+ 0.01)

# sns.set_style('white')
# sns.histplot(tpm_df, x="log2_0AB", kde=True, ax=axs0[1], stat="density")
# axs0[1].axvline(1, c='black', ls=':',linewidth=2.0)
# axs0[1].axvline(3, c='black', ls=':', linewidth=2.0)
# axs0[1].set_yticks([0,0.4])
# axs0[1].set_xticks([-6, -3, 0, 3, 6])

# row 1
axs1 = subfigs[1].subplots(1, 2, width_ratios=[3.5, 1.0])
axs1[0].text(-0.1, 1.1, string.ascii_lowercase[3], transform=axs1[0].transAxes, size=15, weight='bold')


axs1[1].text(-0.1, 1.1, string.ascii_lowercase[4], transform=axs1[1].transAxes, size=15, weight='bold')
axs1[0].axis('off')

# Distributions of expression level of genes which contain and do not contain ELEs. p-value in the Mann-Whitney U test. 
tpm_df = create_rnaseq_df(need_coord=True)
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end', 'name']].sort_values(by=['chrom', 'start']))

time='V'
V_enhancer_noPromoter_bed = BedTool('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time))
genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed, f=0.2).to_dataframe().name.tolist()

tpm_df['group'] = 'w/o ELE'

tpm_df_enh_tmp = tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec),:]
tpm_df_enh_tmp.loc[:, 'group'] = 'with ELE'

tpm_df = pd.concat([tpm_df, tpm_df_enh_tmp], ignore_index=True)
time_int=0
tpm_df['log2_%sAB' % str(time_int)] = np.log2(tpm_df[str(8) + 'AB']+0.01)

sns.violinplot(data=tpm_df, y='log2_0AB', x='group', ax=axs1[1], order=['w/o ELE', 'with ELE'])
axs1[1].set(title="\n".join(wrap("Expression of genes with/without ELE", 25)), ylabel="log10(TPM)", xlabel="")
# add_stat_annotation(g, data=tpm_df, x='group', y='log2_0AB_introns', #order=order,
#                     box_pairs=[('w/o enhancer', 'with enhancers')],
#                     test='Mann-Whitney-ls', text_format='star', loc='inside', verbose=2)
# g.tick_params(axis='x', rotation=30)
# wrap_labels(g, 10)
from scipy.stats import mannwhitneyu
from scipy import stats
tmp1 = tpm_df.query('group == "w/o ELE"')['0AB'].tolist()
tmp2 = tpm_df.query('group == "with ELE"')['0AB'].tolist()
U1, p_value = mannwhitneyu(tmp1, tmp2)
axs1[1].annotate(
            "p-val = " + str(p_value),
            xy=(0.95, 0.75),
            xycoords="axes fraction",
            #bbox=dict(facecolor="pink", alpha=0.3),
            horizontalalignment="right",
            fontsize=6,
        )
# row 2
axs2 = subfigs[2].subplots(1, 5, gridspec_kw={'width_ratios':[1,1,1,1, 0.2]})
# e, Averaged interaction between DGPs separated by different genomic distances.
resolution = 500
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
clr = cooler.Cooler(cooler_path + str(resolution))

df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

flank = 50000

try:
    expected
except NameError:
    expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=2)
thres_high = 8
thres_low = 2
step = 5
from coolpuppy_func import create_DivPairs_exprThres
tmp = create_DivPairs_exprThres(thres_low=thres_low, thres_high=thres_high, intervals_low=False, step=step, return_df=True).rename(columns={"chrom+": "chrom"}).query('anchor_group == "none"')
pairs_df = tmp.loc[:,['chrom', 'end-', 'start+']].rename(columns={"end-": "start", "start+": "end"}) #loc[:,['chrom', 'start', 'end']] #tmp

# pup = coolpup.pileup(clr, pairs_df, features_format='bed', view_df=df_chromsizes, expected_df=expected, flip_negative_strand=True, mindist=5000, maxdist = 45_000, by_distance=[5000, 15_000, 25_000, 35_000, 45_000], flank=20_000, min_diag=2, nproc=4)
# plotpup.plot(pup, score=True, cmap='coolwarm', scale='log', sym=True, vmax=1.6, vmin=0.8, height=3)

from matplotlib.colors import LogNorm, Normalize
from coolpuppy.plotpup import add_heatmap
pupsdf = pup.copy()

max_coordinates = [
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "flank"],
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "resolution"],
    ]
axs2[0].text(-0.1, 1.1, string.ascii_lowercase[5], transform=axs2[0].transAxes, size=15, weight='bold')
i = 0
for expr_group in ['0.005Mb-\n0.015Mb', '0.015Mb-\n0.025Mb', '0.025Mb-\n0.035Mb', '0.035Mb-\n0.045Mb']:
    add_heatmap_custom(
        pupsdf.loc[pupsdf.separation == expr_group, 'data'],
        pupsdf.loc[pupsdf.separation == expr_group, "flank"],
        pupsdf.loc[pupsdf.separation == expr_group, "rescale"],
        pupsdf.loc[pupsdf.separation == expr_group, "rescale_flank"],
        pupsdf.loc[pupsdf.separation == expr_group, "n"],
        max_coordinates=max_coordinates,
        cmap='coolwarm',
        plot_ticks=True,
        height=1,
        aspect="auto",
        color=None,
        norm=LogNorm(0.8, 1.6),
        stripe=False,
        font_scale=1,
        ax=axs2[i]
    )
    for spine in axs2[i].spines.items(): 
        spine[1].set(visible=True, lw=.8, edgecolor="black")
    if i > 0:
        axs2[i].set_yticks([]) 
    i += 1
    #     if i == 1:
    #         axs1[i].set(title="\n".join(wrap("2-8 TPM", 25)), xticks=[-20, -5, 5, 20], yticks=[], xticklabels=[-10, 'TSS', 'TTS', 10],xlabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)))
    #     if i == 2:
    #         axs1[i].set(title="\n".join(wrap("> 8 TPM", 25)), xticks=[-20, -5, 5, 20], yticks=[], xticklabels=[-10, 'TSS', 'TTS', 10], xlabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)))
    # else:
    #     axs1[i].set(title="\n".join(wrap("< 2 TPM", 35)), xticks=[-20, -5, 5, 20], xticklabels=[-10, 'TSS', 'TTS', 10], yticklabels=[10, '', -10], yticks=[-20, 0, 20], xlabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)), ylabel="\n".join(wrap('relative position from TSS/TTS, kbp', 25)))
    #     #, xticks=[0, 14.5, 25.5, 40], yticks=[0, 14.5, 25.5, 40], 
    # axs1[i].axvline(-5, 0, 1, color='black', linestyle='--', linewidth=1)
    # axs1[i].axhline(5, 0, 1, color='black', linestyle='--', linewidth=1)
    # i += 1

cax = axs2[4]
import matplotlib.cm as mpl_cm
cb = plt.colorbar(
    mpl_cm.ScalarMappable(LogNorm(0.8, 1.6), 'coolwarm_pallete'), ticks=[0.5, 1.3], label='Contact freq., obs/exp', cax=cax, format=ticker.FuncFormatter(lambda x, pos: f"{x:.2g}"),
)
cax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
cax.minorticks_off()
fig.savefig("results/pics/paper/FigS5_v2.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()

# %% Fig S6
# Supplementary Figure 5. Averaged Hi-C maps centered at divergent gene pairs (DGPs) with different levels of transcription (genes transcribed at high and low levels are highlighted with dark gray and light gray, respectively).

from coolpuppy import plotpup

resolution = 500
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
clr = cooler.Cooler(cooler_path + str(resolution))

df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

try:
    expected
except NameError:
    expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=2)
thres_high = 8
thres_low = 2
step = 5
from coolpuppy_func import create_DivPairs_exprThres
DivPairs = create_DivPairs_exprThres(thres_low=thres_low, thres_high=thres_high, intervals_low=False, step=step, return_df=True).rename(columns={"chrom+": "chrom"})#.query('anchor_group == "loop anchor"')
df_loops = load_BedInMode("data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed", '0', 'anchors_3bins', resolution=2000)
DivPairs_intergenic = DivPairs.copy()
DivPairs_intergenic['start'] = DivPairs_intergenic['end-']
DivPairs_intergenic['end'] = DivPairs_intergenic['start+']
tmp = bioframe.overlap(DivPairs_intergenic, df_loops, how='inner', return_index=True, return_overlap=True, suffixes=('_1', '_2'))
tmp['overlap_length'] = tmp.overlap_end - tmp.overlap_start

tmp['intergenic_length'] = tmp.end_1 - tmp.start_1
tmp['intergenic_fraction'] = tmp.overlap_length / tmp.intergenic_length

DivPairs['anchor_group_intergenic'] = 'none'
DivPairs['loop_id'] = np.nan
DivPairs.loc[tmp.query('intergenic_fraction == 1.0')['index_1'].unique().tolist(), 'anchor_group_intergenic'] = 'loop anchor'

pairs_df = DivPairs.loc[:,['chrom', 'start', 'end', 'strand+', 'expr_group', 'anchor_group_intergenic']].rename(columns={"end-": "start", "start+": "end", "strand+": "strand"}) #loc[:,['chrom', 'start', 'end']] #tmp
pairs_df = pairs_df.astype({'start': 'int64', 'end': 'int64'})

flank = 50_000
pup = coolpup.pileup(clr, pairs_df, groupby=['expr_group1', 'anchor_group_intergenic2'], #['group1'], #'length_group2'
rescale=True, rescale_size=int(1+flank * 2// resolution), rescale_flank=10,
local=True,
features_format='bed', view_df=df_chromsizes, nproc=4, 
flip_negative_strand=True, 
expected_df=expected, 
flank=flank, min_diag=2)
plotpup.plot(pup,
             cols='expr_group1', col_order=['both low', 'both high', 'high & low'],
             rows='anchor_group_intergenic2', row_order=['loop anchor'],
             #cols='orientation', #col_order=['--', '++'],
             score=False, 
             cmap='coolwarm', #'viridis', #'coolwarm', #
             scale='log',
             #sym=True,
             vmax=1.2, vmin=0.5,
             #vmin=0.0015, vmax=0.025,
             height=5)
# %%
import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from coolpuppy_func import add_heatmap_custom
fig = plt.figure(layout="constrained", figsize=(18*cm, 6*cm))

subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[1, 0.1])
axs0 = subfigs[0].subplots(1, 4, width_ratios=[1,1,1, 0.2])
pupsdf = pup.query('anchor_group_intergenic2 == "loop anchor"').copy()
max_coordinates = [
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "flank"],
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "resolution"],
    ]
axs0[0].text(-0.1, 1.1, string.ascii_lowercase[0], transform=axs0[0].transAxes, size=15, weight='bold')
i = 0
for expr_group in ['both low', 'both high', 'high & low']: #['0.005Mb-\n0.015Mb', '0.015Mb-\n0.025Mb', '0.025Mb-\n0.035Mb', '0.035Mb-\n0.045Mb']:
    add_heatmap_custom(
        pupsdf.loc[pupsdf.expr_group1 == expr_group, 'data'],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "flank"],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "rescale"],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "rescale_flank"],
        pupsdf.loc[pupsdf.expr_group1 == expr_group, "n"],
        max_coordinates=max_coordinates,
        cmap='coolwarm',
        plot_ticks=True,
        height=1,
        aspect="auto",
        color=None,
        norm=LogNorm(0.5, 1.2),
        stripe=False,
        font_scale=1,
        ax=axs0[i]
    )
    axs0[i].axvline(-8, 0, 1, color='black', linestyle='--', linewidth=1)
    axs0[i].axvline(8, 0, 1, color='black', linestyle='--', linewidth=1)
    axs0[i].axhline(-8, 0, 1, color='black', linestyle='--', linewidth=1)
    axs0[i].axhline(8, 0, 1, color='black', linestyle='--', linewidth=1)
    for spine in axs0[i].spines.items(): 
        spine[1].set(visible=True, lw=.8, edgecolor="black")
    if i > 0:
        axs0[i].set_yticks([]) 
    i += 1
cax = axs0[3]
import matplotlib.cm as mpl_cm
cb = plt.colorbar(
    mpl_cm.ScalarMappable(LogNorm(0.5, 1.2), 'coolwarm'), ticks=[0.5, 1.2], label='Contact freq., obs/exp', cax=cax, format=ticker.FuncFormatter(lambda x, pos: f"{x:.2g}"),)
cax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
cax.minorticks_off()
plt.savefig("results/pics/paper/FigS6_v2_newLA.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# %%
mtx_HL = pup.iloc[5,2]
mtx_HH = pup.iloc[4,2]
mtx_subtr = np.subtract(mtx_HL, mtx_HH)
plt.imshow(
    mtx_subtr,#151
    vmin = -0.15,
    vmax = 0.15,
    cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), #'coolwarm', #viridis 
    interpolation='none')

plt.colorbar(label = 'Contact freq., obs/exp')
ticks_pixels = np.linspace(0, flank*2//resolution,5)
ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)
plt.xticks(ticks_pixels, ticks_kbp)
plt.yticks(ticks_pixels, ticks_kbp)
plt.xlabel('relative position, kbp')
plt.ylabel('relative position, kbp')
# plt.axvline(x=91, color='k', linestyle='--')
# plt.axvline(x=109, color='r', linestyle='--')
# plt.axhline(y=5, color='r', linestyle='--')
plt.savefig('results/pics/paper/HH_HL_subtract.flank50kb.DGP.newLA.pdf', dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
# %%

# Supplementary Figure 1. a, Size distribution of DpnII restriction fragments in D. discoideum, S. cerevisiae, D. melanogaster and H. sapiens genomes. GC content of the genome is shown below the boxplots. b, Clustering of Hi-C replicates. c, Cis/trans ratio in Dcity compared to other organisms. d, Whole-genome Hi-C maps at the Dicty development stages. e, Eigenvector profiles across Dicty chromosomes. f, Network of las-loops identified in free-living cells and late aggregates.

# Supplementary Figure 2. a, Fig. 2B for the four development stages and cell cycle phases. b, IS profiles at various slide insulation squares.

# Supplementary Figure 3.  a-c, Pileups for panels 3e-g. d, Scatter IS vs RNA-seq TPM within a 2-kb bin (delta?).

# Supplementary Figure 4. a, Related to Fig. 4a: the same picture for another set of parameters to demonstrate the robustness of the analysis

# Supplementary Figure 5. a, Related to Fig. 5b: CGP enrichment within loops and outside loops. b, Averaged Hi-C maps of divergent CGPs.

# Supplementary Figure 6. a, 

# Supplementary Figure 7. a, Heatmap representation of gene expression level in free-living, migrating and aggregating Dicty cells. b, Venn diagram of differentially expressed genes identified in pairwise comparisons of development stages.


# %%
from matplotlib.colors import LogNorm
vmax = 1000
norm = LogNorm(vmin=1, vmax=100_000)
f, axs = plt.subplots(
    figsize=(13, 10),
    nrows=2, 
    ncols=2,
    sharex=True, sharey=True)
clr = cooler.Cooler(f'../dicty_hic_distiller/subsampled/0AB.filtered.mcool::resolutions/100000')
ax = axs[0, 0]
ax.set_title('Pumpkin Spice')
im = ax.matshow(clr.matrix(balance=False)[:], vmax=vmax, cmap='coolwarm_pallete')
# %%
flank = 30_000
# expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=2)
stack = cooltools.pileup(clr, conv_antipeaks.query('loop_inside == "inside"'), view_df=df_chromsizes, expected_df=expected, flank=flank)
# Mirror reflect snippets when the feature is on the opposite strand
# mask = np.array(sites.strand == '-', dtype=bool)
# stack[:, :, mask] = stack[::-1, ::-1, mask]
fig, axs = plt.subplots(10, 10)
k = 0
l = 0
for i in range(10, 110):
    if k > 9:
        k = 0
        l += 1
    ax = axs[k, l]
    img = ax.matshow(
        np.log2(stack[:, :, i]),  
        vmin=-1,
        vmax=1,
        extent=[-flank//1000, flank//1000, -flank//1000, flank//1000],
        cmap='coolwarm'
    )
    k += 1
    ax.xaxis.tick_bottom()
    # if k > 0:
    ax.yaxis.set_visible(False)
    # if l != 9:
    ax.xaxis.set_visible(False)
    # plt.title(f'{i+1}-th snippet from top \n FC score: {sites.loc[i, "FC_score"]:.2f}\n and motif score: {sites.loc[i, "score"]:.2f}')
    ax.axvline(0, c='black', ls=':')
    ax.axhline(0, c='black', ls=':')
fig.savefig("results/pics/paper/divPairInsideLoop_flank30k.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
# %% Supplementary Table 3. 
# Loop coordinates in vegetative cells, migrating, and aggregating Dicty cells. The following parameters are listed: left coordinate of the 2-kb genomic bin identified as the 5’-anchor, right coordinate of the 2-kb genomic bin identified as the 3’-anchor, loop type, development stage, loop score (may be interpreted as the loop strength).

import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import create_rnaseq_df, compute_avMiddlePixels
from elscore import compute_fc_flames, compute_flames
# import elscore
flank = 20000
resolution = 2000
q = 0.2
loops_array = []
Time = 2
for Time in ['0', '2', '5', '8']: # 
    bedpe_file = "data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bedpe" % Time
    df_bedpe = bioframe.read_table(bedpe_file, schema='bedpe')
    
    df_bedpe['loop_score'] = compute_avMiddlePixels(
        resolution=resolution,
        flank_bins=2,
        cell_cycle=False,
        Timing=Time,
        use_bed= "provided_bed",
        bed_file=bedpe_file)

    cooler_path="../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::/resolutions/" % Time 
    clr = cooler.Cooler(cooler_path + str(resolution))
    df_chromsizes = clr.chromsizes.reset_index()
    df_chromsizes.columns = ['chrom', 'end']
    df_chromsizes.loc[:, 'start'] = 0
    df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes[
    "end"].astype(str)
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes['end'] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

    expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000)
    oe_stack = cooltools.pileup(clr, df_bedpe, view_df=df_chromsizes,
                                    expected_df=expected, flank=flank)
    fc_flames = compute_fc_flames(oe_stack, flank=flank, resolution=resolution, pad=3)

    # create fc column
    df_bedpe['FC'] = fc_flames.tolist()
    df_bedpe['el_group'] = 'regular'
    df_bedpe.loc[df_bedpe['FC'] > np.nanquantile(fc_flames, 1 - q), 'el_group'] = '5\'-el-loop'
    df_bedpe.loc[df_bedpe['FC'] < np.nanquantile(fc_flames, q), 'el_group'] = '3\'-el-loop'
    df_bedpe['Time'] = '%sh' % Time
    loops_array.append(df_bedpe)

loops_df = pd.concat([i for i in loops_array])
loops_df.reset_index(drop=True, inplace=True)
loops_df.drop(['name', 'score', 'strand1', 'strand2',], axis = 1).to_csv('results/pics/paper/!Suppl_table/SupplTab3-loops.tsv', index=False, sep = '\t')
# %% Supplementary Table 5. Loops containing genes with highly similar networks of gene ontology terms, as revealed by the best match average (BMA) analysis (BMA-loops). The following parameters are listed: loop coordinates, median BMA-value, key GO terms. Add FC over Ctrl; add p-value.
import re, seaborn as sns, numpy as np, pandas as pd, random

from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests, fdrcorrection

loop_features_df = pd.read_table('results/skewModMW_KS_FCmean.SingleLoop.tsv',
                                 sep='\t')
loop_features_df.columns = ['skewtest', 'modality', 'MW', "KS", 'FC', 'LoopMean']
loop_features_df.dropna(subset=['KS', 'MW'], inplace=True)
loop_features_df['log10_MW_fdr'] = -np.log10(fdrcorrection(loop_features_df.MW)[1])
loop_features_df['log10_KS'] = -np.log10(fdrcorrection(loop_features_df.KS)[1])

loop_features_df['group'] = 'ctrl'
loop_features_df.loc[loop_features_df['log10_MW_fdr'] > 1.3, 'group'] = 'interest'
loop_features_df = loop_features_df.drop(['MW', "KS"], axis=1)
loop_toUse_id = loop_features_df.query('log10_MW_fdr > 1.3 or log10_KS > 1.3').index.tolist()
loops_thresBMA = loops_df.iloc[loop_toUse_id, :].loc[:, ['chrom', 'start', 'end']]
# loops_thresBMA['id'] = loops_thresBMA.index
loops_thresBMA_bed = BedTool.from_dataframe(loops_thresBMA)

loops_thresBMA = loops_thresBMA.join(loop_features_df.loc[:,['FC', 'log10_MW_fdr', 'log10_KS']], how='inner')

# %%
single_loops_BMA_df = pd.read_table('results/single_loops_BMA_df_zeroLevel.txt')
bma_loops = single_loops_BMA_df.query('mode == "loop"').groupby(['loop_id'])['BMA'].median()
# loops_thresBMA = loops_thresBMA.join(bma_loops, how='inner')
loops_thresBMA.query('BMA > 0 and FC > 1').to_csv('results/pics/paper/!Suppl_table/SupplTab5-BMAloops.tsv', index=False, sep = '\t')
# %%
bma_genomewide = pd.read_table('results/bma_genomewide.db.tsv',
                               index_col=0)#.loc[0:1000,:]
tpm_df = create_rnaseq_df(need_coord=True)
for id in range(loops_thresBMA.shape[0]):
    # loops_thresBMA_id = loops_thresBMA.iloc[[id], ]#.loc[:, ['chrom', 'start', 'end']]
    # loops_thresBMA_bed = BedTool.from_dataframe(loops_thresBMA)
    loops_thresBMA_withGenes = bioframe.overlap(loops_thresBMA.iloc[[id],:], tpm_df, suffixes=('_loop', '_tpm'))
    genes_atloops_thresBMA = loops_thresBMA_withGenes['name_tpm'].tolist()
    bma_genes_atloop = bma_genomewide.query(
        'gene1 == @genes_atloops_thresBMA and gene2 == @genes_atloops_thresBMA') # and bma > 0.1
    loops_thresBMA.loc[loops_thresBMA['id'] == id, 'bma'] = np.nanmedian(bma_genes_atloop.bma)
    bma_genes_atloop_arr.append(bma_genes_atloop)
bma_genes_atloops = pd.concat(bma_genes_atloop_arr)


import gseapy as gp
from goatools.go_enrichment import GOEnrichmentStudy
from statsmodels.stats.multitest import fdrcorrection
from go_func import *
from go_func import plot_gos_uncorrected, GODagSmallPlot_uncorrected

single_loops_enr2GO_list = list()
for i in [49]: # , 164, 461#range(loops_df.shape[0])[93:]:
    loop_bed = BedTool.from_dataframe(loops_df.loc[[i], ['chrom', 'start', 'end']]) #
    tmp = BedTool.from_dataframe(tpm_df).intersect(
        loop_bed, wa=True, f=1.0).to_dataframe().loc[:,
          ['chrom', 'start', 'end', 'name']].drop_duplicates()
    # print(tmp.shape[0])
    # do GSEA
    study = tmp.name.tolist()
    if go_dicty_df.query('Gene == @study').shape[0] > 0:
        print(len(go_dicty_df.query('Gene == @study').Gene.unique().tolist()))
        enr2 = gp.enrich(gene_list=tmp.name.tolist(), # or gene_list=glist
                         gene_sets=go_dicty_dic, #_filtered,
                         background=tpm_df.name.tolist(),
                         outdir=None,#'results/pics/',
                         verbose=False)


        pop = tpm_df.name.tolist()  # list(assoc_dicty_dic.keys()) #
        methods = ["bonferroni", "sidak", "holm", "fdr"]

        g = GOEnrichmentStudy(pop, assoc_dicty_dic, #_filt
                              go,
                              propagate_counts=True,
                              alpha=0.05,
                              methods=["holm"])
        g_res = g.run_study(study, prt=None)
        # g.print_results(g_res, min_ratio=None, pval=0.01)
        #goea_results_sig = [r for r in g_res if r.p_uncorrected < 0.05 and r.depth > 1]
        goid_subset = []
        for ntd in sorted(g_res, key=lambda nt: [nt.p_uncorrected, nt.GO, nt.study_count, nt.depth, nt.study_items]):
            if (ntd.p_holm < 0.05) & (int(ntd.study_count) > 1): #uncorrected
                goid_subset.append(ntd.GO)


# %%
