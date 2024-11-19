# %% import standard python libraries
from math import pow
import bioframe as bf
import networkx as nx
import os
import pickle as pkl


import pandas as pd
# from statannot import add_stat_annotation
from scipy.stats import zscore
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# DE
# from pydeseq2.dds import DeseqDataSet
# from pydeseq2.default_inference import DefaultInference
# from pydeseq2.ds import DeseqStats

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
sys.path.append('/home/mdempg/projects/dicty/hic_loop_study/scripts/functions/modules/')

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

# _register_cmaps()

os.chdir('/home/mdempg/projects/dicty/hic_loop_study/')

# %% import data
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
resolution=500
chromsizes_path='data/genome/dicty.chrom.sizes'
clr = cooler.Cooler(cooler_path + str(resolution))

df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes['end'] = clr.chromsizes.tolist()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True


df_chromosight = pd.read_table('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.tsv')
df_chromosight.head()

# %%
df_chromosight_bed = df_chromosight.loc[:,['chrom1', 'start1', 'end2']]
df_chromosight_bed.columns = ['chrom', 'start', 'end']
# %%
pup = coolpup.pileup(clr, df_chromosight_bed, rescale=True, 
                    #  rescale_size=int(1+flank * 2// resolution),  
                     rescale_flank=1, local=False, features_format='bed', view_df=df_chromsizes, nproc=4, 
                     #flip_negative_strand=True, 
                     #expected_df=expected, 
                     min_diag=2)

# %%
plotpup.plot(pup,
             score=False, 
             cmap='coolwarm', #'viridis', 
             scale='linear', sym=True,
             vmax=1.5, #vmin=0.5,
             #vmin=0.005, vmax=0.025,
             height=5)
# plt.savefig('results/pics/paper/av_ConvPair.HZ.flank50k.pdf', dpi=100, bbox_inches='tight')
plt.show()

# %%
resolution = 500
flank = 50_000
group = "sum gene high"
df_bedpe = conv_pairs_TPM.query(f'chrom in {clr.chromnames}').loc[:,['chrom', 'start', 'end', 'anchor_group_intergenic', 'expr_group_0AB', 'group', 'strand+']]
df_bedpe.columns = ['chrom', 'start', 'end', 'anchor_group_intergenic', 'expr_group_0AB', 'group', 'strand']

df_bedpe.start = df_bedpe.start.astype(np.int64)
df_bedpe.end = df_bedpe.end.astype(np.int64)

paired_sites = df_bedpe[df_bedpe['end'] - df_bedpe['start'] > 0.5 * resolution][df_bedpe['end'] - df_bedpe['start'] < np.quantile(df_bedpe['end'] - df_bedpe['start'], 0.99)].query('anchor_group_intergenic != "none"') # and group == @group
median_size = np.median(paired_sites.end - paired_sites.start)

expected = cooltools.expected_cis(clr, view_df=df_chromsizes,
                                nproc=4, 
                                chunksize=1_000_000,
                                ignore_diags=2)

pup = coolpup.pileup(clr, paired_sites, groupby=['expr_group_0AB1', ], rescale=True, rescale_size=int(1+flank * 2// resolution),  rescale_flank=10, local=True, features_format='bed', view_df=df_chromsizes, nproc=4, flip_negative_strand=True, expected_df=expected, flank=flank, min_diag=2)

# %% Fig S2
# Supplementary Figure 2.
# a, loop size distribution for the four development stages and cell cycle phases. #TODO replace with Proportion and size distribution of consecutive and stand-alone loops
# b, IS profiles at various slide insulation squares.

# Parse files
stages = ['0AB', '2AB', '5AB', '8AB'] 

# Define a list to keep zoomed trans contact matrices
maps_dicty = [] 

# Loop over the stages and loop types
for stage in stages:
    for type in ['all']:#['consecutive', 'standalone']: #'all', 
        # Define the loop file paths based on the stage and type
        if type == "all":
            all_loops = {"%s_all" % stage: f'data/loops_quantifyChromosight/{stage}_chromosight_quantifyMarkedGood.bed' for stage in stages}
        elif type == "consecutive":
            all_loops = {"%s_%s" % (stage, type): f'data/loops_quantifyChromosight/{stage}_consecutive_1binsAdded.bed' for stage in stages}
        elif type == "standalone":
            all_loops = {"%s_%s" % (stage, type): f'data/loops_quantifyChromosight/{stage}_standAlone_1binsAdded.bed' for stage in stages}
        
        # Read the loop file and add the loop type and stage information
        mtx_comb = bf.read_table(all_loops["%s_%s" % (stage, type)], schema='bed3')
        print(mtx_comb.shape)
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
g = sns.violinplot(df_loopSize, x='stage', y='length', hue='type')#)
g.yaxis.set_label_position("right")
g.yaxis.tick_right()
g.legend([],[], frameon=False)
g.set(xlabel='', ylabel='loop length, kbp', xticklabels=['0h', '2h', '5h', '8h'], yticks=[20000, 40000], yticklabels=['20', '40']) #
# subfigs[0].suptitle('Loop length throught development')

plt.savefig('results/pics/paper/violin_loop_size_revision1.pdf', dpi=100, bbox_inches='tight')
plt.show()
# plt.clf()
# %% для 5a пар сделать сравнения как в 5c
# all genes - outside loops - LA - inside loops; CGP vs other genes
tpm_df = create_rnaseq_df(need_coord=True)
names_CGP = conv_pairs_TPM['name+'].tolist() + conv_pairs_TPM['name-'].tolist()
tmp1 = conv_pairs_TPM[['name+', 'anchor_group_intergenic']]
tmp2 = conv_pairs_TPM[['name-', 'anchor_group_intergenic']]
tmp1.columns = ['name', 'anchor_group_intergenic']
tmp2.columns = ['name', 'anchor_group_intergenic']
tmp = pd.concat([tmp1, tmp2])
tpm_df = pd.merge(tpm_df, tmp, on='name', how='left')

tpm_df['CGP'] = 'no'
tpm_df.loc[tpm_df.name.isin(names_CGP), 'CGP'] = 'yes'

tpm_df['loop_group'] = 'none'
for mode in ['inside-1bin', 'anchors_3bins']:
    df_loops = load_BedInMode("data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed", '0', mode, resolution=2000)
    if mode == 'anchors_3bins':
        tpm_df_noCGP = tpm_df.query('CGP == "no"')
        tmp = bf.overlap(tpm_df_noCGP, df_loops, how='inner', return_index=True, return_overlap=True, suffixes=('_1', '_2'))
        tpm_df.loc[(tpm_df.anchor_group_intergenic == "loop anchors") & (tpm_df.CGP == "yes"), "loop_group"] = "anchors_3bins"
    else:
        tmp = bf.overlap(tpm_df, df_loops, how='inner', return_index=True, return_overlap=True, suffixes=('_1', '_2'))
    tmp['overlap_length'] = tmp.overlap_end - tmp.overlap_start
    tmp['gene_length'] = tmp.end_1 - tmp.start_1
    tmp['gene_fraction'] = tmp.overlap_length / tmp.gene_length

    tpm_df.loc[tmp.query('gene_fraction == 1.0')['index_1'].unique().tolist(), 'loop_group'] = mode
df_loops = load_BedInMode("data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed", '0', 'whole', resolution=2000)
df_loops_compl = bf.complement(df_loops)
tmp = bf.overlap(tpm_df,  df_loops_compl, how='inner', return_index=True, return_overlap=True, suffixes=('_1', '_2'))
tmp['overlap_length'] = tmp.overlap_end - tmp.overlap_start
tmp['gene_length'] = tmp.end_1 - tmp.start_1
tmp['gene_fraction'] = tmp.overlap_length / tmp.gene_length

tpm_df.loc[tmp.query('gene_fraction == 1.0')['index_1'].unique().tolist(), 'loop_group'] = 'outside'
tpm_df.anchor_group_intergenic[tpm_df.anchor_group_intergenic.isna()] = 'none'
tpm_df['log10_0AB'] = np.log10(tpm_df['0AB'] + 0.01)
tpm_df.loc[:,['CGP', 'loop_group']].value_counts()
for loop_group in ['inside-1bin', 'outside', 'anchors_3bins']:
    tpm_df.query('CGP == "no" and loop_group == @loop_group').loc[:, ['chrom', 'start', 'end']].to_csv("data/genome/genesnotCGP.%s.bed" % loop_group, index=False, sep = '\t', header=False)
    tpm_df.query('CGP == "yes" and loop_group == @loop_group').loc[:, ['chrom', 'start', 'end']].to_csv("data/genome/genesCGP.%s.bed" % loop_group, index=False, sep = '\t', header=False)

# %%
for group in ['CGP', 'notCGP']:
    for loop_group in ['inside-1bin', 'outside', 'anchors_3bins']:
        pValue_featureOccurenceInLoop(
            file_loops="data/genome/genes%s.%s.bed" % (group, loop_group),
            time=None,
            mode='whole',
            N_shuffle=1000,
            file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
            name="enhancers_in_%s_at_%s.revision1" % (group, loop_group), 
            genome_file="data/genome/dicty.chrom.sizes",
            pic_path="results/pics/paper",
            resolution=2000)
# %% The cluster analysis provides correlations, but it is not clear what the controls were. 
# The authors should include control with randomized loops
os.chdir('/home/mdempg/projects/dicty/hic_loop_study/')

loops_path = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed"

loops_bed = BedTool(loops_path)
loops_df = pd.read_table(loops_path, header=None)
loops_df.columns = ["chrom", "start", "end"]
for group in ['1', '2', '3', '4']:
    feature = BedTool("data/genome/pseudogenes_group" + group + ".bed")
    covByPseudogenes = loops_bed.coverage(feature).to_dataframe()
    loops_df["covByGroup"+str(group)]  = covByPseudogenes.loc[:, 'thickStart'].tolist()

# %% shuffle
data = []
for i in range(100):
    genome_file="data/genome/dicty.chrom.sizes"
    loops_bed_shuffled = loops_bed.shuffle(g=genome_file, chrom=True, seed=i)
    loops_df_shuffled = loops_bed_shuffled.to_dataframe()
    from random import shuffle

    loop_labels_shuffled = ['1'] * 163 + ['2'] * 342 + ['3'] * 147 + ['4'] * 108 + ['5'] * 190 + ['6'] * 336
    shuffle(loop_labels_shuffled)

    for group in ['1', '2', '3', '4']:
        feature = BedTool("data/genome/pseudogenes_group" + group + ".bed")
        covByPseudogenes = loops_bed_shuffled.coverage(feature).to_dataframe()
        loops_df_shuffled["covByGroup"+str(group)]  = covByPseudogenes.loc[:, 'thickStart'].tolist()
    loops_df_shuffled['kmeans_cl'] = loop_labels_shuffled

    loop_df_ShuffGathered = pd.melt(loops_df_shuffled[['covByGroup1', 'covByGroup2', 'covByGroup3',
        'covByGroup4', 'kmeans_cl']], id_vars="kmeans_cl")
    loop_df_shuffledRes = loop_df_ShuffGathered.groupby(['kmeans_cl', 'variable'], as_index=False).median()

     # Append the DataFrame to the list
    data.append(loop_df_shuffledRes)

# Concatenate all DataFrames in the list into a single DataFrame
result_df = pd.concat(data, ignore_index=True)
result_df.head()

# %%
loops_df['loop_id'] = loops_df.index
loops_df_plot = loops_df[['covByGroup1', 'covByGroup2', 'covByGroup3',
                          'covByGroup4']]

loops_dropped = loops_df[['covByGroup1', 'covByGroup2', 'covByGroup3', 'covByGroup4']] 

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0).fit(loops_dropped)
kmeans.labels_
loops_df['kmeans_cl'] = kmeans.labels_

loop_df_gathered = pd.melt(loops_df[['covByGroup1', 'covByGroup2', 'covByGroup3',
       'covByGroup4', 'kmeans_cl']], id_vars="kmeans_cl")

g = sns.boxplot(y="value", data=loop_df_gathered,
            x='kmeans_cl', hue="variable"
            # y="delta_is",
            )
# plt.savefig("results/pics/paper/boxplots_covByPseudoGenes_kmeans.pdf", format="pdf", dpi=300)
plt.show()

# %%
g = sns.boxplot(y="value", data=result_df,
            x='kmeans_cl', hue="variable"
            # y="delta_is",
            )
# plt.savefig("results/pics/paper/boxplots_covByPseudoGenes_kmeans.shuffled.pdf", format="pdf", dpi=300)
plt.show()
# %%
os.chdir('/home/mdempg/projects/dicty/hic_loop_study/')

loops_path = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bed"
loops_bed = BedTool(loops_path)
loops_complement = BedTool.complement(loops_bed,g='data/genome/dicty.chrom.sizes').to_dataframe()

loops_complement.to_csv("data/loops_complement.bed", index=False, sep = '\t', header=False)
# %%
for mode in ['whole']: #['inside', 'anchors_3bins']: #
    pValue_featureOccurenceInLoop(file_loops="data/loops_complement.bed", 
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_features='data/genome/Blocks_of_conserved_gene_order.bed', #'results/0AB_loops_thresBMA.bed',
        name='synthenyLoopsComplement_mode_%s' % (mode))
# %% GO at LA
tpm_df = create_rnaseq_df(need_coord=True)

import gseapy as gp
from goatools.go_enrichment import GOEnrichmentStudy
from statsmodels.stats.multitest import fdrcorrection
from go_func import *
from go_func import plot_gos_uncorrected, GODagSmallPlot_uncorrected

single_loops_enr2GO_list = list()
mode = 'anchors'
df_loops = load_BedInMode("data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed", '0', mode, resolution=2000)
tpm_df["loop_group"] = 'notLA'
if mode == 'anchors':
    tmp = bf.overlap(tpm_df, df_loops, how='inner', return_index=True, return_overlap=True, suffixes=('_1', '_2'))
    # tpm_df.loc[tpm_df.anchor_group_intergenic == "loop anchors", "loop_group"] = "anchors_3bins"

# print(tmp.shape[0])
# do GSEA
study = tmp.name_1.tolist()
if go_dicty_df.query('Gene == @study').shape[0] > 0:
    # print(len(go_dicty_df.query('Gene == @study').Gene.unique().tolist()))
    # enr2 = gp.enrich(gene_list=study, # or gene_list=glist
    #                     gene_sets=go_dicty_dic, #_filtered,
    #                     background=tpm_df.name.tolist(),
    #                     outdir=None,#'results/pics/',
    #                     verbose=False)


    # pop = tpm_df.name.tolist()  # list(assoc_dicty_dic.keys()) #
    # methods = ["bonferroni", "sidak", "holm", "fdr"]

    g = GOEnrichmentStudy(pop, assoc_dicty_dic, #_filt
                            go,
                            propagate_counts=True,
                            alpha=0.05,
                            methods=["fdr_bh"])
    g_res = g.run_study(study, prt=None)
    # # g.print_results(g_res, min_ratio=None, pval=0.01)
    goea_results_sig = [r for r in g_res if r.p_fdr_bh < 0.001 and r.depth > 3 and r.depth < 9 and r.study_count > 20]
    # goid_subset = []
    # for ntd in sorted(g_res, key=lambda nt: [nt.p_uncorrected, nt.GO, nt.study_count, nt.depth, nt.study_items]):
    #     # if (ntd.p_fdr_bh < 0.05) & (int(ntd.study_count) > 1): #uncorrected
    #     #     goid_subset.append(ntd.GO)
    #     goea_results_sig = [r for r in g_res if (r.p_holm < 0.01 and r.depth > 3 and r.study_count > 40)] #
    GO_df = go_it(g, study, GO_items)
    g.wr_xlsx("results/genesAtLA.10-3.filt20.xlsx", goea_results_sig)
    # g.wr_tsv("results/%s_genesWithMergedEnh.10-3.filt20.tsv" % time, goea_results_sig)

# %% prepare for plotting
GO_df.GO = GO_df.GO.astype("category")
# GO_df.GO = GO_df.GO.cat.set_categories(sorter)
GO_df.p_corr = GO_df.p_corr.astype(float)
df = GO_df[GO_df['GO'].notna()]
df['log10_p'] = -np.log10(df.p_corr)
df = df.query("n_genes > 20 and p_corr < 0.001 and depth > 3 and depth < 9")
df.p = df.p.astype(float)

cmap = mpl_cm.Reds_r
norm = mpl.colors.Normalize(vmin=5.21e-10, vmax=0.189)
#(vmin = df.log10_p.min(), vmax = df.log10_p.max())

mapper = mpl_cm.ScalarMappable(norm = norm, cmap = mpl_cm.Reds_r)
plt_width=4
f, axs = plt.subplots(
    figsize=( plt_width+plt_width+2, plt_width+plt_width+1),
    ncols=2,
    gridspec_kw={'height_ratios':[4,],"wspace":0.01,'width_ratios':[1,.05]},
    constrained_layout=True)

cbl = mpl.colorbar.ColorbarBase(axs[1], cmap = cmap, norm = norm, orientation = 'vertical')

# plt.figure(figsize = (18,6))

ax = sns.barplot(data = df, x = 'n_genes', y = 'term', #hue = 'log10_p',
                palette = mapper.to_rgba(df.p.values), 
                ax = axs[0])

ax.set_yticklabels([textwrap.fill(e, 22) for e in df['term']])
plt.savefig("results/pics/paper/barplot_go_GenesAtLA.pdf", format="pdf", dpi=300)

plt.show()
plt.clf()
# %%
########### 
# stripes #
###########
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
resolution=2_000
chromsizes_path='data/genome/dicty.chrom.sizes'
clr = cooler.Cooler(cooler_path + str(resolution))

df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes['end'] = clr.chromsizes.tolist()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
# expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True


df_chromosight = pd.read_table('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.tsv') #0AB_chromosight_quantifyMarkedGood.bedpe', header=None) 
# df_chromosight.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
paired_sites = df_chromosight.query('score > 0.2').iloc[:,0:6].reset_index(drop=True) # and pvalue > 0 .
paired_sites.head()

df_chromosight_bed = paired_sites.loc[:,['chrom1', 'start1', 'end2']]
df_chromosight_bed.columns = ['chrom', 'start', 'end']

coef_flank=2
median_size = np.median(paired_sites.start2 - paired_sites.end1)
flank = int((median_size * coef_flank // resolution) * resolution)

pup = coolpup.pileup(clr, paired_sites, rescale=True, rescale_size=int((median_size+flank * 2)// resolution), rescale_flank=3, local=False, features_format='bedpe', view_df=df_chromsizes, nproc=8, flip_negative_strand=False, flank=flank, min_diag=3, store_stripes=True, mindist=resolution*3) #expected_df=expected,

# %%
# s_list = [pup.vertical_stripe[0], pup.horizontal_stripe[0]]
s_list_concat = [np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0]), axis=0)]
plot_around_loop('', 'stripes_concat.Nov5.ObservedRescaled.Nov7.2kb.pdf', nbins=np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0])).shape[1], mode='mean', window=flank, resolution=resolution, fill=False, how_far_from_edge=flank//median_size, ymin=0, ymax=0.03, vmin=0., vmax=0.05, bed_list=['concat.bed'], s_list=s_list_concat)


# %%
###############
# Stripes zoo #
###############

# %% human
resolution = 5000  # resolution of the cool file
organism = 'human'
windows = [25 * resolution]  # list of window sizes in Kb
bed_list = ['data/hic_zoo/GSE63525_GM12878_primary_HiCCUPS_looplist.bed']  # list of bed files
mode = 'median'  # mode used to calculate the loop profiles
df_chromsizes = bf.read_chromsizes("data/hic_zoo/hg38.chrom.nochr.sizes", filter_chroms=False)  # chromosome sizes

organism = 'human'
mode = 'median'
clr = cooler.Cooler('data/hic_zoo/GSE63525_GM12878_insitu_primary.5kb.mcool::/resolutions/' + str(resolution))

df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

df_chromosight = pd.read_table('data/hic_zoo/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist.txt').iloc[:,0:6]
df_chromosight.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
df_chromosight.head()
# %%
df_chromosight_bed = df_chromosight.loc[:,['chrom1', 'start1', 'end2']]
df_chromosight_bed.columns = ['chrom', 'start', 'end']

expected = pd.read_csv("data/hic_zoo/GSE63525_GM12878_insitu_primary.5kb.expected.tsv", sep='\t')

flank = int((median_size * coef_flank // resolution) * resolution)
median_size = np.median(df_chromosight_bed.end - df_chromosight_bed.start)
coef_flank=4/2
pup = coolpup.pileup(clr, df_chromosight, rescale=False, rescale_size=int(1+flank * 2// resolution),  
rescale_flank=0.5, local=False, features_format='bedpe', view_df=df_chromsizes, nproc=4, 
flip_negative_strand=False, expected_df=expected, flank=flank, min_diag=5, store_stripes=True)

s_list_concat = [np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0]), axis=0)]
plot_around_loop('', 'stripes_concat.human.notrescaled.pdf', nbins=60, mode='mean', window=10000, resolution=500, fill=False, how_far_from_edge=20, ymin=0.6, ymax=3, vmin=0.25, vmax=4.5, bed_list=['concat.bed'], s_list=s_list_concat)

# %% 
#######
# dro #
#######
organism = 'drosophila'
resolution = 5000
bed_file = 'data/hic_zoo/SpG_TADs_final.txt'
cool_file = 'data/hic_zoo/Bam_merged.1000.mcool'
clr = cooler.Cooler('data/hic_zoo/Bam_merged.1000.mcool::/resolutions/' + str(resolution))
df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
paired_sites = pd.read_table("data/hic_zoo/SpG_TADs_final.bedpe", header=None)#.iloc[0:200,:]
paired_sites.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

# expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True

# %%
median_size = np.median(paired_sites.start2 - paired_sites.end1)
coef_flank=4
flank = int((median_size * coef_flank // resolution) * resolution)
print(flank)
pup = coolpup.pileup(clr, paired_sites, rescale=True, rescale_size=int((median_size+flank * 2)// resolution), rescale_flank=20, local=False, features_format='bedpe', view_df=df_chromsizes, nproc=8, flip_negative_strand=False, flank=flank, min_diag=3, store_stripes=True, mindist=resolution*3) #expected_df=expected, 

s_list_concat = [np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0]), axis=0)]
# s_list_concat.to_csv('data/dro_concat.bed')
plot_around_loop('', 'stripes_concat.dro.ObservedRescaled.Nov7.pdf', nbins=np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0])).shape[1], mode='median', window=flank, resolution=resolution, fill=False, how_far_from_edge=flank//median_size, ymin=0, ymax=0.04, vmin=0.00, vmax=0.1, bed_list=['concat.bed'], s_list=s_list_concat)

# %% yeast
organism = 'yeast'
bed_file='data/hic_zoo/Chromosight_loops_45min.bedpe'
paired_sites = bf.read_table(bed_file, schema='bedpe')

resolution = 3_200
clr = cooler.Cooler('data/hic_zoo/GSM4585145_23C-45min.mcool::/resolutions/' + str(resolution))
df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

# expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True

# %%
median_size = np.median(paired_sites.start2 - paired_sites.end1)
coef_flank=1
flank = int((median_size * coef_flank // resolution) * resolution)
print(flank)
pup = coolpup.pileup(clr, paired_sites, rescale=True, rescale_size=int((median_size+flank * 2)// resolution),  rescale_flank=8, local=False, features_format='bedpe', view_df=df_chromsizes, nproc=8, flip_negative_strand=False, flank=flank, min_diag=3, store_stripes=True, mindist=resolution*3) #expected_df=expected, 

s_list_concat = [np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0]), axis=0)]
# s_list_concat.to_csv('data/dro_concat.bed')
plot_around_loop('', 'stripes_concat.yeast.ObservedRescaled.Nov7.pdf', nbins=np.concatenate((pup.vertical_stripe[0], pup.horizontal_stripe[0])).shape[1], mode='mean', window=flank, resolution=resolution, fill=False, how_far_from_edge=flank//median_size, ymin=0.00, ymax=0.025, vmin=0, vmax=0.09, bed_list=['concat.bed'], s_list=s_list_concat)


# %%
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
resolution=2_000
chromsizes_path='data/genome/dicty.chrom.sizes'
clr = cooler.Cooler(cooler_path + str(resolution))

df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes['end'] = clr.chromsizes.tolist()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

df_chromosight = pd.read_table('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.tsv') #0AB_chromosight_quantifyMarkedGood.bedpe', header=None) 
expected_df = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True

# df_chromosight.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
df_chromosight = df_chromosight.query('score > 0.2').iloc[:,0:6].reset_index(drop=True) # and pvalue > 0
paired_sites = pd.DataFrame({
    'chrom1': df_chromosight.chrom1,
    'start1': df_chromosight.start1+2000,
    'end1': df_chromosight.end1-2000,
    'chrom2': df_chromosight.chrom2,
    'start2': df_chromosight.start2+2000,
    'end2': df_chromosight.end2-2000}
)

paired_sites.head()

# %%
# paired_sites = bf.read_table('results/long_loops/0AB_loops_leftFlames0.8.bedpe', schema='bedpe')
# paired_sites.head()
flank = 30_000
stack = cooltools.pileup(clr, paired_sites, view_df=df_chromsizes, expected_df=expected_df, flank=flank, nproc=8)
mtx = np.nanmean(stack, axis=0)
plt.imshow(
    np.log2(mtx),
    vmax = 1,
    vmin = -1,
    cmap='coolwarm')

plt.colorbar(label = 'log2 mean obs/exp')
# ticks_pixels = np.linspace(0, flank*2//resolution,5)
# ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)
# plt.xticks(ticks_pixels, ticks_kbp)
# plt.yticks(ticks_pixels, ticks_kbp)
# plt.xlabel('relative position, kbp')
# plt.ylabel('relative position, kbp')

plt.show()

# %%
cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
resolution=2_000
chromsizes_path='data/genome/dicty.chrom.sizes'
clr = cooler.Cooler(cooler_path + str(resolution))

expected_df = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True

# coef_flank=1
# median_size = np.median(paired_sites.start2 - paired_sites.end1)
# flank = 20_000#int((median_size * coef_flank // resolution) * resolution)

# sns.set_theme(style="whitegrid")

# f, axes = plt.subplots(2, 2,
# figsize=(18, 15))
# plt.ylim([-0.5,0.5])
# c= 0 
# perp_flame = create_perp_flame(paired_sites, clr, df_chromsizes, 
#                                 flank=flank, left = True, 
#                                 pad = 2, expected_df=expected_df)
# sns.boxplot(x="variable", y="log10_value", data=perp_flame, ax=axes[c, 0], showfliers=False).set_title('left perp to right-flamed loops')
# axes[c, 0].set_xlabel('')
# axes[c, 0].set_ylim([-0.5,0.5])

# # right perp to right-flamed loops
# perp_flame = create_perp_flame(paired_sites, clr, df_chromsizes, 
#                                 flank=flank, left = False, 
#                                 pad = 2, expected_df=expected_df)
# sns.boxplot(x="variable", y="log10_value", data=perp_flame, ax=axes[c, 1], showfliers=False).set_title('right perp to right-flamed loops')
# axes[c, 1].set_xlabel('')
# axes[c, 1].set_ylim([-0.5,0.5])

# %%
from scipy.stats import mannwhitneyu
s1 = [x for x in perp_flame.query('variable == 0').value.tolist() if ~np.isnan(x)]
s2 = [x for x in perp_flame.query('variable == 6').value.tolist() if ~np.isnan(x)]
U1, p = mannwhitneyu(s1, s2, method="auto", alternative='less')
print(p)

perp_flame = create_perp_flame(paired_sites, clr, df_chromsizes, 
                                flank=8_000, left = True, 
                                pad = 2, expected_df=expected_df, both=True)
fig = sns.boxplot(x="variable", y="value", data=perp_flame, showfliers=False, log_scale=True)
fig.set_xlabel('')

# annotations = [('0', '6', 0.0001)]

# # Annotate significance
# starbars.draw_annotation(annotations)
fig.set_ylim([0.2,4])
fig.set(yticks=[0.2, 1, 4], yticklabels=['0.2', '1', '4'])
plt.savefig('stripes_perp.dicty.notrescaled.Nov18.pdf', dpi=100, bbox_inches="tight")                        

# %% 
# ############
# drosophila
##############

resolution = 5000
bed_file = 'data/hic_zoo/SpG_TADs_final.txt'
cool_file = 'data/hic_zoo/Bam_merged.1000.mcool'
clr = cooler.Cooler('data/hic_zoo/Bam_merged.1000.mcool::/resolutions/' + str(resolution))
df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
paired_sites = pd.read_table("data/hic_zoo/SpG_TADs_final.bedpe", header=None)#.iloc[0:200,:]
paired_sites.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']


expected_df = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True

coef_flank=1
median_size = np.median(paired_sites.start2 - paired_sites.end1)
flank = 20_000#int((median_size * coef_flank // resolution) * resolution)


import starbars
perp_flame = create_perp_flame(paired_sites, clr, df_chromsizes, 
                                flank=20_000, left = True, 
                                pad = 2, expected_df=expected_df, both=True)
from scipy.stats import mannwhitneyu
s1 = [x for x in perp_flame.query('variable == 0').value.tolist() if ~np.isnan(x)]
s2 = [x for x in perp_flame.query('variable == 4').value.tolist() if ~np.isnan(x)]
U1, p = mannwhitneyu(s1, s2, method="auto", alternative='less')
print(p)

fig = sns.boxplot(x="variable", y="value", data=perp_flame, showfliers=False, log_scale=True)
# annotations = [('0', '4', 0.0001)]

# # Annotate significance
# starbars.draw_annotation(annotations)
fig.set_ylim([0.2,4])
fig.set(yticks=[0.2, 1, 4], yticklabels=['0.2', '1', '4'])
plt.savefig('stripes_perp.dro.notrescaled.Nov18.pdf', dpi=100, bbox_inches="tight")

# %% yeast
organism = 'yeast'
bed_file='data/hic_zoo/Chromosight_loops_45min.bedpe'
paired_sites = bf.read_table(bed_file, schema='bedpe')

resolution = 3_200
clr = cooler.Cooler('data/hic_zoo/GSM4585145_23C-45min.mcool::/resolutions/' + str(resolution))
df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
expected_df = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=8, chunksize=1_000_000, ignore_diags=2) #, median=True

coef_flank=1
median_size = np.median(paired_sites.start2 - paired_sites.end1)
flank = 20_000#int((median_size * coef_flank // resolution) * resolution)

import starbars
perp_flame = create_perp_flame(paired_sites, clr, df_chromsizes, 
                                flank=resolution*4, left = True, 
                                pad = 2, expected_df=expected_df, both=True)
from scipy.stats import mannwhitneyu
s1 = [x for x in perp_flame.query('variable == 0').value.tolist() if ~np.isnan(x)]
s2 = [x for x in perp_flame.query('variable == 4').value.tolist() if ~np.isnan(x)]
U1, p = mannwhitneyu(s1, s2, method="auto", alternative='less')
print(p)

import matplotlib.pyplot as plt
fig = sns.boxplot(x="variable", y="value", data=perp_flame, showfliers=False, log_scale=True)
# annotations = [('0', '6', 0.0001)]

# Annotate significance
# starbars.draw_annotation(annotations)
fig.set_ylim([0.2,4])
plt.savefig('stripes_perp.yeast.notrescaled.Nov18.pdf', dpi=100, bbox_inches="tight")
# %%
