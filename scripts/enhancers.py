# Created by ir.zhegalova at Sept 7, 2022
# Script description:
"""
Count enhancers in loop anchors and create random control for shuffle test
"""
# %%
import multiprocess
import os
os.chdir('~/projects/dicty/hic_loop_study/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import bioframe
import bioframe as br
import cooler
import cooltools.api.expected as ce
import cooltools.api.snipping as clsnip
from cooltools.lib.numutils import LazyToeplitz
import cooltools.lib.plotting
from bioframe import count_overlaps
from bioframe.ops import parse_region
from pybedtools import BedTool

import sys
sys.path.append('~/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import plot_avLoop, create_stack, _create_stack, pValue_featureOccurenceInLoop, load_BedInMode, create_rnaseq_df, wrap_labels, plot_around_anchors, get_expression_sum_interesectingFeature, const_line, names_mergedIDR_dic, time_dic, create_rnaseq_df_rosengarten
# import custom_functions

# %% check versions
print(cooler.__version__)
print(cooltools.__version__)
print(bioframe.__version__)

# %%
# results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak
V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K27ac_idr_peaks.narrowPeak")
S_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K27ac_idrPR_peaks.narrowPeak")
V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idrPR_peaks.narrowPeak")
S_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K4me1_idr_peaks.narrowPeak")
#("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K27ac_idr_peaks.narrowPeak")
# results/atac_seq_wang/bwa/mergedReplicate/macs/narrowPeak
V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #
S_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Streaming_r123.mergeIDR.bed")
promoters = BedTool("data/genome/promoters.bed")



for time in ['V', 'S', 'M' ]: #
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    V_h3k4me3 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][3]))
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) #Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #

    V_chipseq_df = V_h3k27ac.intersect(V_h3k4me1, e=True, f=0.25, F=0.25, wo=True).to_dataframe(names=range(0,40)).iloc[:,[0,1,19,20,21]]
    V_chipseq_df.columns = ['start1', 'end1', 'chrom', 'start2', 'end2']
    V_chipseq_df['start'] = V_chipseq_df[['start1','start2']].min(axis=1)
    V_chipseq_df['end'] = V_chipseq_df[['end1','end2']].max(axis=1)
    V_chipseq_df = V_chipseq_df[['chrom', 'start', 'end']].sort_values(by=['chrom', 'start']).drop_duplicates()
    V_chipseq_bed = BedTool.from_dataframe(V_chipseq_df).saveas('results/%s_idrMergeIdr.chipseqUnion.bed' % (time))

    # V_chipseq_bed = V_h3k27ac.intersect(V_h3k4me1, f=0.25, F=0.25)
    V_enhancer_df = V_atacseq.intersect(V_chipseq_bed, f=0.25, F=0.25).to_dataframe().iloc[:,[0,1,2]] #wo=True, .iloc[:,[0,1,3,4,5]]    .intersect(V_h3k4me3, v=True)
    V_enhancer_df['name'] = V_enhancer_df.index
    V_atacseq.intersect(V_chipseq_bed, f=0.25, F=0.25).intersect(V_h3k4me3, v=True).saveas('results/%s_idrMergeIdr.chipseqUnionInteresectATAC.withH3k4me3.bed' % (time))
    # V_atacseq.intersect(V_chipseq_bed, f=0.25, F=0.25).saveas('results/%s_idrMergeIdr.chipseqUnionInteresectATAC.withPromoters.bed' % (time))
    # filter out promoters
    V_enhancer_noPromoter_bed = BedTool.from_dataframe(V_enhancer_df).intersect(promoters, v=True)
    # filter out genes
    V_enhancer_noGenes_bed = V_enhancer_noPromoter_bed.intersect(genes, v=True)
    # save to file
    # V_enhancer_noPromoter_bed.saveas('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time)) #noH3k4me3.
    # get enhancer length
    V_enhancer_noPromoter_df = V_enhancer_noPromoter_bed.to_dataframe()
    V_enhancer_noPromoter_df['length'] = V_enhancer_noPromoter_df['end'] - V_enhancer_noPromoter_df['start']
    print('Mean is: ', np.mean(V_enhancer_noPromoter_df.length))
    print('Median is: ', np.median(V_enhancer_noPromoter_df.length))

# bedtools intersect -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idrPR_peaks.narrowPeak -wa -f 0.25 -F 0.25 | uniq > results/S_enhancers.supportedByV_h3k4me1.bed
# bedtools intersect -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idrPR_peaks.narrowPeak -wa -f 0.25 -F 0.25 | uniq | wc -l
# 298
# bedtools intersect -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idrPR_peaks.narrowPeak -wa -f 0.25 -F 0.25 | uniq > results/S_enhancers.supportedByV_h3k4me1.bed
# bedtools intersect -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idrPR_peaks.narrowPeak -wa | uniq | wc -l
# 298
# bedtools intersect -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | uniq | wc -l
# 332
# bedtools intersect -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K4me1_idr_peaks.narrowPeak -wa -f 0.25 -F 0.25 | uniq | wc -l
# 359
# bedtools intersect -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K4me1_idr_peaks.narrowPeak -wa | uniq | wc -l
# 364
# bedtools intersect -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | uniq | wc -l
# 601
# bedtools intersect -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -v -wa | bedtools intersect -a stdin -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K4me1_idr_peaks.narrowPeak -wa -f 0.25 -F 0.25 | uniq > results/V_enhancers.supportedByS_h3k4me1.bed

# %%
"""
Part 1. Check whether enhancers are not randomly located
"""

# Percent of overlaps
time = '0'
# "results/long_loops/loops_rightFlames0.2.bedpe",
# "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe"
df_loops = bioframe.read_table(
    #'data/loops_quantifyChromosight/%sAB_chromosight_mergedAnchors.bed' % (time),
    'data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe',
    schema='bedpe', schema_is_strict=False #'bed3'
).iloc[:, 0:6]
el_loops = ['results/long_loops/loops_rightFlames0.2.leftAnchor.7bins.bed',
'results/long_loops/loops_leftFlames0.8.rightAnchor.7bins.bed',
'results/long_loops/loops_leftFlames0.8.leftAnchor.3bins.bed',
'results/long_loops/loops_rightFlames0.2.rightAnchor.3bins.bed']

enh_files = ['results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed']
#['results/VM_common_enhancers.bed',
#'results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed'] #

for loop in el_loops:
    df_loops = bioframe.read_table(
    loop,
    schema='bed3'
    )   
    for enh in enh_files:

    # mode = 'anchors_3bins'
    # file_loops = 'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed'
    # df_loops = load_BedInMode(file_loops, time, mode)

        df_enh = bioframe.read_table(
            enh,
            #'results/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters.bed3',
            schema='bed3'
        )


        # df_loopsWithEnh = bioframe.overlap(df_loops, df_enh, how='left', return_index=True, keep_order=True) #return_input=False, return_overlap=True,
        # df_loopsWithEnh = df_loopsWithEnh.dropna()
        # Percent_anchors_with_enh = df_loopsWithEnh['index'].unique().shape[0] * 100 / df_loops.shape[0]
        # easier way
        df_loopsWithEnh = count_overlaps(df_loops, df_enh)
        df_enhWithLoops = count_overlaps(df_enh, df_loops)

        Percent_enh_with_anchors = sum(df_enhWithLoops['count'] >= 1) * 100 / df_enh.shape[0]
        Percent_anchors_with_enh = sum(df_loopsWithEnh['count'] >= 1) * 100 / df_loops.shape[0]
        print(loop)
        print(enh)
        print(Percent_anchors_with_enh)
        print("Percent_enh inside anchors: %s" % (Percent_enh_with_anchors))
        print(np.mean(df_loopsWithEnh.loc[df_loopsWithEnh['count'] >= 1, 'count']))
        print("\n")

# %%
stage = 'V'
mode = 'whole'
file_features = "results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed" % (stage)
el_loops = ['loops_rightFlames%s.2.leftAnchor.3bins.bed',
'loops_leftFlames%s.8.rightAnchor.3bins.bed',
'loops_leftFlames%s.8.leftAnchor.3bins.bed',
'loops_rightFlames%s.2.rightAnchor.3bins.bed']
# el_loops = ['loops_rightFlames%s.15.rightAnchor.3bins.bed',
# 'loops_rightFlames%s.15.leftAnchor.7bins.bed']
for loop in el_loops: #,  'anchors_3bins', 'inside' []: #
    print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/'+loop,
                                        time=time_dic[stage][0], mode=mode, N_shuffle=1000,
                                        file_features=file_features,
                                        name=loop +'_enhancersUTI_allLoops_mode_' + mode))
# %% Shuffle test
# file_features = "results/V_enhancers_idrMergeIdr.bed"
stage = 'V'
file_features = "results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed" % (stage) #"results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.noH3k4me3.bed" % (stage) # #
for mode in ['start_3bins', 'end_3bins']: #, ,    'anchors_3bins', 'inside' ,'start', 'end'[]: #
    # print(pValue_featureOccurenceInLoop(file_loops='data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed',
    #                                     time=time_dic[stage][0], mode=mode, N_shuffle=1000,
    #                                     file_features=file_features,
    #                                     name=stage +'_enhancersUTI.noH3k4me3_allLoops_mode_' + mode))
    print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_loops_leftFlames0.8.bed',
                                        time=time_dic[stage][0],
                                        mode=mode, N_shuffle=1000,
                                        file_features=file_features, name=stage + '_enhancersUTI_elLoopsLeft_mode_' + mode))
    # print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_regular_loops.bed',
    #                                     time='0',
    #                                     mode=mode, N_shuffle=1000,
    #                                     file_features=file_features, name='V_enhancersUTI.noH3k4me3_elLoopsRegular_mode_' + mode))
    print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_loops_rightFlames0.2.bed',
                                        time=time_dic[stage][0],
                                        mode=mode, N_shuffle=1000,
                                        file_features=file_features, name=stage+'_enhancersUTI_elLoopsRight_mode_' + mode))
    # print(pValue_featureOccurenceInLoop(file_loops='results/%sAB_loops_thresBMA.bed',
    #                                     time='0', mode=mode, N_shuffle=1000,
    #                                     file_features=file_features, name='V_enhancersUTI.noH3k4me3_loopsBMA_mode_' + mode))

# %%
# results/long_loops/0AB_regular_loops.MegredAnchors.1bins_added.bed6
mode='anchors_3bins'
stage = 'V'
file_features = "results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed" % (stage)
print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_regular_loops.bed',
                                    time='0',
                                    mode=mode, N_shuffle=1000,
                                    file_features=file_features, name='V_enhancersUTI_LoopsRegular_mode_' + mode))

# %% DE genes vs enhancers
stage = 'V'
# file_features = "results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed" % (stage)
file_features = ['results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed']#, 'results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed']
# file_feature = "results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed" % (stage)
for feature in file_features:
    mode = 'whole'
    print(pValue_featureOccurenceInLoop(file_loops='data/genome/Blocks_of_conserved_gene_order%s.bed3', #'results/DE/genes_8h_%sh.logFC2down.bed',
                                        time='0', mode=mode, N_shuffle=1000,
                                        file_features=feature,
                                        name=stage + '_enhancersUTI_synthenyBlocks_mode_' + mode))
# %%
V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
M_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

# VM_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, wa=True,f=0.5).saveas('results/VM_common_enhancers.bed')
V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, wa=True, v=True,f=0.9).saveas('results/V_specific_enhancers.bed')
M_specific_enhancers_bed = M_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, wa=True, v=True,f=0.1).saveas('results/M_specific_enhancers.bed')
# %%
genes_withEnh_vec = tpm_bed.intersect(VM_common_enhancers_bed, F=0.8).to_dataframe().name.tolist()
genes_withVspecEnh_vec = tpm_bed.intersect(V_specific_enhancers_bed, F=0.8).to_dataframe().name.tolist()
genes_withMspecEnh_vec = tpm_bed.intersect(M_specific_enhancers_bed, F=0.8).to_dataframe().name.tolist()

# %%
genes = BedTool('data/genome/genes_noRTE_no_pseudogenes.bed')
genes_df = genes.to_dataframe()
genes_df['V_spec'] = genes.coverage(V_specific_enhancers_bed).to_dataframe().thickStart.astype("float").tolist()
genes_df['M_spec'] = genes.coverage(M_specific_enhancers_bed).to_dataframe().thickStart.astype("float").tolist()
genes_df['VMcommon'] = genes.coverage(VM_common_enhancers_bed).to_dataframe().thickStart.tolist()
genes_df_fil = genes_df.query('VMcommon > 0 or M_spec > 0 or V_spec > 0')
for name in ['V_spec', 'M_spec', 'VMcommon']:
    sns.histplot(data=genes_df_fil, x=name, kde=True,  stat="percent")
    plt.savefig("results/pics/paper/%s_percInGenes.pdf" % (name), dpi=100,
                bbox_inches='tight')
    plt.show()
    plt.clf()
# %%
genes = BedTool('data/genome/genes.bed3')
V_specific_enhancers = M_specific_enhancers_bed.to_dataframe()
V_specific_enhancers['gene_cov'] = V_specific_enhancers_bed.coverage(genes).to_dataframe().thickStart.astype("float").tolist()
# genes_df['M_spec'] = genes.coverage(M_specific_enhancers_bed).to_dataframe().thickStart.astype("float").tolist()
# genes_df['VMcommon'] = genes.coverage(VM_common_enhancers_bed).to_dataframe().thickStart.tolist()
# genes_df_fil = genes_df.query('VMcommon > 0 or M_spec > 0 or V_spec > 0')
# for name in ['V_spec', 'M_spec', 'VMcommon']:
sns.histplot(data=V_specific_enhancers, x='gene_cov', kde=True,  stat="percent")
# plt.savefig("results/pics/paper/%s_percInGenes.pdf" % (name), dpi=100,
#             bbox_inches='tight')
plt.show()
plt.clf()
# %% scatter for enh expression
file_features = ['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed']
DE_genes = pd.read_table('results/DE/genes_8h_0h.bed', header=None)
DE_genes.columns = ['gene', 'chrom', 'start', 'end', '_', 'strand', 'log2FC']
DE_genes = DE_genes.query('log2FC > 2.5 or log2FC < 2.5') #1.32

tpm_df = create_rnaseq_df(need_coord=True)
tpm_df_DE = tpm_df.loc[tpm_df.name.isin(DE_genes.gene),:]
tpm_DE_bed = BedTool.from_dataframe(tpm_df_DE.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))
tpm_df['group'] = 'w/o enhancer'
for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01)
for file_feature in file_features:
    V_enhancer_noPromoter_bed = BedTool(file_feature)
    genes_withEnh_vec = tpm_DE_bed.intersect(V_enhancer_noPromoter_bed).to_dataframe().name.tolist()
    tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = file_feature.split('/')[1]
tpm_df['delta_8_0'] = tpm_df['log2_8_TPM'] - tpm_df['log2_0_TPM']
tpm_df_with = tpm_df.loc[tpm_df.group != 'w/o enhancer',]
g = sns.scatterplot(data=tpm_df_with, y='delta_8_0', x='log2_8_TPM', hue='group', alpha=0.5, size=10)
plt.legend([],[], frameon=False)
plt.show()
# %%
f, ax = plt.subplots(figsize=(10, 10))
colors = {'V_specific_enhancers.bed':'red', 'M_specific_enhancers.bed':'green'}
tpm_df_with = tpm_df.loc[tpm_df.group != 'w/o enhancer',]
tpm_df_with = tpm_df_with.loc[tpm_df.group != 'VM_common_enhancers.bed',]
tpm_df_with = tpm_df_with.loc[tpm_df.group != 'M_specific_enhancers.bed',]

ax.scatter(data=tpm_df_with, y='log2_8_TPM', x='log2_0_TPM', 
alpha=0.2, c=tpm_df_with['group'].map(colors))
# g = sns.lmplot(x = "log2_0_TPM", y = "log2_8_TPM", col = "group",
#            hue = "group", data = tpm_df_with, 
#            scatter_kws={"s": 20})
# axes.Axes.axline((0, 0), (1, 1), linewidth=4, color='r')
# ax.plot([-5, -5], [10, 10], ls="--", c=".3")
add_identity(ax, color='r', ls="--")
plt.legend([],[], frameon=False)
plt.show()

# %% scatter for genes at anchors with/without enhancer
files_anchors= ['loops_rightFlames0.2.leftAnchor.3bins.bed',
'loops_leftFlames0.8.rightAnchor.3bins.bed',
'loops_leftFlames0.8.leftAnchor.3bins.bed',
'loops_rightFlames0.2.rightAnchor.3bins.bed',
'0AB_regular_loops.MegredAnchors.bed']
#'data/loops_quantifyChromosight/0AB_mergedAnchors.q=02.bins_added.bed'
file_features = ['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed']
tpm_df = create_rnaseq_df(need_coord=True)
#tpm_df_DE = tpm_df.loc[tpm_df.name.isin(DE_genes.gene),:]
tpm_bed = BedTool.from_dataframe(tpm_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))

tpm_TSS_df = create_rnaseq_df(need_TSS=True)
tpm_TSS_bed = BedTool.from_dataframe(tpm_TSS_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'TSS_start']))

tpm_df['group'] = 'w/o enhancer'
for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01)
for file_feature in file_features:
    V_enhancer_noPromoter_bed = BedTool(file_feature)
    genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed).to_dataframe().name.tolist()
    tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = file_feature.split('/')[1].split('.')[0]
tpm_df['delta_8_0'] = tpm_df['log2_8_TPM'] - tpm_df['log2_0_TPM']
for file_anchors in files_anchors:
    tpm_df['anchor_group'] = 'outside anchors'
    loopAnchor_bed = BedTool("results/long_loops/" +  file_anchors)
    genes_insideanchors_vec = tpm_TSS_bed.intersect(loopAnchor_bed).to_dataframe().name.tolist()
    tpm_df.loc[tpm_df['name'].isin(genes_insideanchors_vec), 'anchor_group'] = 'inside anchors'

    # f, ax = plt.subplots(figsize=(10, 10))
    colors = {'outside anchors':'red', 'inside anchors':'green'}
    shapes = {'V_specific_enhancers': 'o', 'w/o enhancer': 'x', 'VM_common_enhancers': '+'}
    tpm_df_with = tpm_df#.loc[tpm_df.group != 'w/o enhancer',]
    # tpm_df_with = tpm_df_with.loc[tpm_df.group != 'VM_common_enhancers.bed',]
    # tpm_df_with = tpm_df_with.loc[tpm_df.group != 'M_specific_enhancers',]

    # ax.scatter(data=tpm_df_with, y='log2_8_TPM', x='log2_0_TPM', 
    # alpha=0.05, c=tpm_df_with['anchor_group'].map(colors),
    # marker=tpm_df_with['group'].map(shapes).tolist())
    g = sns.FacetGrid(tpm_df_with, col="anchor_group", hue='group', height=15)
    g.map(sns.scatterplot, 'log2_0_TPM','log2_8_TPM', alpha=0.2)
    g.add_legend()

    g.map(const_line)
    #add_identity(g, color='r', ls="--")
    plt.savefig("results/pics/paper/scatter_TPM_at" + file_anchors.split('.bed')[0] + ".pdf", 
    bbox_inches='tight') #todo

    # sns.scatterplot(data=tpm_df_with, x='log2_0_TPM', alpha=0.2, y='log2_8_TPM', style='group', hue='anchor_group', ax=ax)
    # g = sns.lmplot(x = "log2_0_TPM", y = "log2_8_TPM", col = "group",
    #            hue = "group", data = tpm_df_with, 
    #            scatter_kws={"s": 20})
    # axes.Axes.axline((0, 0), (1, 1), linewidth=4, color='r')
    # ax.plot([-5, -5], [10, 10], ls="--", c=".3")
    
    # plt.legend([],[], frameon=False)
    plt.show()
    plt.close()

# %% scatter for genes at anchors with/without enhancer
files_anchors= ['loops_rightFlames0.2.leftAnchor.3bins.bed',
'loops_leftFlames0.8.rightAnchor.3bins.bed',
'loops_leftFlames0.8.leftAnchor.3bins.bed',
'loops_rightFlames0.2.rightAnchor.3bins.bed',
'0AB_regular_loops.MegredAnchors.bed']
#'data/loops_quantifyChromosight/0AB_mergedAnchors.q=02.bins_added.bed'

tpm_df = create_rnaseq_df(need_coord=True) #create_rnaseq_df_rosengarten() #
#tpm_df_DE = tpm_df.loc[tpm_df.name.isin(DE_genes.gene),:]
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end','name']].sort_values(by=['chrom', 'start']))

for time in ['V']: #, 'S', 'M'

    V_enhancer_noPromoter_bed = BedTool('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time))
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    M_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % ('M', names_mergedIDR_dic['M'][0]))
    V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    M_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % ('M', names_mergedIDR_dic['M'][1]))
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) 
    M_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_dic['M'][1], names_mergedIDR_dic['M'][2])) #Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #
    V_h3k4me3 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][3]))
    M_h3k4me3 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % ('M', names_mergedIDR_dic['M'][3]))

    genes_with_atacseq_bed = tpm_bed.intersect(V_atacseq, f=0.2)
    genes_with_atacseq_vec = tpm_bed.intersect(V_atacseq, f=0.2).to_dataframe().name.tolist()
    genes_with_h3k27ac_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k27ac, f=0.2).intersect(V_h3k4me1, f=0.2, v=True).intersect(V_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist()

    genes_with_h3k4me3_vec = tpm_bed.intersect(V_h3k4me3, f=0.2).to_dataframe().name.tolist() #intersect(V_h3k27ac, f=0.2, v=True).intersect(V_h3k4me1, f=0.2, v=True).intersect(genes_with_atacseq_bed, f=0.2)
    genes_with_VspecH3k4me3_vec = tpm_bed.intersect(V_h3k4me3, f=0.2).intersect(M_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist() #intersect(V_h3k27ac, f=0.2, v=True).intersect(V_h3k4me1, f=0.2, v=True).intersect(genes_with_atacseq_bed, f=0.2)
    genes_with_MspecH3k4me3_vec = tpm_bed.intersect(M_h3k4me3, f=0.2).intersect(V_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist()

file_features = ['results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.noH3k4me3.bed'] #['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed']


tpm_TSS_df = create_rnaseq_df(need_TSS=True)
tpm_TSS_bed = BedTool.from_dataframe(tpm_TSS_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'TSS_start']))

tpm_df['group'] = 'w/o enhancer'
for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01) # 'hr0'+ str(time_int)
for file_feature in file_features:
    V_enhancer_noPromoter_bed = BedTool(file_feature)
    genes_NearbyEnh_vec = tpm_TSS_bed.closest(V_enhancer_noPromoter_bed, d=True).to_dataframe().query('thickEnd >= 0 and thickEnd < 30000').name.tolist()
    # genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed).to_dataframe().name.tolist()
    tpm_df.loc[tpm_df['name'].isin(genes_NearbyEnh_vec), 'group'] = file_feature.split('/')[1].split('.')[0]
tpm_df['delta_8_0'] = tpm_df['log2_8_TPM'] - tpm_df['log2_0_TPM']
# for file_anchors in files_anchors:
#     tpm_df['anchor_group'] = 'outside anchors'
#     loopAnchor_bed = BedTool("results/long_loops/" +  file_anchors)
#     genes_insideanchors_vec = tpm_TSS_bed.intersect(loopAnchor_bed).to_dataframe().name.tolist()
#     tpm_df.loc[tpm_df['name'].isin(genes_insideanchors_vec), 'anchor_group'] = 'inside anchors'

tpm_df['h3k4me3_group'] = 'no'
tpm_df.loc[tpm_df['name'].isin(genes_with_h3k4me3_vec), 'h3k4me3_group'] = 'common h3k4me3'
tpm_df.loc[tpm_df['name'].isin(genes_with_VspecH3k4me3_vec), 'h3k4me3_group'] = 'V-spec h3k4me3'
tpm_df.loc[tpm_df['name'].isin(genes_with_MspecH3k4me3_vec), 'h3k4me3_group'] = 'M-spec h3k4me3'


tpm_df_with = tpm_df.loc[tpm_df.h3k4me3_group != 'no',]
# tpm_df_with = tpm_df_with.loc[tpm_df.group != 'VM_common_enhancers.bed',]
# tpm_df_with = tpm_df_with.loc[tpm_df.group != 'M_specific_enhancers',]
# %%
from statannot import add_stat_annotation
sns.set(style='white', context='poster', rc={'figure.figsize':(10,10)})
# g = sns.FacetGrid(tpm_df_with, #col="group", 
# hue='h3k4me3_group', height=15)
# g.map(sns.scatterplot, 'log2_0_TPM','log2_8_TPM', alpha=0.5)
g = sns.violinplot(data=tpm_df_with, y='delta_8_0', x='h3k4me3_group')
add_stat_annotation(g, data=tpm_df_with, x='h3k4me3_group', y='delta_8_0', #order=order,
                    box_pairs=[('common h3k4me3', 'M-spec h3k4me3'),
                    ('common h3k4me3', 'V-spec h3k4me3'),
                    ('M-spec h3k4me3', 'V-spec h3k4me3')],test='Mann-Whitney', text_format='star', loc='inside', verbose=2)


# g.add_legend()

# g.map(const_line)
#add_identity(g, color='r', ls="--")
wrap_labels(g, 10)
plt.savefig("results/pics/paper/violinplot_TPM_withH3k4me3.pdf", 
bbox_inches='tight') 

plt.show()
plt.close()


# %%
DE_genes = pd.read_table('results/DE/genes_8h_0h.bed', header=None)
DE_genes.columns = ['gene', 'chrom', 'start', 'end', '_', 'strand', 'log2FC']
DE_genes_filtered = DE_genes.loc[DE_genes.log2FC > 3,].sort_values(by=['chrom', 'start'])
DE_genes_filtered.loc[:,['chrom', 'start', 'end']].to_csv('results/DE/genes_8h_0h.logFC2up.bed', sep="\t", index=False,header=False)

DE_genes_DownFiltered = DE_genes.loc[DE_genes.log2FC < -3,].sort_values(by=['chrom', 'start']).query('chrom != "chrBF"') #np.quantile(DE_genes.log2FC, 0.25)
DE_genes_DownFiltered.loc[:,['chrom', 'start', 'end']].to_csv('results/DE/genes_8h_0h.logFC2down.bed', sep="\t", index=False,header=False)

# %%
"""
Part 3. Check whether not-enhancer peaks are not randomly located
"""
# %%
# select peaks outside promoters and enhancers
for time in ['M']: #'V', 'S'
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) #Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #

    V_enhancer_noPromoter = BedTool(
        'results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time))
        
    V_h3k27ac.intersect(V_enhancer_noPromoter, v=True).saveas('results/%s_h3k27ac.notEnhancers.bed' % (time))

    V_h3k4me1.intersect(V_enhancer_noPromoter, v=True).saveas('results/%s_h3k4me1.notEnhancers.bed' % (time))
    V_atacseq.intersect(V_enhancer_noPromoter, v=True).saveas('results/%s_atacseq.notEnhancers.bed' % (time))
    V_h3k27ac.intersect(V_enhancer_noPromoter, v=True).intersect(promoters, v=True).saveas('results/%s_h3k27ac.neitherPromotersNorEnhancers.bed' % (time))
    V_h3k4me1.intersect(V_enhancer_noPromoter, v=True).intersect(promoters, v=True).saveas('results/%s_h3k4me1.neitherPromotersNorEnhancers.bed' % (time))
    V_atacseq.intersect(V_enhancer_noPromoter, v=True).intersect(promoters, v=True).saveas('results/%s_atacseq.neitherPromotersNorEnhancers.bed' % (time))
    # plot peaks occurence
    for feature in ['h3k27ac', 'h3k4me1', 'atacseq']:
        file_features = 'results/%s_%s.neitherPromotersNorEnhancers.bed' % (time, feature) #
        for mode in ['anchors', 'start', 'end']: # [ 'anchors_3bins', 'inside' ]: #
            print(pValue_featureOccurenceInLoop(file_loops=
            'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed',
                                                time='0', mode=mode, N_shuffle=1000,
                                                file_features=file_features,
                                                name='%s_%s_%s_allLoops_mode_' % (time, feature, 'noPnoE') + mode))
            print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_loops_leftFlames0.8.bed',
                                                time='0',
                                                mode=mode, N_shuffle=1000,
                                                file_features=file_features, name='%s_%s_%s_elLoopsLeft_mode_' % (time, feature, 'noPnoE') + mode))
            # print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_regular_loops.bed',
            #                                     time='0',
            #                                     mode=mode, N_shuffle=100,
            #                                     file_features=file_features, name='%s_%s_%s_elLoopsRegular_mode_' % (time, feature, 'noPnoE') + mode))
            print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_loops_rightFlames0.2.bed',
                                                time='0',
                                                mode=mode, N_shuffle=1000,
                                                file_features=file_features, name='%s_%s_%s_elLoopsRight_mode_' % (time,feature, 'noPnoE') + mode))
            # print(pValue_featureOccurenceInLoop(file_loops='results/%sAB_loops_thresBMA.bed',
            #                                     time='0', mode=mode, N_shuffle=100,
            #                                     file_features=file_features, name='V_%s_%s_loopsBMA_mode_' % (feature, 'noPnoE') + mode))

# %%
features = ['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed']
sns.set(style='white', context='poster', rc={'figure.figsize':(10,10)})
time='V'
for feature in features: #'V', 'S'

    for mode in ['anchors_3bins']: # [ 'anchors_3bins', 'inside', , 'start', 'end' ]: #
        print(pValue_featureOccurenceInLoop(file_loops=
        'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed' ,
                                            time='0', mode=mode, N_shuffle=100,
                                            file_features=feature,
                                            name='%s_%s_%s_allLoops_mode_' % (time, feature, 'noPnoE') + mode))


# %% select promoters peaks
for time in ['M']: #, 'S'
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) #Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #
    V_enhancer_noPromoter = BedTool(
        'results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time))
        
    V_h3k27ac.intersect(promoters, v=False).saveas('results/%s_h3k27ac.Promoters.bed' % (time))
    V_h3k4me1.intersect(promoters, v=False).saveas('results/%s_h3k4me1.Promoters.bed' % (time))
    V_atacseq.intersect(promoters, v=False).saveas('results/%s_atacseq.Promoters.bed' % (time))
    # plot peaks occurence
    for feature in ['h3k27ac', 'h3k4me1', 'atacseq']:
        file_features = 'results/%s_%s.Promoters.bed' % (time, feature) #
        for mode in ['anchors', 'start', 'end']: # [, 'anchors_3bins', 'inside'  ]: #
            print(pValue_featureOccurenceInLoop(file_loops=
            'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed',
                                                time='0', mode=mode, N_shuffle=1000,
                                                file_features=file_features,
                                                name='%s_%s_%s_allLoops_mode_' % (time, feature, 'Promoters') + mode))
            print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_loops_leftFlames0.8.bed',
                                                time='0',
                                                mode=mode, N_shuffle=1000,
                                                file_features=file_features, name='%s_%s_%s_elLoopsLeft_mode_' % (time, feature, 'Promoters') + mode))
            # print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_regular_loops.bed',
            #                                     time='0',
            #                                     mode=mode, N_shuffle=100,
            #                                     file_features=file_features, name='%s_%s_%s_elLoopsRegular_mode_' % (time, feature, 'Promoters') + mode))
            print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/%sAB_loops_rightFlames0.2.bed',
                                                time='0',
                                                mode=mode, N_shuffle=1000,
                                                file_features=file_features, name='%s_%s_%s_elLoopsRight_mode_' % (time,feature, 'Promoters') + mode))
            # print(pValue_featureOccurenceInLoop(file_loops='results/%sAB_loops_thresBMA.bed',
            #                                     time='0', mode=mode, N_shuffle=100,
            #                                     file_features=file_features, name='%s_%s_%s_loopsBMA_mode_' % (time, feature, 'Promoters') + mode))



# %% create zscore
import bioframe
from scipy.stats import zscore
df_chromsizes = bioframe.read_chromsizes('data/genome/dicty.chrom.sizes')
for state in ['M']: #'V', 'S', 
    bw_list = []
    for i in range(1,7):
        bw_list.append(bioframe.read_bigwig("bw/%s_H3K27ac_merged.mLb.clN.bgNorm.bw" % (state), chrom='chr' + str(i)))
    H3K27ac_df = pd.concat(bw_list)
    H3K27ac_df['z_value'] = zscore(H3K27ac_df.value)
    H3K27ac_df = H3K27ac_df.drop(['value'], axis=1)
    #  how to save to bw

    bioframe.to_bigwig(df=H3K27ac_df, chromsizes=df_chromsizes,
                       outpath='bw/%s_H3K27ac_merged.mLb.clN.bgNorm.zscore.bw' % (state),
                       path_to_binary='~/micromamba/envs/omics_env/bin/bedGraphToBigWig')
# %% zscore atacseq
import bioframe
from scipy.stats import zscore
df_chromsizes = bioframe.read_chromsizes('data/genome/dicty.chrom.sizes')
for state in ['Vegetative']: #'V', 'S', 
    bw_list = []
    for i in range(1,7):
        bw_list.append(bioframe.read_bigwig('bw/%s_merged.BPM.bw' % (state), chrom='chr' + str(i)))
    H3K27ac_df = pd.concat(bw_list)
    H3K27ac_df['z_value'] = zscore(H3K27ac_df.value)
    H3K27ac_df = H3K27ac_df.drop(['value'], axis=1)
    #  how to save to bw

    bioframe.to_bigwig(df=H3K27ac_df, chromsizes=df_chromsizes,
                       outpath='bw/%s_merged.BPM.zscore.bw' % (state),
                       path_to_binary='~/micromamba/envs/omics_env/bin/bedGraphToBigWig')

# %% cov by peaks
import bioframe
from scipy.stats import zscore
df_chromsizes = bioframe.read_chromsizes('data/genome/dicty.chrom.sizes')
sys.path.append('~/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import plot_around_loop
for state in ['M']: #, 'M' , 'S'
    # load windows
    windows_w500 = bioframe.read_table('data/genome/dicty_w500.bed', 'bed3')
    enhancers_df = bioframe.read_table('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (state), 'bed3')
    df1_coverage = bioframe.coverage(windows_w500, enhancers_df)
    df1_coverage['coverage'] = zscore(df1_coverage.coverage/500)
    # df1_coverage.query('coverage > 0')['coverage'] = 1
    # export bioframe directly to bigwig
    bioframe.to_bigwig(df=df1_coverage, chromsizes=df_chromsizes,
                           outpath='bw/%s_enhancers_cov.w500.zscore.bw' % (state),
                           path_to_binary='~/micromamba/envs/omics_env/bin/bedGraphToBigWig')
# %% plot enhancer coverage around loops
mode = 'median'
from custom_functions import plot_around_loop
for state in ['F']:  # , 'V', 'S'
    plot_around_loop("bw/IS/0AB.20k.IS.cooltools.bw", #'bw/wang_rnaseq_bw/Fruiting_R2.TPM_log2.bw',#'bw/0_8_deltaTPM_log2.bw',
                     "%s_IS_around_enhancers_cov.%s.pdf" % (state, mode),
                     fill=True, mode=mode,
                     window=20000, nbins=21,
                     how_far_from_edge=10,
                     ymin=-0.5, ymax=0.5,
                     vmin=-1, vmax=1,
                     bed_list=['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed'])
# %%
for state in ['M']:  # , 'V', 'S'
    plot_around_loop('bw/%s_enhancers_cov.w500.zscore.bw' % (state),
                     "%s_enhancers_cov_AroundLoops_pybbi.w500.zscore.mean.test1.pdf" % (state),
                     fill=True, mode='mean',
                     window=20000, nbins=30,
                     how_far_from_edge=10,
                     ymin=-0.3, ymax=0.3,
                     vmin=-1, vmax=1, bed_list=["results/long_loops/loops_leftFlames0.8.bedpe",
                               "results/long_loops/0AB_regular_loops.bedpe",
                               "results/long_loops/loops_rightFlames0.2.bedpe",
                               "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe"])
# %% shuffle test - earlier versions
# dicty_chrom = pd.read_table("data/genome/dicty.chrom.sizes", header=None, index_col=0)
# dicty_chrom.columns = ['length']
# dicty_chrom_ser = dicty_chrom.squeeze()
# dicty_binned = bioframe.binnify(dicty_chrom_ser, 2000)
#
# # create shuffled loop anchors
# import random
#
# random.seed(42)
#
# shuffled_inter_vec = []
# N_shuffle = 1000
# for i in range(N_shuffle):
#     shuffled_indexes = sorted(random.sample(range(dicty_binned.shape[0]), df_loops.shape[0]))
#     shuffled_df = dicty_binned.iloc[shuffled_indexes,]
#     df_loopsWithEnh_shuffled = count_overlaps(shuffled_df, df_enh)
#     Percent_anchors_with_enh_shuffled = \
#         sum(df_loopsWithEnh_shuffled['count'] >= 1) * 100 / shuffled_df.shape[0]
#     shuffled_inter_vec.append(Percent_anchors_with_enh_shuffled)
#
# # plot shuffle test
# inter_shuffle_df = pd.DataFrame({"Shuffle": shuffled_inter_vec})
#
# # inter_lasLoops = las_loops.intersect(ncRNA_andFeature)
# # print(inter_lasLoops.count())
# fig = sns.histplot(data=inter_shuffle_df, x="Shuffle", kde=True, stat="percent", binwidth=1)
# fig.axvline(Percent_anchors_with_enh, color="red", lw=3)
# p_value = np.round(np.min([len(inter_shuffle_df[inter_shuffle_df['Shuffle'] > Percent_anchors_with_enh]) / N_shuffle,
#                            len(inter_shuffle_df[inter_shuffle_df['Shuffle'] < Percent_anchors_with_enh]) / N_shuffle]),
#                    3)
# if p_value == 0:
#     plt.annotate("p-value < " + str(1 / N_shuffle),
#                  xy=(0.95, 0.95), xycoords='axes fraction',
#                  bbox=dict(facecolor='pink',
#                            alpha=0.5),
#                  horizontalalignment='right',
#                  fontsize=12)
# else:
#     plt.annotate("p-value = " + str(p_value),
#                  xy=(0.95, 0.95), xycoords='axes fraction',
#                  bbox=dict(facecolor='pink',
#                            alpha=0.5),
#                  horizontalalignment='right',
#                  fontsize=12)
#
# plt.savefig("results/pics/paper/enh_abundance_shuffledTest.pdf",
#             bbox_inches='tight')
# plt.show()
# plt.clf()

# %% load rna-seq - not useful
# path_bw = "bw/0_merged.TPM_noSmooth.bw"
# from bioframe import read_bigwig
# rna_bw = []
# for chr in dicty_chrom.index.tolist():
#     tmp = read_bigwig(path_bw, chr, start=None, end=None, engine='auto')
#     rna_bw.append(tmp)
# rna_bw_df = pd.DataFrame(np.concatenate(rna_bw))
# rna_bw_df.columns = tmp.columns.tolist()
# rna_bw_df['start'] = pd.to_numeric(rna_bw_df['start'])
# rna_bw_df['end'] = pd.to_numeric(rna_bw_df['end'])
# rna_bw_df['value'] = pd.to_numeric(rna_bw_df['value'])

# tmp = bioframe.coverage(dicty_binned, rna_bw_df)
# tmp.head()

# %% load binned rna-seq
# multiBigwigSummary bins -b bw/0_merged.TPM_noSmooth.bw -bs 2000 -p 2 -o bw/0_merged.TPM_noSmooth_binned.npz
rnaseq_binned_vec = np.load('bw/0_merged.TPM_noSmooth_binned.2kb.npz', allow_pickle=True)
# create list of binned values
rnaseq_binned_vec = np.concatenate(rnaseq_binned_vec['matrix.npy']).tolist()

# %% sliding window

seq = rnaseq_binned_vec
half_window_size = 1
rnaseq_binned_vec_slidWin = []
for i in range(len(seq)):
    # print(i)
    if i < half_window_size:
        rnaseq_binned_vec_slidWin.append(None)
        continue
    if i == len(seq) - half_window_size:
        rnaseq_binned_vec_slidWin.append(None)
        continue
    rnaseq_binned_vec_slidWin.append(np.mean(seq[i - half_window_size: i + half_window_size + 1]))

rnaseq_binned = dicty_binned
rnaseq_binned['value'] = rnaseq_binned_vec_slidWin

"""
Part 4. Select genes with enhancers
"""
# %%
from statannot import add_stat_annotation

time = 'V'
# tpm_df_rosengarten = create_rnaseq_df_rosengarten()
# tpm_df = tpm_df_rosengarten.reset_index()
# tpm_df = create_rnaseq_df(need_coord=True)
tpm_df = df_TPM_genesIntrons
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end', 'name']].sort_values(by=['chrom', 'start']))


for time in ['V']: #, 'S', 'M'

    V_enhancer_noPromoter_bed = BedTool('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.noH3k4me3.bed' % (time))#('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time))
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) #Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #
    V_h3k4me3 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][3]))

    genes_with_atacseq_bed = tpm_bed.intersect(V_atacseq, f=0.2)
    genes_with_atacseq_vec = tpm_bed.intersect(V_atacseq, f=0.2).to_dataframe().name.tolist()
    genes_with_h3k27ac_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k27ac, f=0.2).intersect(V_h3k4me1, f=0.2, v=True).intersect(V_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist()

    genes_with_h3k4me1_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k4me1, f=0.2).intersect(V_h3k27ac, f=0.2, v=True).intersect(V_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist()

    genes_with_h3k4me3_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k4me3, f=0.2).intersect(V_h3k4me1, f=0.2, v=True).intersect(V_h3k27ac, f=0.2, v=True).to_dataframe().name.tolist()

    genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed).to_dataframe().name.tolist()
    genes_withEnh_upperHalf_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed, f=0.5).to_dataframe().name.tolist()
    tpm_df['group'] = 'w/o enhancer'

    tpm_df.loc[tpm_df['name'].isin(genes_with_atacseq_vec), 'group'] = 'with ATAC-seq'
    tpm_df.loc[tpm_df['name'].isin(genes_with_h3k4me3_vec), 'group'] = 'with H3K4me3 & ATAC-seq'
    tpm_df.loc[tpm_df['name'].isin(genes_with_h3k27ac_vec), 'group'] = 'with H3K27ac & ATAC-seq'
    tpm_df.loc[tpm_df['name'].isin(genes_with_h3k4me1_vec), 'group'] = 'with H3K4me1 & ATAC-seq'

    tpm_df_enh_tmp = tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec),:]
    tpm_df_enh_tmp.loc[:, 'group'] = 'with enhancers'
    tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = 'with enhancer cov < 0.5'
    tpm_df.loc[tpm_df['name'].isin(genes_withEnh_upperHalf_vec), 'group'] = 'with enhancer cov > 0.5'
    tpm_df = pd.concat([tpm_df, tpm_df_enh_tmp], ignore_index=True)
    sorter = ['w/o enhancer', 'with ATAC-seq', 'with enhancer cov < 0.5', 'with enhancer cov > 0.5', 'with enhancers', 'with H3K27ac & ATAC-seq', 'with H3K4me1 & ATAC-seq', 'with H3K4me3 & ATAC-seq']
    tpm_df.group = tpm_df.group.astype("category")
    tpm_df.group.cat.set_categories(sorter)
    time_int = time_dic[time][0]
    
    # tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01) #str(time_int) + 'AB'

    sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
    # sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
    g = sns.violinplot(data=tpm_df, y='log2_%sAB_introns' % (time_int), x='group')
    add_stat_annotation(g, data=tpm_df, x='group', y='log2_%sAB_introns' % (time_int), #order=order,
                        box_pairs=[('w/o enhancer', 'with enhancer cov < 0.5'),
                                    ('w/o enhancer', 'with enhancer cov > 0.5'),
                                    ('w/o enhancer', 'with enhancers'),
                                    ('with enhancers', 'with H3K4me3 & ATAC-seq'),
                                    ('with enhancers', 'with H3K4me1 & ATAC-seq'),
                                    ('with enhancer cov > 0.5', 'with H3K4me1 & ATAC-seq'),
                                    ('with enhancer cov < 0.5', 'with H3K4me1 & ATAC-seq'),
                                    ('with enhancers', 'with H3K27ac & ATAC-seq')],
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # g.tick_params(axis='x', rotation=30)
    wrap_labels(g, 10)
    # plt.savefig("results/pics/paper/%sAB_log2TPM_boxplot_comparedToEnhancersCovATACchip.enhWithouth3k4me3.pdf" % (time_int), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()

# %%
from statannot import add_stat_annotation

time = 'V'
# tpm_df_rosengarten = create_rnaseq_df_rosengarten()
# tpm_df = tpm_df_rosengarten.reset_index()
tpm_df = create_rnaseq_df(need_coord=True)
# tpm_df = df_TPM_genesIntrons
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end', 'name']].sort_values(by=['chrom', 'start']))



for time in ['V']: #, 'S', 'M'

    V_enhancer_noPromoter_bed = BedTool('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % (time))
    #('results/pseudoDifEnhacerswithAtac.bed')#
    # V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    # V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    # time_atac = time_dic[time][1]
    # V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) #Vegetative_r123.narrowPeak") #Vegetative.mRp.clN_peaks.narrowPeak") #
    # V_h3k4me3 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][3]))

    # genes_with_atacseq_bed = tpm_bed.intersect(V_atacseq, f=0.2)
    # genes_with_atacseq_vec = tpm_bed.intersect(V_atacseq, f=0.2).to_dataframe().name.tolist()
    # genes_with_h3k27ac_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k27ac, f=0.2).intersect(V_h3k4me1, f=0.2, v=True).intersect(V_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist()

    # genes_with_h3k4me1_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k4me1, f=0.2).intersect(V_h3k27ac, f=0.2, v=True).intersect(V_h3k4me3, f=0.2, v=True).to_dataframe().name.tolist()

    # genes_with_h3k4me3_vec = tpm_bed.intersect(genes_with_atacseq_bed, f=0.2).intersect(V_h3k4me3, f=0.2).intersect(V_h3k4me1, f=0.2, v=True).intersect(V_h3k27ac, f=0.2, v=True).to_dataframe().name.tolist()

    genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed, f=0.2).to_dataframe().name.tolist()
    genes_withEnh_upperHalf_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed, f=0.5).to_dataframe().name.tolist()
    tpm_df['group'] = 'w/o enhancer'

    # tpm_df.loc[tpm_df['name'].isin(genes_with_atacseq_vec), 'group'] = 'with ATAC-seq'
    # tpm_df.loc[tpm_df['name'].isin(genes_with_h3k4me3_vec), 'group'] = 'with H3K4me3 & ATAC-seq'
    # tpm_df.loc[tpm_df['name'].isin(genes_with_h3k27ac_vec), 'group'] = 'with H3K27ac & ATAC-seq'
    # tpm_df.loc[tpm_df['name'].isin(genes_with_h3k4me1_vec), 'group'] = 'with H3K4me1 & ATAC-seq'

    tpm_df_enh_tmp = tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec),:]
    tpm_df_enh_tmp.loc[:, 'group'] = 'with enhancers'
    # tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = 'with enhancer cov < 0.5'
    # tpm_df.loc[tpm_df['name'].isin(genes_withEnh_upperHalf_vec), 'group'] = 'with enhancer cov > 0.5'
    tpm_df = pd.concat([tpm_df, tpm_df_enh_tmp], ignore_index=True)
    # sorter = ['w/o enhancer', 'with ATAC-seq', 'with enhancer cov < 0.5', 'with enhancer cov > 0.5', 'with enhancers', 'with H3K27ac & ATAC-seq', 'with H3K4me1 & ATAC-seq', 'with H3K4me3 & ATAC-seq']
    # tpm_df.group = tpm_df.group.astype("category")
    # tpm_df.group.cat.set_categories(sorter)
    # time_int = time_dic[time][0]
    time_int=0
    tpm_df['log2_%sAB_introns' % str(time_int)] = np.log2(tpm_df[str(8) + 'AB']+0.01)

    sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
    # sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
    g = sns.violinplot(data=tpm_df, y='log2_0AB_introns', x='group')
    add_stat_annotation(g, data=tpm_df, x='group', y='log2_0AB_introns', #order=order,
                        box_pairs=[('w/o enhancer', 'with enhancers')],
                        test='Mann-Whitney-ls', text_format='star', loc='inside', verbose=2)
    # g.tick_params(axis='x', rotation=30)
    wrap_labels(g, 10)
    # plt.savefig("results/pics/paper/%sAB_log2TPM_boxplot_comparedToDifEnh.simplified.introns.pdf" % (time_int), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()
    
# %%
from statannot import add_stat_annotation

# tpm_df_rosengarten = create_rnaseq_df_rosengarten()
# tpm_df = tpm_df_rosengarten.reset_index()
tpm_df = create_rnaseq_df(need_coord=True)
# tpm_df = df_TPM_genesIntrons
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end', 'name']].sort_values(by=['chrom', 'start']))

for time in ['V', 'S', 'M']:
    V_enhancer_noPromoter_bed = BedTool('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % time)
    genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed, f=0.2).to_dataframe().name.tolist()

    tpm_df['group'] = 'w/o enhancer'

    tpm_df_enh_tmp = tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec),:]
    tpm_df_enh_tmp.loc[:, 'group'] = 'with enhancers'

    tpm_df = pd.concat([tpm_df, tpm_df_enh_tmp], ignore_index=True)

    tpm_df['log2_%sAB' % str(time_dic[time][0])] = np.log2(tpm_df['%sAB' % str(time_dic[time][0])]+0.01)

    sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
    # sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
    g = sns.violinplot(data=tpm_df, y='log2_%sAB' % str(time_dic[time][0]), x='group')
    add_stat_annotation(g, data=tpm_df, x='group', y='log2_%sAB' % str(time_dic[time][0]), #order=order,
                        box_pairs=[('w/o enhancer', 'with enhancers')],
                        test='Mann-Whitney-ls', text_format='star', loc='inside', verbose=2)
    # g.tick_params(axis='x', rotation=30)
    wrap_labels(g, 10)
    # plt.savefig("results/pics/paper/%sAB_log2TPM_boxplot_comparedToDifEnh.simplified.pdf" % str(time_dic[time][0]), dpi=100, bbox_inches='tight')
    plt.show()
    plt.clf()

# %%
# bedtools coverage -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K4me1_idr_peaks.narrowPeak results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_r1234.mergeIDR.bed | awk '$5 == 0' > results/0_pseudoDifEnhacerswithAtac.bed

# bedtools coverage -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_r1234.mergeIDR.bed | awk '$5 == 0' > results/0_pseudoDifEnhacersAtacH3K27ac.bed

# bedtools coverage -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_r1234.mergeIDR.bed | awk '$5 == 0' > results/5_not8_pseudoDifEnhacersAtacH3K27ac.bed

# bedtools coverage -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.noH3k4me3.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_r1234.mergeIDR.bed | awk '$5 == 0' > results/5_not8_pseudoDifEnhacersnotH3K4me3.AtacH3K27ac.bed

# bedtools coverage -a results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.noH3k4me3.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Streaming_r123.mergeIDR.bed | awk '$5 == 0' > results/8_not5_pseudoDifEnhacersnotH3K4me3.AtacH3K27ac.bed


# bedtools coverage -a results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idr_peaks.narrowPeak results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_r123.sorted.narrowPeak | awk '$5 == 0' > results/5_not0_pseudoDifEnhacersAtacH3K27ac.bed

# bedtools coverage -a results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/S_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Streaming_r123.mergeIDR.bed | awk '$5 == 0' > results/8_not5_pseudoDifEnhacersAtacH3K27ac.bed

# bedtools coverage -a results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idr_peaks.narrowPeak results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_r123.sorted.narrowPeak | awk '$5 == 0' > results/8_pseudoDifEnhacerswithAtac.bed

# bedtools coverage -a results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K27ac_idr_peaks.narrowPeak results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_r123.sorted.narrowPeak | awk '$5 == 0' > results/8_pseudoDifEnhacersAtacH3K27ac.bed

# bedtools coverage -a results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/V_H3K4me1_idr_peaks.narrowPeak =results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_r123.sorted.narrowPeak | awk '$5 == 0' > results/8_pseudoDifEnhacerswithAtac.bed
# tpm_df_rosengarten = create_rnaseq_df_rosengarten()
tpm_df = df_TPM_genesIntrons.copy()
# tpm_df = create_rnaseq_df(need_coord=True)
tpm_df['group'] = 'w/o enhancer'
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end', 'name']].sort_values(by=['chrom', 'start']))

V_enhancer_noPromoter_bed = BedTool('results/M_notS_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.noH3k4me3.bed')
# V_enhancer_noPromoter= bioframe.read_table('results/5_pseudoDifEnhacersAtacH3K27ac.bed', schema='bed')
genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed, f=0.2).to_dataframe().name.tolist()
# V_enhancer_noPromoter_expanded = bioframe.expand(V_enhancer_noPromoter, pad=2000)
# genes_inEnhWindow_vec = bioframe.overlap(V_enhancer_noPromoter_expanded, tpm_df, how='inner').name_.tolist()
# V_enhancer_noPromoter_bed.closest(tpm_bed).to_dataframe().columns[11]
genes_closeEnh_vec = V_enhancer_noPromoter_bed.closest(tpm_bed).to_dataframe().thickEnd.tolist() #DDB_G0267526 DDB_G0269296
tpm_df.loc[tpm_df['name'].isin(genes_closeEnh_vec),'group'] = 'close enhancers'
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec),'group'] = 'with enhancers'
# tpm_df.loc[tpm_df['name'].isin(genes_inEnhWindow_vec),'group'] = 'in window enhancers'
# tpm_df = pd.concat([tpm_df, tpm_df_enh_tmp], ignore_index=True)
for time_int in [0, 5, 8]:
    tpm_df['log2_%sAB' % str(time_int)] = np.log2(tpm_df[str(time_int) + 'AB_introns']+0.01)
    # tpm_df['log2_%sAB' % str(time_int)] = np.log2(tpm_df['hr0'+str(time_int)]+0.01)
tpm_df['delta_85'] = tpm_df['log2_8AB'] - tpm_df['log2_5AB']
tpm_df_filt = tpm_df.filter(regex='(delta_)|(group)|(log2_)')
tpm_df_melter = pd.melt(tpm_df_filt, id_vars="group") #.query('group != "w/o enhancer"')

tpm_df_melter.head()
sns.violinplot(tpm_df, x='group', y='delta_85')
# sns.scatterplot(tpm_df, x='log2_8AB', y='log2_0AB', hue='group', alpha=0.6)
# %%

V_enhancer_bed = V_atacseq.intersect(V_chipseq_bed, f=0.25, F=0.25)
V_enh_withH3k4me3 = V_enhancer_bed.coverage(V_h3k4me3).to_dataframe()
# sns.histplot(data=V_enh_withH3k4me3, x='thickEnd', kde=True,  stat="percent")
V_enh_withH3k4me3['length'] = V_enh_withH3k4me3['end'] - V_enh_withH3k4me3['start']
sns.scatterplot(V_enh_withH3k4me3, x='thickEnd', y='length')
#.intersect(V_h3k4me3, v=True)#.to_dataframe().iloc[:,[0,1,2]] #wo=True, .iloc[:,[0,1,3,4,5]]   

# %%
V_h3k4me3_df = V_h3k4me3.to_dataframe().iloc[:,[0,1,2]]
V_h3k4me3_df.columns = ['chrom', 'start', 'end']
V_h3k4me3_df['length'] = V_h3k4me3_df['end'] - V_h3k4me3_df['start']
sns.histplot(data=V_h3k4me3_df, x='length', kde=True,  stat="percent")

# %% fraction of feature covered with ATAC-seq
for feature in ['exons', 'introns', 'intergenic', 'promoters']:
    df = pd.read_table('results/%s_covATAC.txt' % (feature), header=None).iloc[:,[4,5]] #
    df.columns = ['covered', 'length']
    print(feature, ": ", np.sum(df.covered)*100/np.sum(df.length))


# %% check los-loops
for time in [0, 8]:
    file_loops = 'data/las_loops/%sAB_las_loops_4kb.bed'
    file_features = 'results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed'
    for mode in ['anchors']: # [ 'anchors', 'anchors_3bins', , 'start', 'end'  ]: #
        print(pValue_featureOccurenceInLoop(file_loops=file_loops,
                                            time=time, mode=mode, N_shuffle=100,
                                            file_features=file_features,
                                            name='%s_enhancersAtLasLoops_mode_' % (time) + mode))
# %% check data/genome/0AB_loops_withGroup' + str(cl) + '.bed3
for cl in range(0,6):
    file_loops = 'data/genome/0AB_loops_withGroup%s.bed3' % cl
    file_features = 'results/V_specific_enhancers.bed' #'results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed'
    for mode in ['whole']: # , 'anchors_3bins', 'inside' [ 'anchors', 'anchors_3bins', , 'start', 'end'  ]: #
        print(pValue_featureOccurenceInLoop(file_loops=file_loops,
                                            time=None, mode=mode, N_shuffle=1000,
                                            file_features=file_features,
                                            name='V-specific_enhancersInLoops_withGroup%s_mode_%s' % (cl, mode)))

# %% check number of enhancers inside gene and outside
genes_bed = BedTool("data/genome/genes_noRTE_no_pseudogenes.bed")
enhancers_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

print(enhancers_bed.intersect(genes_bed, wa=True, f=1.0, u=True).to_dataframe().shape[0]/enhancers_bed.to_dataframe().shape[0])
print(enhancers_bed.intersect(genes_bed, wa=True, f=1.0, v=True).to_dataframe().shape[0]/enhancers_bed.to_dataframe().shape[0])
"""
Part 5. Select differential enhancers
"""
# %% one enhacner one gene
from statannot import add_stat_annotation

time = 'V'
# tpm_df = df_TPM_genesIntrons
tpm_df = create_rnaseq_df(need_coord=True)
tpm_bed = BedTool.from_dataframe(tpm_df.loc[:,['chrom', 'start', 'end', 'name']].sort_values(by=['chrom', 'start']))
tpm_df['log2_0_TPM'] = np.log2(tpm_df['0AB']+0.01)
tpm_df['log2_2_TPM'] = np.log2(tpm_df['2AB']+0.01)
tpm_df['log2_5_TPM'] = np.log2(tpm_df['5AB']+0.01)
tpm_df['log2_8_TPM'] = np.log2(tpm_df['8AB']+0.01)
tpm_df['delta_8_0'] = tpm_df['8AB'] - tpm_df['0AB']
# tpm_df['delta_8_0'] = tpm_df['log2_8AB_introns'] - tpm_df['log2_0AB_introns']

V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
M_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

VM_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, wa=True).saveas('results/VM_common_enhancers.bed')
V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, v=True).saveas('results/V_specific_enhancers.bed')
M_specific_enhancers_bed = M_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True).saveas('results/M_specific_enhancers.bed')
S_specific_enhancers_bed = S_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True).saveas('results/S_specific_enhancers.bed')
genes_withEnh_vec = tpm_bed.intersect(VM_common_enhancers_bed).to_dataframe().name.tolist()
genes_withVspecEnh_vec = tpm_bed.intersect(V_specific_enhancers_bed).to_dataframe().name.tolist()
genes_withSspecEnh_vec = tpm_bed.intersect(M_specific_enhancers_bed).to_dataframe().name.tolist()

tpm_df['group'] = 'w/o enhancer'
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = 'V-S common'
tpm_df.loc[tpm_df['name'].isin(genes_withVspecEnh_vec), 'group'] = 'V-specific'
tpm_df.loc[tpm_df['name'].isin(genes_withSspecEnh_vec), 'group'] = 'S-specific'

sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
# sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
g = sns.boxplot(data=tpm_df, y='log2_8_TPM', x='group')
add_stat_annotation(g, data=tpm_df, x='group', y='log2_8_TPM', 
                    box_pairs=[('w/o enhancer', 'V-S common'),
                                ('w/o enhancer', 'V-specific'),
                                ('V-specific', 'S-specific'),
                                ('V-S common', 'V-specific'),
                                ('V-S common', 'S-specific')],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# g.tick_params(axis='x', rotation=30)
wrap_labels(g, 10)
g.axhline(0)
# plt.savefig("results/pics/paper/log2_0AB_introns_TPMintrons_boxplot_difEnh.pdf", dpi=100, bbox_inches='tight')
plt.show()

# %% one gene one enhancer without stage specific
# %% one enhacner one gene
from statannot import add_stat_annotation

time = 'V'
# tpm_df = create_rnaseq_df(need_coord=True)
# tpm_bed = BedTool.from_dataframe(tpm_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))
# tpm_df['log2_0_TPM'] = np.log2(tpm_df['0AB']+0.01)
# tpm_df['log2_2_TPM'] = np.log2(tpm_df['2AB']+0.01)
# tpm_df['log2_5_TPM'] = np.log2(tpm_df['5AB']+0.01)
# tpm_df['log2_8_TPM'] = np.log2(tpm_df['8AB']+0.01)
# tpm_df['delta_8_0'] = tpm_df['8AB'] - tpm_df['0AB']

V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

VS_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, wa=True).saveas('results/VM_common_enhancers.bed')
V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, v=True).saveas('results/V_specific_enhancers.bed')
S_specific_enhancers_bed = M_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True).saveas('results/M_specific_enhancers.bed')
genes_withEnh_vec = tpm_bed.intersect(V_enhancer_noPromoter_bed).to_dataframe().name.tolist()
genes_withVspecEnh_vec = tpm_bed.intersect(V_specific_enhancers_bed).to_dataframe().name.tolist()
genes_withSspecEnh_vec = tpm_bed.intersect(S_specific_enhancers_bed).to_dataframe().name.tolist()

tpm_df['group'] = 'w/o enhancer'
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = 'with enhancer'

sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
# sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
g = sns.boxplot(data=tpm_df, y='delta_8_0', x='group')
add_stat_annotation(g, data=tpm_df, x='group', y='delta_8_0', #order=order,
                    box_pairs=[('w/o enhancer', 'with enhancer')],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# g.tick_params(axis='x', rotation=30)
wrap_labels(g, 10)
g.axhline(0)
plt.savefig("results/pics/paper/delta_8_0_TPM_boxplot_enh.pdf", dpi=100, bbox_inches='tight')
plt.show()

# %%
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes
sns.set(style='white', context='poster', rc={'figure.figsize':(40,40)})
tpm_df_with = tpm_df.loc[tpm_df.group != 'w/o enhancer',]
tpm_df_with = tpm_df_with.loc[tpm_df.group != 'V-S common',]
tpm_df_with = tpm_df_with.loc[tpm_df.group != 'V-specific',]
# g = sns.scatterplot(data=tpm_df_with, y='log2_8_TPM', x='log2_0_TPM', hue='group', alpha=0.5, size=10)
f, ax = plt.subplots(figsize=(6, 6))
colors = {'V-specific':'red', 'S-specific':'green'}

ax.scatter(data=tpm_df_with, y='log2_8_TPM', x='log2_5_TPM', 
alpha=0.2, c=tpm_df_with['group'].map(colors))
# g = sns.lmplot(x = "log2_0_TPM", y = "log2_8_TPM", col = "group",
#            hue = "group", data = tpm_df_with, 
#            scatter_kws={"s": 20})
# axes.Axes.axline((0, 0), (1, 1), linewidth=4, color='r')
# ax.plot([-5, -5], [10, 10], ls="--", c=".3")
add_identity(ax, color='r', ls="--")
#plt.legend([],[], frameon=False)
plt.savefig("results/pics/paper/scatter_8_5_log2TPM_boxplot_difEnh.pdf", dpi=100,
            bbox_inches='tight')
plt.show()

# %% Venn
sys.path.append('~/projects/dicty/hic_loop_study/scripts/functions/modules/')
from venn_mpl import venn_mpl
V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
M_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

sns.set(style='white', context='poster', rc={'figure.figsize':(10,10)})
venn_mpl(V_enhancer_noPromoter_bed, 
         S_enhancer_noPromoter_bed,
         M_enhancer_noPromoter_bed,
         outfn='results/pics/paper/enh_venn.pdf', 
         labels=['V', 'S', 'M'])

# %% add  all stages to plot
tpm_df_filt = tpm_df.filter(regex='(log2_)|(group)')
tpm_df_melter = pd.melt(tpm_df_filt, id_vars="group")
sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
# sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
g = sns.boxplot(data=tpm_df_melter, y='value', x='group', hue='variable')
add_stat_annotation(g, data=tpm_df_melter, x='group', y='value',#order=order,
                    box_pairs=[('w/o enhancer', 'V-S common'),
                                ('w/o enhancer', 'V-specific'),
                                ('V-specific', 'S-specific'),
                                ('V-S common', 'V-specific'),
                                ('V-S common', 'S-specific')],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# g.tick_params(axis='x', rotation=30)
wrap_labels(g, 10)
plt.savefig("results/pics/paper/log2TPM_boxplot_difEnh_8_0.pdf", dpi=100,
            bbox_inches='tight')

# %% select closest gene but not the one with enhancer
from statannot import add_stat_annotation

time = 'V'
# tpm_df = create_rnaseq_df(need_coord=True)
# tpm_bed = BedTool.from_dataframe(tpm_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))
# tpm_df['log2_0_TPM'] = np.log2(tpm_df['0AB']+0.01)
# tpm_df['log2_2_TPM'] = np.log2(tpm_df['2AB']+0.01)
# tpm_df['log2_5_TPM'] = np.log2(tpm_df['5AB']+0.01)
# tpm_df['log2_8_TPM'] = np.log2(tpm_df['8AB']+0.01)
# tpm_df['delta_5_0'] = tpm_df['log2_8_TPM'] - tpm_df['log2_0_TPM']

V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

VS_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, wa=True).saveas('results/VM_common_enhancers.bed')
V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, v=True).saveas('results/V_specific_enhancers.bed')
S_specific_enhancers_bed = S_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True).saveas('results/M_specific_enhancers.bed')
genes_withEnh_vec = VS_common_enhancers_bed.closest(tpm_bed, io=True).to_dataframe().thickStart.tolist()
genes_withVspecEnh_vec = V_specific_enhancers_bed.closest(tpm_bed, io=True).to_dataframe().thickStart.tolist()
genes_withSspecEnh_vec = S_specific_enhancers_bed.closest(tpm_bed, io=True).to_dataframe().thickStart.tolist()

tpm_df['group'] = 'w/o enhancer'
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = 'V-S common'
tpm_df.loc[tpm_df['name'].isin(genes_withVspecEnh_vec), 'group'] = 'V-specific'
tpm_df.loc[tpm_df['name'].isin(genes_withSspecEnh_vec), 'group'] = 'S-specific'

sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
# sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')
g = sns.boxplot(data=tpm_df, y='delta_8_0', x='group')
add_stat_annotation(g, data=tpm_df, x='group', y='delta_8_0', #order=order,
                    box_pairs=[('w/o enhancer', 'V-S common'),
                                ('w/o enhancer', 'V-specific'),
                                ('V-specific', 'S-specific'),
                                ('V-S common', 'V-specific'),
                                ('V-S common', 'S-specific')],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# g.tick_params(axis='x', rotation=30)
wrap_labels(g, 10)
plt.savefig("results/pics/paper/0_8_delta_log2TPMintrons_boxplot_difEnhClosestGene.pdf", dpi=100,
            bbox_inches='tight')
plt.show()


# %% select genes in window from enhancer but not merge them
from statannot import add_stat_annotation

window = 4000
time = 'V'
tpm_df = create_rnaseq_df(need_coord=True)
tpm_bed = BedTool.from_dataframe(tpm_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))
tpm_df['log2_0_TPM'] = np.log2(tpm_df['0AB']+0.01)
tpm_df['log2_2_TPM'] = np.log2(tpm_df['2AB']+0.01)
tpm_df['log2_5_TPM'] = np.log2(tpm_df['5AB']+0.01)
tpm_df['log2_8_TPM'] = np.log2(tpm_df['8AB']+0.01)
tpm_df['delta_5_0'] = tpm_df['8AB'] - tpm_df['0AB']

V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
V_enhancer_noPromoter_bed_window = V_enhancer_noPromoter_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)


VS_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, wa=True)
VS_common_enhancers_bed_window = VS_common_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, v=True)
V_specific_enhancers_bed_window = V_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
S_specific_enhancers_bed = S_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True)
S_specific_enhancers_bed_window = S_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
genes_withEnh_vec = tpm_bed.intersect(VS_common_enhancers_bed_window, wb=True).to_dataframe().name.tolist()
genes_withVspecEnh_vec = tpm_bed.intersect(V_specific_enhancers_bed_window).to_dataframe().name.tolist()
genes_withSspecEnh_vec = tpm_bed.intersect(S_specific_enhancers_bed_window).to_dataframe().name.tolist()

tpm_df['group'] = 'w/o enhancer'
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'group'] = 'V-S common'
tpm_df.loc[tpm_df['name'].isin(genes_withVspecEnh_vec), 'group'] = 'V-specific'
tpm_df.loc[tpm_df['name'].isin(genes_withSspecEnh_vec), 'group'] = 'S-specific'

tpm_df_filt = tpm_df.filter(regex='(delta_)|(group)')
tpm_df_melter = pd.melt(tpm_df_filt, id_vars="group")
sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
# sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')

g = sns.boxplot(data=tpm_df_melter, y='value', x='group', hue='variable')
add_stat_annotation(g, data=tpm_df_melter, x='group', y='value',#order=order,
                    box_pairs=[('w/o enhancer', 'V-S common'),
                                ('w/o enhancer', 'V-specific'),
                                ('V-specific', 'S-specific'),
                                ('V-S common', 'V-specific'),
                                ('V-S common', 'S-specific')],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# g.tick_params(axis='x', rotation=30)
wrap_labels(g, 10)
plt.savefig("results/pics/paper/delta_8_0_TPM_boxplot_difEnh_window%s.pdf" % (window), dpi=100,
            bbox_inches='tight')

# %% sum/mean/median for genes
from statannot import add_stat_annotation

# tpm_df = create_rnaseq_df(need_coord=True)
# tpm_bed = BedTool.from_dataframe(tpm_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))
# tpm_df['log2_0_TPM'] = np.log2(tpm_df['0AB']+0.01)
# tpm_df['log2_2_TPM'] = np.log2(tpm_df['2AB']+0.01)
# tpm_df['log2_5_TPM'] = np.log2(tpm_df['5AB']+0.01)
# tpm_df['log2_8_TPM'] = np.log2(tpm_df['8AB']+0.01)
# tpm_df['delta_8_0'] = tpm_df['8AB'] - tpm_df['0AB']

V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
M_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

for window in [2000, 4000, 6000, 8000]:
    time = 'V'
    V_enhancer_noPromoter_bed_window = V_enhancer_noPromoter_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
    VS_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, wa=True)
    VS_common_enhancers_bed_window = VS_common_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
    V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, v=True)
    V_specific_enhancers_bed_window = V_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
    S_specific_enhancers_bed = S_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True)
    S_specific_enhancers_bed_window = S_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)

    tpm_df_common = get_expression_sum_interesectingFeature(feature=VS_common_enhancers_bed_window, tpm_df=tpm_df, group='V-M common')
    tpm_df_v = get_expression_sum_interesectingFeature(feature=V_specific_enhancers_bed_window, tpm_df=tpm_df, group='V-specific')
    tpm_df_s = get_expression_sum_interesectingFeature(feature=S_specific_enhancers_bed_window, tpm_df=tpm_df, group='M-specific')
    tpm_df_enh = pd.concat([tpm_df_common, tpm_df_v, tpm_df_s], ignore_index=True)

    tpm_df_filt = tpm_df_enh.filter(regex='(delta_)|(group)')
    tpm_df_melter = pd.melt(tpm_df_filt, id_vars="group")
    sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
    # sns.scatterplot(data=tpm_df, x='0AB', y='5AB', hue='group')

    g = sns.boxplot(data=tpm_df_melter, y='value', x='group', hue='variable')
    add_stat_annotation(g, data=tpm_df_melter, x='group', y='value',#order=order,
                        box_pairs=[#('w/o enhancer', 'V-S common'),
                                    #('w/o enhancer', 'V-specific'),
                                    ('V-specific', 'M-specific'),
                                    ('V-M common', 'V-specific'),
                                    ('V-M common', 'M-specific')],
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # g.tick_params(axis='x', rotation=30)
    wrap_labels(g, 10)
    plt.savefig("results/pics/paper/delta_8_0_TPMintrons_boxplot_difEnh_Meanwindow%s.pdf" % (window), dpi=100, bbox_inches='tight')
    plt.close()


# %% same as above but do not include intersecting gene
from statannot import add_stat_annotation

tpm_TSS_df = create_rnaseq_df(need_TSS=True)
tpm_TSS_bed = BedTool.from_dataframe(tpm_TSS_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'TSS_start']))
tpm_TSS_df['log2_0_TPM'] = np.log2(tpm_TSS_df['0AB']+0.01)
tpm_TSS_df['log2_2_TPM'] = np.log2(tpm_TSS_df['2AB']+0.01)
tpm_TSS_df['log2_5_TPM'] = np.log2(tpm_TSS_df['5AB']+0.01)
tpm_TSS_df['log2_8_TPM'] = np.log2(tpm_TSS_df['8AB']+0.01)
tpm_TSS_df['delta_8_0'] = tpm_TSS_df['log2_8_TPM'] - tpm_TSS_df['log2_0_TPM']

V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
S_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
M_enhancer_noPromoter_bed = BedTool('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')

V_enhancer_noPromoter_bed_window = V_enhancer_noPromoter_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)

VM_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, wa=True, f=0.5)
VM_common_enhancers_bed_window = VM_common_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(M_enhancer_noPromoter_bed, v=True)
V_specific_enhancers_bed_window = V_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
M_specific_enhancers_bed = M_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True)
M_specific_enhancers_bed_window = M_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)


tpm_df_v = tpm_TSS_bed.closest(M_specific_enhancers_bed, d=True).to_dataframe().query('thickEnd > 0') # and thickEnd < 800 # io=True, 
tpm_TSS_df['group'] = '< 1000'
for window in [1000, 8000]: #, 4000, 6000, , 10000]
    tpm_TSS_df.loc[tpm_TSS_df.name.isin(tpm_df_v.loc[tpm_df_v.thickEnd > window, 'name']), 'group'] = '> ' + str(window)
    
    # tpm_df_common = get_expression_sum_interesectingFeature(feature=VS_common_enhancers_bed_window, tpm_df=tpm_TSS_df, group='VM common')
    # tpm_df_v = get_expression_sum_interesectingFeature(feature=V_specific_enhancers_bed_window, tpm_df=tpm_TSS_df, group='V-specific')
    # tpm_df_s = get_expression_sum_interesectingFeature(feature=S_specific_enhancers_bed_window, tpm_df=tpm_TSS_df, group='M-specific')
    # tpm_df_enh = pd.concat([tpm_df_common, tpm_df_v, tpm_df_s], ignore_index=True)

# tpm_df_filt = tpm_TSS_df.filter(regex='(delta_)|(group)')
# tpm_df_melter = pd.melt(tpm_df_filt, id_vars="group")
tpm_TSS_df_filt = tpm_TSS_df.loc[tpm_TSS_df.group != "> 1000"]
tpm_TSS_df_filt = tpm_TSS_df_filt.loc[tpm_TSS_df_filt.group != "> 8000"]
# tpm_TSS_df_filt = tpm_TSS_df_filt.loc[tpm_TSS_df_filt.group != "< 2000"]
sns.set(style='white', context='poster', rc={'figure.figsize':(28,20)})
# sns.scatterplot(data=tpm_TSS_df_filt, x='log2_0_TPM', y='log2_8_TPM', hue='group')
f, ax = plt.subplots(figsize=(10, 10))
colors = {'< 1000': 'blue', '> 1000':'green', '> 4000':'green', '> 6000': 'blue', '> 8000':'red'}
ax.scatter(data=tpm_TSS_df_filt, y='log2_8_TPM', x='log2_0_TPM', 
alpha=0.2, c=tpm_TSS_df_filt['group'].map(colors))
# ax.scatter(data=tpm_TSS_df, y='log2_8_TPM', x='log2_0_TPM', 
# alpha=0.2)
add_identity(ax, color='r', ls="--")

# g = sns.boxplot(data=tpm_df_melter, y='value', x='group', hue='variable')
# add_stat_annotation(g, data=tpm_df_melter, x='group', y='value',#order=order,
#                     box_pairs=[#('w/o enhancer', 'V-S common'),
#                                 #('w/o enhancer', 'V-specific'),
#                                 ('V-specific', 'M-specific'),
#                                 ('VM common', 'V-specific'),
#                                 ('VM common', 'M-specific')],
#                     test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# # g.tick_params(axis='x', rotation=30)
# wrap_labels(g, 10)
# plt.savefig("results/pics/paper/delta_8_0_log2TPM_scatter_difEnh_Meanwindow%s.pdf" % (window), dpi=100,
#             bbox_inches='tight')
plt.show()
# plt.close()

# %% find and load to python loops with enhancers
time = 0
# find loops with enhancers at anchors
# intersectBed -a data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bedpe \
# -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed \
# -wa > data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.withEnhAtAnchor.bedpe

# intersectBed -a data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed \
# -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed \
# -wa > data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.withEnh.bed

# data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bedpe

df_loops_bedpe_withEnh = bioframe.read_table(
    'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.withEnh.bedpe' % (time),
    schema='bedpe'
)

df_loops_bed_withEnh = df_loops_bedpe_withEnh.iloc[:, [0, 2, 4]]
df_loops_bed_withEnh.columns = ['chrom', 'start', 'end']
df_loops_bed_withEnh.to_csv('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.withEnh.bed', sep="\t", index=False,header=False)

# %%
from statannot import add_stat_annotation

for window in [2000]: #, 4000, 6000, 8000]
    time = 'V'
    tpm_df = create_rnaseq_df(need_coord=True)
    tpm_bed = BedTool.from_dataframe(tpm_df.iloc[:,[0,1,2,3]].sort_values(by=['chrom', 'start']))
    tpm_df['log2_0_TPM'] = np.log2(tpm_df['0AB']+0.01)
    tpm_df['log2_2_TPM'] = np.log2(tpm_df['2AB']+0.01)
    tpm_df['log2_5_TPM'] = np.log2(tpm_df['5AB']+0.01)
    tpm_df['log2_8_TPM'] = np.log2(tpm_df['8AB']+0.01)

    V_enhancer_noPromoter_bed = BedTool('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
    S_enhancer_noPromoter_bed = BedTool('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed')
    V_enhancer_noPromoter_bed_window = V_enhancer_noPromoter_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)

    VS_common_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, wa=True)
    VS_common_enhancers_bed_window = VS_common_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
    V_specific_enhancers_bed = V_enhancer_noPromoter_bed.intersect(S_enhancer_noPromoter_bed, v=True)
    V_specific_enhancers_bed_window = V_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)
    S_specific_enhancers_bed = S_enhancer_noPromoter_bed.intersect(V_enhancer_noPromoter_bed, v=True)
    S_specific_enhancers_bed_window = S_specific_enhancers_bed.slop(g='data/genome/dicty.chrom.sizes', l=window, r=window)

# intersectBed -a data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bedpe \
# -b results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed \
# -wa > data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.withEnhAtAnchor.bedpe
V_loops = bioframe.read_table('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.1bins_added.bedpe', schema='bedpe')
V_loops['group'] = 'noEnh'
V_loops_left_bed = BedTool.from_dataframe(V_loops.loc[:, ['chrom1', 'start1', 'end1']])
V_loops_right_bed = BedTool.from_dataframe(V_loops.loc[:, ['chrom2', 'start2', 'end2']])
V_loops_left_with_enh = V_loops_left_bed.intersect(VS_common_enhancers_bed, loj=True).to_dataframe()
V_loops_right_with_enh = V_loops_right_bed.intersect(VS_common_enhancers_bed, loj=True).to_dataframe()

V_loops.loc[V_loops_left_with_enh.strand > 0, 'group'] = 'centEnh'
V_loops.loc[V_loops_right_with_enh.strand > 0, 'group'] = 'telEnh'

# %%
from sum_expr_around_anchors \
    import compute_argminDerative, compute_LoopExprAndStrength, compute_Derivative, compute_exprSumAroundBedAnchor, create_distancesToNextLoop
# ['sum_only']: #  , 'sum_and_subtraction', 'ctrl_for_sum_only'
tmp_genes = compute_LoopExprAndStrength(#middle_pixel_dict,
                                        mode="two_strands_per_anchor",
                                        twostrands_mode='sum_only',  #'sum_only', #,sum_and_subtraction
                                        prefix='0AB',
                                        which_to_compute_list=['0AB', '2AB', '5AB', '8AB',
                                                       'FC_5_0']
                                        )
V_loops['FC_5_0_sum_left'] = tmp_genes['FC_5_0_sum_left']
V_loops['FC_5_0_sum_right'] = tmp_genes['FC_5_0_sum_right']

# %%
g = sns.violinplot(data=V_loops, y='FC_5_0_sum_left', x='group')
plt.savefig("results/pics/paper/VScommonEnhloops_enh_left.pdf",
            format="pdf", dpi=300)
plt.clf()
g = sns.violinplot(data=V_loops, y='FC_5_0_sum_right', x='group')
plt.savefig("results/pics/paper/VScommonEnhloops_enh_right.pdf",
            format="pdf", dpi=300)
plt.clf()

# %%
for mode in ['mean']: #, 'median'
    # plot_around_anchors("bw/V_enhancers_cov.w500.zscore.bw",
    #                     "0_enhancerZscore_AroundAnchors_pybbi.%s.pdf" % (mode),
    #                     resolution=2000,
    #                     fill=True, mode=mode,
    #                     window=20000, nbins=21,
    #                     how_far_from_edge=10,
    #                     ymin=-0.1, ymax=1.0,
    #                     vmin=-0, vmax=1,
    #                     bed_list=["data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.withEnhAtAnchor.bedpe"])
    # plot_around_anchors("bw/V_enhancers_cov.w500.zscore.bw",
    #                     "0_enhancerZscore_AroundAnchorsFromLoopsWithEnh_pybbi.%s.pdf" % (mode),
    #                     resolution=2000,
    #                     fill=True, mode=mode,
    #                     window=20000, nbins=21,
    #                     how_far_from_edge=10,
    #                     ymin=-0.1, ymax=1.0,
    #                     vmin=-0, vmax=1,
    #                     bed_list=["data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.withEnh.bed"])
    # plot_around_anchors("bw/0_merged.TPM_log2.zscore.bw",
    #                     "0_TPMZscore_AroundEnh_pybbi.%s.pdf" % (mode),
    #                     resolution=2000,
    #                     fill=True, mode=mode,
    #                     window=20000, nbins=21,
    #                     how_far_from_edge=10,
    #                     ymin=-0.1, ymax=1.0,
    #                     vmin=-0, vmax=1,
    #                     bed_list=["results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed"])
    # plot_around_anchors('bw/0AB_CS.1kb.bw',
    #                     "0_CS_AroundEnh_pybbi.%s.pdf" % (mode),
    #                     resolution=2000,
    #                     fill=True, mode=mode,
    #                     window=20000, nbins=21,
    #                     how_far_from_edge=10,
    #                     ymin=-0.3, ymax=1.0,
    #                     vmin=-0.5, vmax=1,
    #                     bed_list=["results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed"])
    # plot_around_anchors('bw/IS/0AB.20k.IS.cooltools.bw',
    #                     "0_IS_AroundEnh_pybbi.%s.pdf" % (mode),
    #                     resolution=2000,
    #                     fill=True, mode=mode,
    #                     window=20000, nbins=21,
    #                     how_far_from_edge=10,
    #                     ymin=-0.1, ymax=1.0,
    #                     vmin=-0, vmax=1,
    #                     bed_list=["results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed"])
    # plot_around_anchors('bw/0AB_CSxRNA.1kb.zscore.bw',
    #                 "0_CSxRNA_AroundEnhSpecific_pybbi.%s.pdf" % (mode),
    #                 resolution=2000,
    #                 fill=True, mode=mode,
    #                 window=20000, nbins=21,
    #                 how_far_from_edge=10,
    #                 ymin=-0.1, ymax=1.0,
    #                 vmin=-0, vmax=1,
    #                 bed_list=["results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed", 'results/V_specific_enhancers.bed', 'results/S_specific_enhancers.bed', 'results/VS_common_enhancers.bed'])
    plot_around_anchors('bw/0_5_deltais.zscore.bw',
                    "0_5_deltaisZscore_AroundEnh_pybbi.%s.pdf" % (mode),
                    resolution=2000,
                    fill=True, mode=mode,
                    window=20000, nbins=21,
                    how_far_from_edge=10,
                    ymin=-0.1, ymax=1.0,
                    vmin=-0, vmax=1,
                    bed_list=["results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed", 'results/V_specific_enhancers.bed', 'results/S_specific_enhancers.bed', 'results/VS_common_enhancers.bed'])
    plot_around_anchors('bw/0_5_deltaTPM_log2.zscore.bw',
                "0_5_deltaTPM_log2Zscore_AroundEnh_pybbi.%s.pdf" % (mode),
                resolution=2000,
                fill=True, mode=mode,
                window=20000, nbins=21,
                how_far_from_edge=10,
                ymin=-1.0, ymax=1.0,
                vmin=-1.0, vmax=1,
                bed_list=["results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed", 'results/V_specific_enhancers.bed', 'results/S_specific_enhancers.bed', 'results/VS_common_enhancers.bed'])
    # plot_around_anchors('bw/V_enhancers_cov.w500.zscore.bw',
    #                     "0_enhCovAroundGenes_pybbi.%s.pdf" % (mode),
    #                     resolution=500,
    #                     fill=True, mode=mode,
    #                     window=5000, nbins=21,
    #                     how_far_from_edge=10,
    #                     ymin=-0.1, ymax=1.0,
    #                     vmin=-0, vmax=1,
    #                     bed_list=['data/genome/genes.plus.bed3', 'data/genome/genes.minus.bed3'])
                        
                        
                        

# %% find HE-genes inside loops
# create bed without borders to find HE-genes

rnaseq_insideLoop = bioframe.overlap(df_loops_bed_withEnh, rnaseq_binned, how='left', return_index=True, keep_order=True)  # return_input=False, return_overlap=True,

# select bin with maximum TPM, one per loop
rnaseq_insideLoop_max = []
for i in range(df_loops_bed_withEnh.shape[0]):  # rnaseq_insideLoop['index'].unique():
    values = rnaseq_insideLoop.loc[rnaseq_insideLoop['index'] == i]
    rnaseq_insideLoop_max.append(values.iloc[np.argmax(values['value_'])].tolist())

rnaseq_insideLoop_max_df = pd.DataFrame(rnaseq_insideLoop_max,
                                        columns=rnaseq_insideLoop.columns.tolist())
# select bin with minimum TPM, one per loop
rnaseq_insideLoop_min = []
for i in range(df_loops_bed_withEnh.shape[0]):  # rnaseq_insideLoop['index'].unique():
    values = rnaseq_insideLoop.loc[rnaseq_insideLoop['index'] == i]
    rnaseq_insideLoop_min.append(values.iloc[np.argmin(values['value_'])].tolist())

rnaseq_insideLoop_min_df = pd.DataFrame(rnaseq_insideLoop_min,
                                        columns=rnaseq_insideLoop.columns.tolist())

# %% select anchors with enhancers saving indexes (for merging)
df = df_loops_bedpe_withEnh.iloc[:, 0:3]
df.columns = df_loops.columns.tolist()
df_anchorsWithEnh = bioframe.overlap(df, df_enh, how='left', return_index=True, keep_order=True)

df_anchorsWithEnh_indexed = []
df_anchorsWithEnh_indexed.append(df_anchorsWithEnh.iloc[:, 0:4])

df = df_loops_bedpe_withEnh.iloc[:, 3:6]
df.columns = df_loops.columns.tolist()
df_anchorsWithEnh = bioframe.overlap(df, df_enh, how='left', return_index=True, keep_order=True)
df_anchorsWithEnh_indexed.append(df_anchorsWithEnh.iloc[:, 0:4])

df_anchorsWithEnh_indexed = pd.DataFrame(np.concatenate(df_anchorsWithEnh_indexed))
df_anchorsWithEnh_indexed.columns = ['index'] + df_loops.columns.tolist()
df_anchorsWithEnh_indexed = df_anchorsWithEnh_indexed.set_index('index')
df_anchorsWithEnh_indexed.head()

# %% select top-10%
q09 = np.quantile(rnaseq_insideLoop_max_df['value_'], q=0.01)
bins_with_q09_TPM = rnaseq_insideLoop_max_df.loc[rnaseq_insideLoop_max_df['value_'] >= q09]
bins_with_q09_TPM = bins_with_q09_TPM.set_index('index')

q01 = np.quantile(rnaseq_insideLoop_min_df['value_'], q=0.99)
bins_with_q01_TPM = rnaseq_insideLoop_max_df.loc[rnaseq_insideLoop_max_df['value_'] <= q01]
bins_with_q01_TPM = bins_with_q01_TPM.set_index('index')

# %% function for av plot from bedpe



# %% build av loop
def create_bedpe_with_expr(bins_with_expr = bins_with_q09_TPM):
    tmp = df_anchorsWithEnh_indexed.join(bins_with_expr, lsuffix='_anchors', rsuffix='_rnaseq')
    tmp = tmp.dropna()
    # tmp['length'] = tmp['start2'] - tmp['end1']
    # sns.boxplot(x=tmp["length"] / 500)
    tmpcheck = tmp.loc[tmp['start_anchors'] >= tmp['start_']]

    tmp = tmp.iloc[:, [0, 1, 2, 7, 8, 9]]
    tmp.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
    tmp['start1'] = tmp['start1'].astype(np.int64)
    tmp['end1'] = tmp['end1'].astype(np.int64)
    tmp['start2'] = tmp['start2'].astype(np.int64)
    tmp['end2'] = tmp['end2'].astype(np.int64)

    tmp_flip = tmp.loc[tmp['start1'] > tmp['start2']]
    tmp_flip = tmp.loc[tmp['start1'] - tmp['end2'] > 6000]
    tmp_flip = tmp_flip[['chrom2', 'start2', 'end2', 'chrom1', 'start1', 'end1']]
    tmp_flip.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

    tmp = tmp.loc[tmp['start1'] < tmp['start2']]
    tmp = tmp.loc[tmp['start2'] - tmp['end1'] > 6000]
    return(tmp, tmp_flip)

# %% load chromsizes
df_chromsizes = pd.read_csv('data/genome/dicty.chrom.sizes', sep='\t', header=None, )
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes[
    "end"].astype(str)
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

# supports = df_chromsizes[['chrom', 'start', 'end']].values

# %% load clr
cooler_dir = "../dicty_hic_distiller/subsampled"

resolution = 2000
pad = 2
q = 0.2
nthreads = 16
flank = 20000

Timing = 0
clr = cooler.Cooler(os.path.join(cooler_dir, "0AB.filtered.mcool::/resolutions/500"))

expected = cooltools.expected_cis(clr, view_df=df_chromsizes,
                                  nproc=2, chunksize=1_000_000)

# %% prepare
tmp1, tmp2 = create_bedpe_with_expr(bins_with_q09_TPM)
tmp3, tmp4 = create_bedpe_with_expr(bins_with_q01_TPM)
# tmp['start_'] = pd.to_numeric(tmp['start_'])

# bins_with_q09_TPM_bed = bins_with_q09_TPM.iloc[:,[1,2,3]]
# df_loopsWithEnh_bed = df_loopsWithEnh.loc[df_loopsWithEnh['count'] >= 1].iloc[:,0:3]
# print(rnaseq_insideLoop_max_df.shape)
# print(df_loopsWithEnh_bed.shape)
# plot_avLoop(tmp)

#%%
stack1 = cooltools.pileup(clr, tmp3, view_df=df_chromsizes,
                         expected_df=expected, flank=100_000)
stack2 = cooltools.pileup(clr, tmp4, view_df=df_chromsizes,
                         expected_df=expected, flank=100_000)
tmp0 = np.transpose(np.rot90(np.rot90(stack2, axes=(1, 0)), axes=(1, 0)), axes=(1, 0, 2))
stack_rotated = np.dstack((stack1, tmp0)) #[stack1, tmp])
# mtx1 = np.nanmean(stack1, axis=2)
# mtx2 = np.nanmean(stack2, axis=2)
# tmp = np.array([mtx1, mtx2])
mtx = np.nanmean(stack_rotated, axis=2)
# mtx = np.nanmean(stack, axis=2)

flank = 250

# av pixels
av_pixels = np.nanmean(np.nanmean(stack_rotated[200:201,200:210, :], axis=2))
plt.annotate(str(np.round(np.log2(av_pixels), 3)),xy=(20,20))

plt.imshow(
    np.log2(mtx),
    vmax = 1,
    vmin = -1,
    cmap='coolwarm')

plt.colorbar(label = 'log2 mean obs/exp')
ticks_pixels = np.linspace(0, flank*2//resolution,5)
ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)
# plt.xticks(ticks_pixels, ticks_kbp)
# plt.yticks(ticks_pixels, ticks_kbp)
plt.xlabel('relative position, kbp')
plt.ylabel('relative position, kbp')
plt.axvline(200, linewidth=0.3, ls='--')
plt.axhline(y=200, linestyle='--', linewidth=0.3, color=(0, 0, 0, 0.75)) #color='r',
#plt.vlines(x=50, ls='--', ymin = 0, ymax = 100)


plt.show()
#%% plot loops (to check)
bed_file = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe"
df_bed = pd.read_table(bed_file, header=None)
df_bed.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

stack = cooltools.pileup(clr, df_bed, view_df=df_chromsizes,
                         expected_df=expected, flank=100_000)

mtx = np.nanmean(stack, axis=2)

flank = 250

plt.imshow(
    np.log2(mtx),
    vmax = 1,
    vmin = -1,
    cmap='coolwarm')

plt.colorbar(label = 'log2 mean obs/exp')
ticks_pixels = np.linspace(0, flank*2//resolution,5)
ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)
# plt.xticks(ticks_pixels, ticks_kbp)
# plt.yticks(ticks_pixels, ticks_kbp)
plt.xlabel('relative position, kbp')
plt.ylabel('relative position, kbp')
plt.axvline(50, linewidth=0.3, ls='--')
plt.axhline(y=50, linestyle='--', linewidth=0.3, color=(0, 0, 0, 0.75)) #color='r',
#plt.vlines(x=50, ls='--', ymin = 0, ymax = 100)

plt.show()

# %% use coolpuppy to create pics for enhancer
flank = 50000
V_enh = bioframe.read_table('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed')#.query('score == "none"')
df_anchors = load_BedInMode('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed', mode='anchors_3bins')
EnhWithAnchors_ind = bioframe.overlap(V_enh, df_anchors, how='inner', return_input=False, return_index=True)['index'].tolist()
EnhWithAnchors_indUniq = np.unique(EnhWithAnchors_ind)
V_enh['loop_anchor'] = 'none'
V_enh.loc[:,'loop_anchor'][EnhWithAnchors_indUniq] = 'loop_anchor'

pup = coolpup.pileup(clr, V_enh, groupby=['loop_anchor1'], #['group1'], #'length_group2'
rescale=True, rescale_size=int(1+flank * 2// resolution),  rescale_flank=10,
local=True,
features_format='bed', view_df=df_chromsizes, nproc=4, 
#by_strand=True, 
flip_negative_strand=True, 
expected_df=expected, 
flank=flank, min_diag=2)

# %%
plotpup.plot(pup,
             cols='loop_anchor1',
             #rows='orientation',
             #cols='name1', col_order=['both low', 'both high', 'high & low'], #col_order=['low', 'average', 'high'],
             #rows='name2', row_order=['both low', 'both high', 'high & low'],
             row_order=['none', 'loop anchor'],
             #rows='length_group2', row_order=['short', 'average', 'long'],
             #cols='orientation', #col_order=['--', '++'],
             score=False, 
             cmap='coolwarm', #'viridis', #'coolwarm', #
             scale='log',
             #sym=True,
             #vmax=1.2, vmin=0.5,
             vmax=1.1, vmin=0.6,
             #vmin=0.0015, vmax=0.025,
             #vmin=0.005, vmax=0.025,
             height=5)
plt.savefig('results/pics/paper/av_enhancer.flank50k.byAnchor.pdf', dpi=100, bbox_inches='tight')
plt.show()

# %% count enhancer reads
# V_enhancer_noPromoter_bed.each(featurefuncs.bed2gff).saveas('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.gff')
# M_enhancer_noPromoter_bed.each(featurefuncs.bed2gff).saveas('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.gff')
# awk -v OFS='\t' '{ print $1, $2, "enhancer", $4, $5, $6,$7, $8, $9 }' results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.gff > V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.gff

# %% calculate enhancer score
Vatac_merged = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_r123.mergeIDR.narrowPeak', schema='bed')
V_enhancers = bioframe.read_table('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')

V_atac_r1 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_R1.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r2 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_R2.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r3 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Vegetative_R3.mLb.clN_peaks.narrowPeak', schema='narrowPeak')

tmp1 = bioframe.overlap(V_enhancers, V_atac_r1, how='left', suffixes=['', '_r1'])
tmp2 = bioframe.overlap(tmp1, V_atac_r2, how='left', suffixes=['', '_r2'])
Vatac_merged_withSignal = bioframe.overlap(tmp2, V_atac_r3, how='left', suffixes=['', '_r3'])

Vatac_merged_withSignal = Vatac_merged_withSignal.loc[:, ['chrom', 'start', 'end', 'fc_r1', 'fc_r2', 'fc_r3']].groupby(['chrom', 'start', 'end']).apply(lambda x : np.nanmean(x))
Vatac_merged_withSignal = Vatac_merged_withSignal.reset_index()
# Vatac_merged_withSignal['start'] = Vatac_merged_withSignal['start'].astype(np.int64)
# Vatac_merged_withSignal['end'] = Vatac_merged_withSignal['end'].astype(np.int64)
Vatac_merged_withSignal.columns = ['chrom', 'start', 'end', 'FC_atac']

# add chipseq
V_chipseq_df = bioframe.read_table('results/V_idrMergeIdr.chipseqUnion.bed', schema='bed3')

df_enh_score = bioframe.overlap(Vatac_merged_withSignal, V_chipseq_df, how='inner', suffixes=['', '_chipseq'])
df_enh_score['length_chipseq'] = (df_enh_score['end_chipseq'] - df_enh_score['start_chipseq'])/100
df_enh_score['enh_score'] = df_enh_score.length_chipseq * df_enh_score.FC_atac
df_Venh_score = df_enh_score.sort_values(by=['chrom', 'start'])

Vspec_df = bioframe.read_table('results/V_specific_enhancers.bed', schema='bed3')
df_enh_score_Vspec = bioframe.overlap(df_enh_score, Vspec_df, how='outer')
# %%
sns.histplot(data=df_enh_score, x="enh_score", kde=True)
plt.savefig("results/pics/paper/enh_score.pdf", dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
#%%
tpm_df = create_rnaseq_df(need_coord=True)
for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01)
tmp = bioframe.overlap(tpm_df, df_enh_score_Vspec, how='left')
tmp.loc[tmp['enh_score_'].isnull(), 'enh_score_'] = 0
g = sns.FacetGrid(tmp, height=15)
g.map(sns.scatterplot, 'log2_0_TPM','enh_score_', alpha=0.5, palette="viridis")
# plt.savefig("results/pics/paper/enh_score_vs0TPM.pdf", dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
# %%
g = sns.FacetGrid(tmp, height=15, palette = 'seismic')
g.map(sns.scatterplot, 'log2_0_TPM','log2_8_TPM', alpha=0.5, palette="viridis")
# plt.savefig("results/pics/paper/enh_score_vs0TPM.pdf", dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
# %%
tmp.loc[tmp['enh_score_'] > 100, 'enh_score_'] = 100
tmp = tmp.query('enh_score_ > 80')
sns.scatterplot(data=tmp, x='log2_0_TPM',y='log2_8_TPM', hue='enh_score_', alpha=0.5)
# plt.savefig("results/pics/paper/0TPMvs8TPM.hueEnhScore.pdf", dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
# %% S-specific
V_enhancers = bioframe.read_table('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')

V_atac_r1 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Streaming_R1.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r2 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Streaming_R2.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r3 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Streaming_R3.mLb.clN_peaks.narrowPeak', schema='narrowPeak')

tmp1 = bioframe.overlap(V_enhancers, V_atac_r1, how='left', suffixes=['', '_r1'])
tmp2 = bioframe.overlap(tmp1, V_atac_r2, how='left', suffixes=['', '_r2'])
Vatac_merged_withSignal = bioframe.overlap(tmp2, V_atac_r3, how='left', suffixes=['', '_r3'])

Vatac_merged_withSignal = Vatac_merged_withSignal.loc[:, ['chrom', 'start', 'end', 'fc_r1', 'fc_r2', 'fc_r3']].groupby(['chrom', 'start', 'end']).apply(lambda x : np.nanmean(x))
Vatac_merged_withSignal = Vatac_merged_withSignal.reset_index()
# Vatac_merged_withSignal['start'] = Vatac_merged_withSignal['start'].astype(np.int64)
# Vatac_merged_withSignal['end'] = Vatac_merged_withSignal['end'].astype(np.int64)
Vatac_merged_withSignal.columns = ['chrom', 'start', 'end', 'FC_atac']

# add chipseq
V_chipseq_df = bioframe.read_table('results/S_idrMergeIdr.chipseqUnion.bed', schema='bed3')

df_enh_score = bioframe.overlap(Vatac_merged_withSignal, V_chipseq_df, how='inner', suffixes=['', '_chipseq'])
df_enh_score['length_chipseq'] = (df_enh_score['end_chipseq'] - df_enh_score['start_chipseq'])/100
df_enh_score['enh_score'] = df_enh_score.length_chipseq * df_enh_score.FC_atac
df_Senh_score = df_enh_score.sort_values(by=['chrom', 'start'])

# Vspec_df = bioframe.read_table('results/V_specific_enhancers.bed', schema='bed3')
# df_enh_score_Vspec = bioframe.overlap(df_enh_score, Vspec_df, how='outer')

# %% M-specific
V_enhancers = bioframe.read_table('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')

V_atac_r1 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_R1.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r2 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_R2.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r3 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_R3.mLb.clN_peaks.narrowPeak', schema='narrowPeak')
V_atac_r4 = bioframe.read_table('results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/Mound_R4.mLb.clN_peaks.narrowPeak', schema='narrowPeak')

tmp1 = bioframe.overlap(V_enhancers, V_atac_r1, how='left', suffixes=['', '_r1'])
tmp2 = bioframe.overlap(tmp1, V_atac_r2, how='left', suffixes=['', '_r2'])
tmp3 = bioframe.overlap(tmp2, V_atac_r3, how='left', suffixes=['', '_r3'])
Vatac_merged_withSignal = bioframe.overlap(tmp3, V_atac_r4, how='left', suffixes=['', '_r4'])

Vatac_merged_withSignal = Vatac_merged_withSignal.loc[:, ['chrom', 'start', 'end', 'fc_r1', 'fc_r2', 'fc_r3', 'fc_r4']].groupby(['chrom', 'start', 'end']).apply(lambda x : np.nanmean(x))
Vatac_merged_withSignal = Vatac_merged_withSignal.reset_index()
# Vatac_merged_withSignal['start'] = Vatac_merged_withSignal['start'].astype(np.int64)
# Vatac_merged_withSignal['end'] = Vatac_merged_withSignal['end'].astype(np.int64)
Vatac_merged_withSignal.columns = ['chrom', 'start', 'end', 'FC_atac']

# add chipseq
V_chipseq_df = bioframe.read_table('results/M_idrMergeIdr.chipseqUnion.bed', schema='bed3')

df_enh_score = bioframe.overlap(Vatac_merged_withSignal, V_chipseq_df, how='inner', suffixes=['', '_chipseq'])
df_enh_score['length_chipseq'] = (df_enh_score['end_chipseq'] - df_enh_score['start_chipseq'])/100
df_enh_score['enh_score'] = df_enh_score.length_chipseq * df_enh_score.FC_atac
df_Menh_score = df_enh_score.sort_values(by=['chrom', 'start'])

# %%
df_enh = bioframe.overlap(df_Venh_score, df_Menh_score, how='outer', suffixes=['_Venh', '_Senh'])
df_enh.loc[df_enh['enh_score_Venh'].isnull(), 'enh_score_Venh'] = 0
df_enh.loc[df_enh['enh_score_Senh'].isnull(), 'enh_score_Senh'] = 0
df_enh.loc[df_enh['length_chipseq_Venh'].isnull(), 'length_chipseq_Venh'] = 0
df_enh.loc[df_enh['length_chipseq_Senh'].isnull(), 'length_chipseq_Senh'] = 0
df_enh.loc[df_enh['FC_atac_Venh'].isnull(), 'FC_atac_Venh'] = 0
df_enh.loc[df_enh['FC_atac_Senh'].isnull(), 'FC_atac_Senh'] = 0
df_enh['delta_atac'] = df_enh.FC_atac_Senh - df_enh.FC_atac_Venh
df_enh['delta_chipseq'] = df_enh.length_chipseq_Senh - df_enh.length_chipseq_Venh

sns.set(style='white', context='poster', rc={'figure.figsize':(10,10)})
g = sns.FacetGrid(df_enh, height=15)
g.map(sns.scatterplot, 'delta_atac','delta_chipseq', alpha=0.25)
g.add_legend()
plt.axvline(0, linewidth=0.8, ls='--')
plt.axhline(y=0, linestyle='--', linewidth=0.8, color=(0, 0, 0, 0.75))
plt.ylim(-25, 25) #5
plt.xlim(-7.5, 6) #2



g.map(const_line)
# sns.scatterplot(data=df_enh, x='length_chipseq_Venh',y='length_chipseq_Senh', alpha=0.5) #hue='enh_score_', 
# plt.savefig("results/pics/paper/0TPMvs8TPM.hueEnhScore.pdf", dpi=100, bbox_inches='tight')
plt.ylim(-1, 20)
plt.xlim(-1, 20)
plt.show()
plt.clf()

# %%
df_enh['group'] = 'no'
thr_atac = 2
thr_chip = 5
df_enh.loc[(df_enh['delta_chipseq'] < -5) &
                (df_enh['delta_atac'] < -2), 'group'] = 'low_left'
# df_enh.loc[(df_enh['FC_atac_Venh'] > threshold) &
#                 (df_enh['length_chipseq_Venh'] < threshold), 'group'] = 'low_right'
# df_enh.loc[(df_enh['FC_atac_Venh'] < threshold) &
#                 (df_enh['length_chipseq_Venh'] > threshold),'group'] = 'upper_left'
df_enh.loc[(df_enh['delta_chipseq'] > 5) &
                (df_enh['delta_atac'] > 2), 'group'] = 'upper_right'
df_enh['chrom'] = df_enh.chrom_Venh
df_enh.loc[df_enh['chrom'].isnull(), 'chrom'] = df_enh.chrom_Senh
for col in ['start_Senh', 'start_Venh', 'end_Senh', 'end_Venh']:
    df_enh.loc[df_enh[col].isnull(), col] = np.nan
df_enh = df_enh.fillna(np.nan)

df_enh['start'] = df_enh.start_Venh
df_enh.loc[df_enh['start'].isnull(), 'start'] = df_enh.start_Senh
df_enh['end'] = df_enh.end_Venh
df_enh.loc[df_enh['end'].isnull(), 'end'] = df_enh.end_Senh

# df_enh['start'] = df_enh[['start_Senh', 'start_Venh']].apply(lambda x : np.nanmin(x))
# df_enh['end'] = df_enh[['end_Senh', 'end_Venh']].apply(lambda x : np.nanmax(x))

df_enh_Vspec_tpm = bioframe.overlap(tpm_df, df_enh, how='inner')
df_enh_Vspec_tpm['delta_8_0'] = df_enh_Vspec_tpm.log2_8_TPM - df_enh_Vspec_tpm.log2_0_TPM
g = sns.violinplot(df_enh_Vspec_tpm, y='delta_8_0', x='group_')
# g.refline(x = 0,
#           color = "blue",
#           lw = 3)
g.axhline(0)

# %%
df_enh_Vspec = df_enh.query('enh_score_Venh > 0')
df_enh_Vspec_zscore = stats.zscore(df_enh_Vspec[['FC_atac_Venh','length_chipseq_Venh']], axis=0, ddof=1)
df_enh_Vspec_zscore[df_enh_Vspec_zscore > 3] = 3
df_enh_Vspec_zscore[df_enh_Vspec_zscore < -3] = -3
# sns.boxplot(df_enh_Vspec, y='FC_atac_Venh')
sns.set(style='white', context='poster', rc={'figure.figsize':(10,10)})
g = sns.FacetGrid(df_enh_Vspec_zscore, height=15)
g.map(sns.scatterplot, 'FC_atac_Venh','length_chipseq_Venh', alpha=0.2)
g.add_legend()
g.map(const_line)

# %%
df_enh_Vspec.columns = ['chrom', 'start', 'end', 'FC_atac_Venh',
       'chrom_chipseq_Venh', 'start_chipseq_Venh', 'end_chipseq_Venh',
       'length_chipseq_Venh', 'enh_score_Venh', 'chrom_Senh', 'start_Senh',
       'end_Senh', 'FC_atac_Senh', 'chrom_chipseq_Senh', 'start_chipseq_Senh',
       'end_chipseq_Senh', 'length_chipseq_Senh', 'enh_score_Senh', 'group']
threshold = 0
df_enh_Vspec['group'] = 'no'
df_enh_Vspec.loc[(df_enh_Vspec_zscore['FC_atac_Venh'] < threshold) &
                (df_enh_Vspec_zscore['length_chipseq_Venh'] < threshold), 'group'] = 'low_left'
df_enh_Vspec.loc[(df_enh_Vspec_zscore['FC_atac_Venh'] > threshold) &
                (df_enh_Vspec_zscore['length_chipseq_Venh'] < threshold), 'group'] = 'low_right'
df_enh_Vspec.loc[(df_enh_Vspec_zscore['FC_atac_Venh'] < threshold) &
                (df_enh_Vspec_zscore['length_chipseq_Venh'] > threshold),'group'] = 'upper_left'
df_enh_Vspec.loc[(df_enh_Vspec_zscore['FC_atac_Venh'] > threshold) &
                (df_enh_Vspec_zscore['length_chipseq_Venh'] > threshold), 'group'] = 'upper_right'

df_enh_Vspec_tpm = bioframe.overlap(tpm_df, df_enh_Vspec, how='inner')
sns.boxplot(df_enh_Vspec_tpm, y='log2_0_TPM', x='group_')

# %%
df_enh_toMerge = bioframe.overlap(df_Senh_score, df_Venh_score, how='outer', suffixes=['_Senh', '_Venh'])
df_enh_toMerge.loc[df_enh_toMerge['enh_score_Venh'].isnull(), 'enh_score_Venh'] = 0
df_enh_toMerge.loc[df_enh_toMerge['enh_score_Senh'].isnull(), 'enh_score_Senh'] = 0
df_enh_toMerge.columns = ['chrom_Senh', 'start_Senh', 'end_Senh', 'FC_atac_Senh',
       'chrom_chipseq_Senh', 'start_chipseq_Senh', 'end_chipseq_Senh',
       'length_chipseq_Senh', 'enh_score_Senh', 'chrom', 'start',
       'end', 'FC_atac_Venh', 'chrom_chipseq_Venh', 'start_chipseq_Venh',
       'end_chipseq_Venh', 'length_chipseq_Venh', 'enh_score_Venh']
tpm_df = create_rnaseq_df(need_coord=True)
for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01)
df_enh_withGenes = bioframe.overlap(df_enh_toMerge, tpm_df, how='inner', suffixes=['', '_genes'])
enhScore_thr = np.quantile(df_enh_withGenes.loc[df_enh_withGenes['enh_score_Senh'] == 0, 'enh_score_Venh'].tolist(), 0.75)

df_enh_withGenes = df_enh_withGenes.query('enh_score_Venh > @enhScore_thr')
df_enh_withGenes['delta_8_0'] = df_enh_withGenes.log2_8_TPM_genes - df_enh_withGenes.log2_0_TPM_genes
g = sns.boxplot(data=df_enh_withGenes, y='delta_8_0',)
plt.show()

# %%
# M_H3K27ac_merged.mLb.clN.bw
# V_atacseq.zscore
mode = 'median'
from custom_functions import plot_around_loop
for state in ['V']:  # , 'V', 'S'
    plot_around_loop("bw/%s_atacseq.zscore.bw"% state, #'bw/wang_rnaseq_bw/Fruiting_R2.TPM_log2.bw',#'bw/0_8_deltaTPM_log2.bw',
                     "%s_atacseq.zscore_around_enhancers_cov.%s.pdf" % (state, mode),
                     fill=True, mode=mode,
                     window=20000, nbins=21,
                     how_far_from_edge=10,
                     ymin=-0.5, ymax=3,
                     vmin=-3, vmax=3,
                     bed_list=['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed'])

# %%
# M_H3K27ac_merged.mLb.clN.bw
# V_atacseq.zscore
mode = 'median'
from custom_functions import plot_around_loop
for state in ['S']:  # , 'V', 'S'
    plot_around_loop('bw/bigwig/%s_H3K4me1_merged.mLb.clN.bw' % state, #"results/atac_seq_wang_2023/bwa/merged_replicate/bigwig/Mound.mRp.clN.bigWig", #'bw/wang_rnaseq_bw/Fruiting_R2.TPM_log2.bw',#'bw/0_8_deltaTPM_log2.bw',
                     "%s_H3K4me1_around_enhancers.%s.pdf" % (state, mode),
                     fill=True, mode=mode,
                     window=20000, nbins=21,
                     how_far_from_edge=10,
                     ymin=00, ymax=100,
                     vmin=10, vmax=100,
                     bed_list=['results/V_specific_enhancers.bed', 'results/M_specific_enhancers.bed', 'results/VM_common_enhancers.bed'])

# %% create matrix of enhancer signal
V_enhancers = bioframe.read_table('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
S_enhancers = bioframe.read_table('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
M_enhancers = bioframe.read_table('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
df_enhancers = pd.concat([V_enhancers, S_enhancers, M_enhancers]) #
df_EnhMerged = bioframe.merge(df_enhancers, min_dist=0).sort_values(['chrom', 'start'])
df_EnhMerged.to_csv('results/enhancers_merged.bed', sep="\t", index=False,header=False)

peak_list = []
# atac
# results/atac_seq_wang_2023/bwa/merged_replicate/macs2/narrow_peak/%s.mRp.clN_peaks.narrowPeak
# h3k27ac
# results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_H3K27ac_merged_peaks.narrowPeak
# bigwig_list

signal_dic = {}
for stage in ['Vegetative','Mound']: # 'Streaming', 
    for assay in ['H3K4me1', 'H3K27ac', 'atac_seq']:
        signal_list = []
        if assay == 'atac_seq':
            bw_url = 'results/atac_seq_wang_2023/bwa/merged_replicate/bigwig/%s.mRp.clN.bigWig' % stage
        else:
            bw_url = './bw/bigwig/%s_%s_merged.mLb.clN.bw' % (stage[0], assay)
# 
        for i in range(len(df_EnhMerged)):
            signal_value = bioframe.read_bigwig(bw_url, df_EnhMerged.loc[i, "chrom"], start=df_EnhMerged.loc[i, "start"], end=df_EnhMerged.loc[i, "end"]).value.median()
            signal_list.append(signal_value)
        signal_dic['%s_%s' % (stage[0], assay)] = signal_list

# %%
from scipy import stats
from scipy.spatial import distance
from scipy.cluster import hierarchy
 #, 
signal_df = pd.DataFrame(signal_dic)
signal_df = signal_df[['V_H3K4me1', 'V_H3K27ac', 'V_atac_seq', 'M_H3K4me1', 'M_H3K27ac', 'M_atac_seq']]
# correlations = signal_df.corr()
# correlations_array = np.asarray(df.corr())

# row_linkage = hierarchy.linkage(
#     distance.pdist(correlations_array), method='average')
signalZscore_df = stats.zscore(signal_df, axis=0, ddof=1)
signalZscore_df[signalZscore_df > 3] = 3
signalZscore_df[signalZscore_df < -3] = -3
signalZscore_df['M'] = signalZscore_df[['M_H3K4me1', 'M_H3K27ac', 'M_atac_seq']].min(axis=1)
signalZscore_df['V'] = signalZscore_df[['V_H3K4me1', 'V_H3K27ac', 'V_atac_seq']].min(axis=1)
# sns.boxplot(data=tmp_z,y='M_atac_seq')
g=sns.clustermap(signalZscore_df[['V', 'M']], 
    #row_linkage=row_linkage, 
    figsize=(10, 6),
    # z_score=1,
    vmin=-3, vmax=3,
    cmap="coolwarm",
    # method='ward',
    metric='euclidean', #'correlation',
    col_cluster=False,
    dendrogram_ratio=(.3, .3),
    cbar_pos=(0, .2, .03, .4))
plt.savefig("results/pics/paper/enh_cluster.VMmeans.pdf", dpi=100, bbox_inches='tight')
# %%
from scipy.spatial import distance_matrix
tmp = np.zeros((1737, 868, 1))
from sklearn.cluster import KMeans
rld_mat_cor = signalZscore_df.T.corr()
rld_mat_cor_inv = 1 - rld_mat_cor
# rld_mat_dist = distance_matrix(1 - rld_mat_cor)
#clusters <- kmeans(rld_mat_dist, 6)
model = KMeans(n_clusters=4, random_state=42) 
signal_df_cl = model.fit(signalZscore_df[['V', 'M']]) #[['V', 'M']]
# %%
lut = dict(zip(np.unique(signal_df_cl.labels_.tolist()), "rbg"))
row_colors = pd.Series(signal_df_cl.labels_.tolist()).map(lut)
g=sns.clustermap(signal_df, 
    #row_linkage=row_linkage, 
    row_colors=row_colors,
    figsize=(10, 6),
    z_score=1,
    vmin=-3, vmax=3,
    cmap="coolwarm",
    # method='ward',
    metric='euclidean', #'correlation',
    col_cluster=False,
    dendrogram_ratio=(.3, .3),
    cbar_pos=(0, .2, .03, .4))
# %%
signalZscore_df['cl'] = signal_df_cl.labels_.tolist()
# signalZscore_df_test = signalZscore_df.query('cl == 0')
sns.heatmap(signalZscore_df.reindex(signalZscore_df.cl.sort_values().index, axis=0), cmap="coolwarm", vmin=-3, vmax=3,)
plt.savefig("results/pics/paper/enh_KmeansCluster4.VMmin.pdf", dpi=100, bbox_inches='tight')
# [['V', 'M', 'cl']]
#.reindex(signal_df.mean().sort_values().index, axis=0)
#loc[:,['V_H3K4me1', 'V_H3K27ac', 'V_atac_seq',]].mean(axis=1)
# %%
grid = sns.FacetGrid(col='cl',
                    data=signalZscore_df.reindex(signalZscore_df.V_atac_seq.sort_values().index, axis=0),
                    col_wrap=10,)

grid.map_dataframe(sns.heatmap, cmap="coolwarm", vmin=-3, vmax=3,)
plt.savefig("results/pics/paper/enh_KmeansCluster4.VM.pdf", dpi=100, bbox_inches='tight')

# %%
from statannot import add_stat_annotation
df_EnhMerged['cl'] = signal_df_cl.labels_.tolist()
tpm_df = create_rnaseq_df(need_coord=True)

for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01)
tpm_df_Enh = bioframe.overlap(tpm_df, df_EnhMerged, how='left')
tpm_df_Enh['delta_8_0'] = tpm_df_Enh.log2_8_TPM - tpm_df_Enh.log2_0_TPM
tpm_df_Enh['group'] = 'with enhancer'
tpm_df_Enh.loc[tpm_df_Enh['cl_'].isnull(), 'group'] = 'w/o enhancer'
# tpm_df_Enh.loc[tpm_df_Enh['cl_'].isnull(), 'cl_'] = 0
# grid = sns.FacetGrid(#col='cl_',
                    # data=tpm_df_Enh,
                    #col_wrap=10,)

g = sns.boxplot(data=tpm_df_Enh, x='cl_', y='log2_0_TPM') #'group', hue
# grid.add_legend()
add_stat_annotation(g, data=tpm_df_Enh, x='cl_', y='log2_0_TPM', #order=order,
                    box_pairs=[('0.0', '1.0')],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.savefig("results/pics/paper/boxplot_KmeansCluster4.VM.pdf", dpi=100, bbox_inches='tight')

# %%
signal_df['delta_h3k27'] = signal_df.M_H3K27ac - signal_df.V_H3K27ac
signal_df['delta_atac'] = signal_df.M_atac_seq - signal_df.V_atac_seq
g = sns.scatterplot(data=signal_df, y='delta_atac', x='delta_h3k27', alpha=0.5, size=10) #hue='group', 
plt.legend([],[], frameon=False)
plt.show()

# %%
df_chrom = bioframe.read_chromsizes('data/genome/dicty.chrom.sizes')
df_bins = bioframe.binnify(chromsizes=df_chrom, binsize=500)

signal_dic = {}
for stage in ['Vegetative','Mound']: # 'Streaming', 
    for assay in ['H3K27ac']:
        signal_list = []
        if assay == 'atac_seq':
            bw_url = 'results/atac_seq_wang_2023/bwa/merged_replicate/bigwig/%s.mRp.clN.bigWig' % stage
        else:
            bw_url = './bw/bigwig/%s_%s_merged.mLb.clN.bw' % (stage[0], assay)
# 
        for i in range(10000):
            signal_value = bioframe.read_bigwig(bw_url, df_bins.loc[i, "chrom"], start=df_bins.loc[i, "start"], end=df_bins.loc[i, "end"]).value.median()
            signal_list.append(signal_value)
        signal_dic['%s_%s' % (stage[0], assay)] = signal_list
df_bins_signals = pd.DataFrame(signal_dic)

g = sns.scatterplot(data=df_bins_signals, y='V_H3K27ac', x='M_H3K27ac', alpha=0.5, size=10) #hue='group', 
plt.legend([],[], frameon=False)
plt.show()

# %% checnk dispersion fo enhancer genes
V_enhancers = bioframe.read_table('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
S_enhancers = bioframe.read_table('results/S_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
M_enhancers = bioframe.read_table('results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
df_enhancers = pd.concat([V_enhancers, S_enhancers, M_enhancers]) #
df_EnhMerged = bioframe.merge(df_enhancers, min_dist=0).sort_values(['chrom', 'start'])

tpm_df_rosengarten = create_rnaseq_df_rosengarten()

# %%
df_EnhMerged = bioframe.read_table('results/pseudoDifEnhacerswithAtac.bed', schema='bed')
df_EnhMerged['units'] = range(len(df_EnhMerged))
tpm_df_rosengarten['units'] = range(len(tpm_df_rosengarten))
tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['', '_enh'])
tpm_df_rosengartenEnh['group'] = 'with enhancer'
tpm_df_rosengartenEnh.loc[tpm_df_rosengartenEnh.units_enh.isnull(), 'units_enh'] = range(np.sum(tpm_df_rosengartenEnh.units_enh.isnull()))
tpm_df_rosengartenEnh.loc[tpm_df_rosengartenEnh.start_enh.isnull(), 'group'] = 'w/o enhancer'
tmp = tpm_df_rosengartenEnh[['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24',
       'units_enh', 'group']]
tpm_df_rosengartenEnh_melt = pd.melt(tmp, id_vars=['units_enh', 'group'], value_vars=['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24'])

# %%
sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(10,10)})

# Create an array with the colors you want to use
colors = ["#B2ADAE", "#8e042d"]# Set your custom color palette
cherrry_and_ash = sns.set_palette(sns.color_palette(colors))

tpm_df_rosengartenEnh_melt['log2_TPM'] = np.log2(tpm_df_rosengartenEnh_melt.value + 0.01)
g = sns.lineplot(
    data=tpm_df_rosengartenEnh_melt,
    x="variable", y="log2_TPM", hue="group", #units="units_enh",
    estimator='mean', lw=1, errorbar='se', 
    alpha=0.5, 
    #size=5,
    palette=cherrry_and_ash
)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
# wrap_labels(g, 10)
# plt.savefig("results/pics/paper/lineplot_enh_DifEnh.pdf", dpi=100, bbox_inches='tight')
plt.legend([],[], frameon=False)
plt.show()
plt.clf()

# %% track of enhancers for Uni and Multi
tpm_df_rosengartenEnh['type'] = np.nan
for gene_types in ["Unicellular", "Multicellular"]:
    genes_orthology = bioframe.read_table('data/' + gene_types + 'ity_genes.bed3', schema='bed3')
    geneWithType_ind = bioframe.overlap(tpm_df_rosengartenEnh, genes_orthology, how='inner', return_input=False, return_index=True)['index'].tolist()
    geneWithType_indUniq = np.unique(geneWithType_ind)
    
    tpm_df_rosengartenEnh.loc[:,'type'][geneWithType_indUniq] = gene_types

tmp = tpm_df_rosengartenEnh[['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24', 'units_enh', 'group', 'type']]
tpm_df_rosengartenEnh_melt = pd.melt(tmp, id_vars=['units_enh', 'group', 'type'], value_vars=['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24'])

sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(10,10)})

# Create an array with the colors you want to use
colors = ["#B2ADAE", "#8e042d"]# Set your custom color palette
cherrry_and_ash = sns.set_palette(sns.color_palette(colors))

tpm_df_rosengartenEnh_melt['log2_TPM'] = np.log2(tpm_df_rosengartenEnh_melt.value + 0.01)
g = sns.lineplot(
    data=tpm_df_rosengartenEnh_melt,
    x="variable", y="log2_TPM", hue="group", #units="units_enh",
    estimator='mean', lw=1, errorbar='se',
    style="type",
    alpha=0.3, 
    #size=5,
    palette=cherrry_and_ash
)
plt.savefig("results/pics/paper/lineplot_enh_UniMulti.pdf", dpi=100, bbox_inches='tight')
plt.legend([],[], frameon=False)
plt.show()
plt.clf()

# %%
tpm_df_rosengartenEnh['type'] = np.nan

genes_orthology = bioframe.read_table('data/dicty_HK.bed', schema='bed3')
geneWithType_ind = bioframe.overlap(tpm_df_rosengartenEnh, genes_orthology, how='inner', return_input=False, return_index=True)['index'].tolist()
geneWithType_indUniq = np.unique(geneWithType_ind)

tpm_df_rosengartenEnh.loc[:,'type'][geneWithType_indUniq] = 'HK'

tmp = tpm_df_rosengartenEnh[['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24', 'units_enh', 'group', 'type']]
tpm_df_rosengartenEnh_melt = pd.melt(tmp, id_vars=['units_enh', 'group', 'type'], value_vars=['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24'])

sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(10,10)})

# Create an array with the colors you want to use
colors = ["#B2ADAE", "#8e042d"]# Set your custom color palette
cherrry_and_ash = sns.set_palette(sns.color_palette(colors))

tpm_df_rosengartenEnh_melt['log2_TPM'] = np.log2(tpm_df_rosengartenEnh_melt.value + 0.01)
g = sns.lineplot(
    data=tpm_df_rosengartenEnh_melt,
    x="variable", y="log2_TPM", hue="group", #units="units_enh",
    estimator='mean', lw=1, errorbar='se',
    style="type",
    alpha=0.3, 
    #size=5,
    palette=cherrry_and_ash
)
plt.ylim(0, 7) #5
plt.savefig("results/pics/paper/lineplot_enh_HK.pdf", dpi=100, bbox_inches='tight')
plt.legend([],[], frameon=False)
plt.show()
plt.clf()

# %%
for mode in ['whole']:
    print(pValue_featureOccurenceInLoop(file_loops='results/%sAB_loops_thresBMA.bed',
                                        time='0', mode=mode, N_shuffle=1000,
                                        file_features='results/enhancers_merged.bed', 
                                        name='EnhancersIn_loopsBMA_mode_%s' % mode))

# %% check whether enhancer is the main player here
time = 0
tpm_df = create_rnaseq_df(need_coord=True)
for time_int in [0,2,5,8]:
    tpm_df['log2_%s_TPM' % (time_int)] = np.log2(tpm_df[str(time_int) + 'AB']+0.01)
tpm_df['enhancer'] = 'no'
enhancer_df = bioframe.read_table('results/enhancers_merged.bed', schema='bed3')
genes_withEnh_vec = bioframe.overlap(tpm_df, enhancer_df, how='inner').name.tolist()
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'enhancer'] = 'yes'

# loop anchor
# df_loops = bioframe.read_table('data/loops_quantifyChromosight/%sAB_chromosight_mergedAnchors.bed' % (time). schema='bed3')
df_loops = load_BedInMode('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed', time=None, mode='anchors_3bins', resolution=2000)
tpm_df['loop'] = 'outside'
genes_withEnh_vec = bioframe.overlap(tpm_df, df_loops, how='inner').name.tolist()
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'loop'] = 'anchors'

# inside
df_loops = load_BedInMode('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed', time=None, mode='inside-1bin', resolution=2000)
genes_withEnh_vec = bioframe.overlap(tpm_df, df_loops, how='inner').name.tolist()
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'loop'] = 'inside'

# ConvPairs
df_convPairs = bioframe.read_table('data/genome/genesInConvPairs.bed', schema='bed3')
tpm_df['ConvPairs'] = 'w/o pair'
genes_withEnh_vec = bioframe.overlap(tpm_df, df_convPairs, how='inner').name.tolist()
tpm_df.loc[tpm_df['name'].isin(genes_withEnh_vec), 'ConvPairs'] = 'ConvPair'


# %%
sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(20,20)})

import seaborn as sns, matplotlib.pyplot as plt
g = sns.FacetGrid(tpm_df, col='enhancer', row="loop") #,  height=5
g.map(sns.boxplot, "ConvPairs", 'log2_0_TPM',palette='muted')
# g.add_legend()
plt.savefig("results/pics/paper/expr_comparedToFeatures.pdf", dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
# sns.boxplot(tpm_df, x="ConvPairs", y='log2_0_TPM', hue='loop')
# %% endurers intersect with DE genes
counts_df = pd.read_table('data/all_gene_markdup_htseq', index_col=0, header=None).T
counts_df.index = ['0A', '0B','2A', '2B','5A', '5B','8A', '8B']

metadata = pd.DataFrame({'sample': ['0A', '0B','2A', '2B','5A', '5B','8A', '8B'], 'condition': ['0h', '0h', '2h', '2h', '5h', '5h', '8h', '8h']}).set_index('sample')

inference = DefaultInference(n_cpus=4)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design_factors="condition",
    refit_cooks=True,
    inference=inference,
)
dds.deseq2()

stat_res = DeseqStats(dds, inference=inference)
stat_res.summary()
stat_res.results_df.to_csv(os.path.join('results/DE/', "results.csv"))

stat_res_B_vs_A = DeseqStats(dds, contrast=["condition", "0h", "2h"], inference=inference)
tmp = stat_res.results_df.query('padj < 0.05').index.tolist()


upset_df = pd.DataFrame({'genes': counts_df.columns})
for t1, t2 in [[0,8]]:
    stat_res = DeseqStats(dds, contrast=["condition", "%sh" % t1, "%sh" % t2], inference=inference)
    stat_res.summary()
    tmp = stat_res.results_df.query('padj < 0.05 and (log2FoldChange < 0.5 or log2FoldChange > 1)').index.tolist()
    upset_df["%sh_%sh" % (t1, t2)] = 0
    upset_df.loc[upset_df['genes'].isin(tmp), "%sh_%sh" % (t1, t2)] = 1

# %%
# tpm_df.loc[tpm_df.name.isin(tmp),:].to_csv()
for mode in ['whole']:
        file_features = 'results/DE/genes_8h_0h.bed3'
        print(pValue_featureOccurenceInLoop(file_loops='results/%s_idrMergeIdr.chipseqUnion.bed',
                                            time='V',
                                            mode=mode, N_shuffle=1000,
                                            file_features=file_features, name='0h_enh_vs_DE'))

# %%
# bedtools coverage -a results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed -b results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K4me1_idr_peaks.narrowPeak results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/M_H3K27ac_idr_peaks.narrowPeak | awk '$5 ==0' > results/pseudoDifEnhacers.bed

# %% enhancer revisited Jan 15, 2024

for time in ['V', 'S', 'M' ]: #
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    V_h3k4me1 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]))
    V_h3k4me3 = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][3]))
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2]))

    V_enhancer_df = V_h3k27ac.intersect(V_atacseq, e=True, f=0.25, F=0.25).to_dataframe(names=range(0,20)).iloc[:,[0,1,2]]
    V_enhancer_df.columns = ['chrom', 'start', 'end']
    V_enhancer_df.to_csv('results/%s_idrMergeIdr.acetylInteresectATAC.bed' % (time), sep="\t", index=False,header=False)
    
    promoters = BedTool("data/genome/promoters.bed")
    genes = BedTool("data/genome/genes.bed")
    V_enhancer_noPromoter_bed = BedTool.from_dataframe(V_enhancer_df).intersect(promoters, v=True)
    V_enhancer_noPromoter_bed.saveas('results/%s_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time))

# %%
for time1, time2 in [['V', 'M'], ['V', 'S'], ['M', 'V'], ['S', 'M'], ['M', 'S'], ['V', 'S']]:
    V_enhancer_noPromoter_bed = BedTool.from_dataframe(bioframe.read_table('results/%s_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % time1, schema='bed3'))
    M_h3k27ac = BedTool.from_dataframe(bioframe.read_table("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time2, names_mergedIDR_dic[time2][0]), schema='bed3'))
    # M_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time2, names_mergedIDR_dic[time2][0]))
    # M_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_H3K27ac_merged_summits.bed" % (time2))
    time_atac = time_dic[time2][1]
    M_atacseq = BedTool.from_dataframe(bioframe.read_table("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time2][2]), schema='bed3'))
    V_enhancer_noPromoter_bed.coverage(b=[M_h3k27ac, M_atacseq]).to_dataframe().query('name == 0').to_csv('results/%svs%s_enhancers_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time1, time2), sep="\t", index=False, header=False)

# %% checking gene overlap
genes = bioframe.read_table("data/genome/genes.bed", schema='bed3')
for time1, time2 in [['V', 'M'], ['V', 'S'], ['M', 'V'], ['S', 'M'], ['M', 'S'], ['V', 'S']]:
    df_EnhMerged = bioframe.read_table('results/%svs%s_enhancers_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time1, time2), schema='bed3')
    print(df_EnhMerged.shape[0])

    print(len(set(bioframe.overlap(genes, df_EnhMerged, how='inner', suffixes=['', '_enh'], return_index=True).index_enh.tolist())))

# %%
for time in ['V', 'S', 'M' ]:
    for mode in ['anchors_3bins']: # [ 'anchors_3bins', 'inside', , 'start', 'end' ]: #
            print(pValue_featureOccurenceInLoop(file_loops=
            'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed' ,
            time='0', mode=mode, N_shuffle=1000,
            file_features='results/%s_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time),
            name='%s_enhAcetylOnly_allLoops_mode_%s' % (time, mode)))
# %%
for time in ['V']: #, 'S', 'M' 
    for mode in ['start', 'end']: # [ 'anchors_3bins', 'inside', ]:
        file_features = "results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed" % (time)
        el_loops = ['%sAB_loops_rightFlames0.2.bed'] #'%sAB_loops_leftFlames0.8.bed'
        for loop in el_loops:
            print(pValue_featureOccurenceInLoop(file_loops='results/long_loops/'+loop, time=time_dic[time][0], mode=mode, N_shuffle=1000, file_features='results/%s_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time), name=loop +'_enhancersAcetylOnly_allLoops_mode_' + mode))
# %%

tpm_df_rosengarten = create_rnaseq_df_rosengarten()
tpm_df_rosengarten['units'] = range(len(tpm_df_rosengarten))
# tpm_df_rosengartenEnh = tpm_df_rosengarten.copy()
tpm_df_rosengarten['group'] = 'w/o enhancer'

df_EnhMerged = bioframe.read_table('results/enhancers_merged.bed', schema='bed3')
tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['_gene', '_enh'], return_input=False,return_index=True,).dropna().index_gene.tolist()
tpm_df_rosengarten.loc[tpm_df_rosengartenEnh, 'group'] = 'endurers'

for time1, time2 in [['V', 'M'], ['V', 'S'], ['M', 'V'], ['S', 'M'], ['M', 'S'], ['V', 'S']]:
    df_EnhMerged = bioframe.read_table('results/%svs%s_enhancers_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time1, time2), schema='bed3')
    df_EnhMerged['units'] = range(len(df_EnhMerged))

    tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['_gene', '_enh'], return_input=False,return_index=True,).dropna().index_gene.tolist()
    # .filter(regex='(delta_)|(group)|(log2_)')
    # tpm_df_rosengartenEnh.loc[tpm_df_rosengartenEnh.units_enh.isnull(), 'units_enh'] = range(np.sum(tpm_df_rosengartenEnh.units_enh.isnull()))
    tpm_df_rosengarten.loc[tpm_df_rosengartenEnh, 'group'] = '%svs%s' % (time1, time2)
    # tpm_df_rosengartenEnh = tpm_df_rosengartenEnh.drop("start_enh", axis=1)
    
# %%
tmp = tpm_df_rosengarten[['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24', 'group']]
tpm_df_rosengartenEnh_melt = pd.melt(tmp, id_vars=['group'], value_vars=['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24'])

tpm_df_rosengartenEnh_melt['log2_TPM'] = np.log2(tpm_df_rosengartenEnh_melt.value + 0.01)
sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(10,10)})

# Create an array with the colors you want to use
colors = ["#B2ADAE", "#8e042d"]# Set your custom color palette
cherrry_and_ash = sns.set_palette(sns.color_palette(colors))

g = sns.lineplot(
    data=tpm_df_rosengartenEnh_melt,
    x="variable", y="log2_TPM", hue="group", #units="units_enh",
    estimator='median', lw=3, errorbar='se', 
    alpha=0.3, 
    #size=5,
    #palette=cherrry_and_ash
)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
# wrap_labels(g, 10)
plt.savefig("results/pics/paper/lineplot_enh_DifEnhNo2.acetylOnly.pdf", dpi=100, bbox_inches='tight')
# plt.legend([],[], frameon=False)
plt.show()
plt.clf()

# %%
tpm_df_rosengarten = create_rnaseq_df_rosengarten()
# tpm_df_rosengarten['units'] = range(len(tpm_df_rosengarten))
# tpm_df_rosengartenEnh = tpm_df_rosengarten.copy()
tpm_df_rosengarten['group'] = 'w/o enhancer'

df_EnhMerged = bioframe.read_table('results/enhancers_merged.bed', schema='bed3')
tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['_gene', '_enh'], return_input=False,return_index=True,).dropna().index_gene.tolist()
# tpm_df_rosengarten.loc[~tpm_df_rosengartenEnh.start_enh.isnull(), 'group'] = 'endurers'
tpm_df_rosengarten.loc[tpm_df_rosengartenEnh, 'group'] = 'endurers'

# %%
for time in ['V', 'S', 'M']:
    df_EnhMerged = bioframe.read_table('results/%s_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time), schema='bed3')
    df_EnhMerged['units'] = range(len(df_EnhMerged))

    tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['_gene', '_enh'], return_input=False,return_index=True,).dropna().index_gene.tolist()
    tpm_df_rosengarten.loc[tpm_df_rosengartenEnh, 'group'] = '%s-stage' % (time)
    # tpm_df_rosengartenEnh = tpm_df_rosengartenEnh.drop("start_enh", axis=1)
    
tmp = tpm_df_rosengarten[['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24', 'group']]
tpm_df_rosengartenEnh_melt = pd.melt(tmp, id_vars=['group'], value_vars=['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24'])

tpm_df_rosengartenEnh_melt['log2_TPM'] = np.log2(tpm_df_rosengartenEnh_melt.value + 0.01)
sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(10,10)})

# Create an array with the colors you want to use
colors = ["#B2ADAE", "#8e042d"]# Set your custom color palette
cherrry_and_ash = sns.set_palette(sns.color_palette(colors))

g = sns.lineplot(
    data=tpm_df_rosengartenEnh_melt,
    x="variable", y="log2_TPM", hue="group", #units="units_enh",
    estimator='median', lw=3, errorbar='se', 
    alpha=0.3, 
    #size=5,
    #palette=cherrry_and_ash
)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
# wrap_labels(g, 10)
plt.savefig("results/pics/paper/lineplot_enh.acetylOnly.pdf", dpi=100, bbox_inches='tight')
# plt.legend([],[], frameon=False)
plt.show()
plt.clf()
# %% h3k4me1/me3
tpm_df_rosengarten = create_rnaseq_df_rosengarten()
tpm_df_rosengarten['group'] = 'w/o enhancer'

df_EnhMerged = bioframe.read_table('results/enhancers_merged.bed', schema='bed3')
tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['_gene', '_enh'], return_input=False,return_index=True,).dropna().index_gene.tolist()
tpm_df_rosengarten.loc[tpm_df_rosengartenEnh, 'group'] = 'endurers'
for time in ['V', 'S', 'M' ]: #
    V_h3k27ac = BedTool("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][0]))
    V_h3k4me1 = bioframe.read_table("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][1]), schema='bed')
    V_h3k4me3 = bioframe.read_table("results/histones_Wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s_peaks.narrowPeak" % (time, names_mergedIDR_dic[time][3]), schema='bed')
    time_atac = time_dic[time][1]
    V_atacseq = BedTool("results/atac_seq_wang/bwa/mergedLibrary/macs/narrowPeak/%s_%s.bed" % (time_atac, names_mergedIDR_dic[time][2])) #Vegetative_r123.narrowPeak" Vegetative.mRp.clN_peaks.narrowPeak" #
    
    V_enhancer_df = V_h3k27ac.intersect(V_atacseq, e=True, f=0.25, F=0.25).to_dataframe(names=range(0,20)).iloc[:,[0,1,2]]
    V_enhancer_df.columns = ['chrom', 'start', 'end']
    # V_enhancer_df.to_csv('results/%s_idrMergeIdr.acetylInteresectATAC.bed' % (time), sep="\t", index=False,header=False)
    V_enhancer_h3k4me3 = bioframe.overlap(V_enhancer_df, V_h3k4me3, how='left', suffixes=['', '_h3k4me3'])
    V_enhancer_h3k4me31 = bioframe.overlap(V_enhancer_h3k4me3, V_h3k4me1, how='left', suffixes=['', '_h3k4me1'])
    V_enhancer_h3k4me31.loc[:,['score_h3k4me1', 'score_h3k4me3']] = V_enhancer_h3k4me31.loc[:,['score_h3k4me1', 'score_h3k4me3']].fillna(100)
    V_enhancer_h3k4me31['ratio_me31'] = V_enhancer_h3k4me31.score_h3k4me1 / V_enhancer_h3k4me31.score_h3k4me3
    # sns.scatterplot(data=V_enhancer_h3k4me31, x='score_h3k4me1', y='score_h3k4me3')
    # plt.show()
    df_EnhMerged = V_enhancer_h3k4me31.copy().query('ratio_me31 > 9')

    tpm_df_rosengartenEnh = bioframe.overlap(tpm_df_rosengarten, df_EnhMerged, how='left', suffixes=['_gene', '_enh'], return_input=False,return_index=True,).dropna().index_gene.tolist()
    tpm_df_rosengarten.loc[tpm_df_rosengartenEnh, 'group'] = '%s-stage' % (time)
    # tpm_df_rosengartenEnh = tpm_df_rosengartenEnh.drop("start_enh", axis=1)
    
tmp = tpm_df_rosengarten[['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24', 'group']]
tpm_df_rosengartenEnh_melt = pd.melt(tmp, id_vars=['group'], value_vars=['hr00', 'hr01', 'hr02', 'hr03', 'hr04','hr05', 'hr06', 'hr07', 'hr08', 'hr09', 'hr10', 'hr11', 'hr12', 'hr14','hr16', 'hr18', 'hr20', 'hr22', 'hr24'])

tpm_df_rosengartenEnh_melt['log2_TPM'] = np.log2(tpm_df_rosengartenEnh_melt.value + 0.01)
sns.set_theme(style='ticks', context='paper', rc={'figure.figsize':(10,10)})
    # BedTool.from_dataframe(V_enhancer_df).intersect(V_h3k4me3, wo=True, e=True, f=0.25, F=0.25).to_dataframe()
    
    # promoters = BedTool("data/genome/promoters.bed")
    # genes = BedTool("data/genome/genes.bed")
    # V_enhancer_noPromoter_bed = BedTool.from_dataframe(V_enhancer_df).intersect(promoters, v=True)
    # V_enhancer_noPromoter_bed.saveas('results/%s_idrMergeIdr.acetylInteresectATAC.noPromoters.bed' % (time))
g = sns.lineplot(
    data=tpm_df_rosengartenEnh_melt,
    x="variable", y="log2_TPM", hue="group", #units="units_enh",
    estimator='median', lw=3, errorbar='se', 
    alpha=0.3, 
    #size=5,
    #palette=cherrry_and_ash
)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
# wrap_labels(g, 10)
plt.savefig("results/pics/paper/lineplot_enh.acetylOnly.ratio_me31.pdf", dpi=100, bbox_inches='tight')
# plt.legend([],[], frameon=False)
plt.show()
plt.clf()
# %%
os.chdir('~/projects/dicty/hic_loop_study/')
plot_around_loop(#'bw/IS/0AB.20k.IS.cooltools.bw', 
                'bw/0AB_CS.1kb.bw',
                #'bw/dicty.IS.5k.bw',
                #'bw/V_enhancers_cov.bw',
                    #"IS500_around_enhancers.flippedNeg.median.pdf",
                     "CS_around_enh.flippedNeg.median.pdf",
                     nbins=12, resolution=2000,
                     chrom_file=dicty_chrom,
                     window=10000, mode='median',
                     ymin=-0.5, ymax=1.0,
                     vmin=-2.0, vmax=2.51,
                     norm=False,
                     fill=True,
                     how_far_from_edge=5,
                     bed_list=[
                        #'data/genome/genesNonConvNonInt_expLevel0.bed', 
                     #'data/genome/genesNonConvNonInt_expLevel1.bed',
                     #'data/genome/genesNonConvNonInt_expLevel2.bed',
                     #'data/genome/genes.bed',
                     #'data/genome/genesConvPairs.bed',
                     #"data/genome/genesConvPairs.overlap.bed",
                      'results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed'
                     #"data/genome/genesConvPairs.overlap.highExpr.bed",
                     #"data/genome/genesInConvPairs.bed"
                     ],
                     flip_negative_strand=True)
# %%
mode='whole'
for value in ['sum gene high', 'both high', 'intergenic high', 'both low']:
    pValue_featureOccurenceInLoop(
        file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
        time=None,
        mode=mode,
        N_shuffle=100,
        file_loops='results/convPairs.%s.newLA.bed' % value , #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
        name="ConvPairsAll50_%s." % value, 
        genome_file="data/genome/dicty.chrom.sizes",
        pic_path="results/pics/paper",
        resolution=2000, fraction=0.5)
# %% #TODO check why it is so?
pValue_featureOccurenceInLoop(
        file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_loops='results/divPairs.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
        name="DivPairsAll75_%s." % value, 
        genome_file="data/genome/dicty.chrom.sizes",
        pic_path="results/pics/paper",
        resolution=2000, fraction=0.5)
# %%
for conv_pair in ['insideLoop.newLA', 'notnewLA']:
    pValue_featureOccurenceInLoop(
        file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_loops='results/CGP.%s.bed' % conv_pair, #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
        name="CGP%s.All75." % conv_pair, 
        genome_file="data/genome/dicty.chrom.sizes",
        pic_path="results/pics/paper",
        resolution=2000, fraction=0.5)
# %%
for loop_group in ['outside', 'inside-1bin', 'anchors_3bins']:
    #'outside', 
    tpm_df.query('CGP == "no" and loop_group == @loop_group').loc[:, ['chrom', 'start', 'end']].sort_values(by=['chrom', 'start']).query('start > 3000').reset_index(drop=True).to_csv("data/genome/genesnotCGP.%s.bed" % loop_group, index=False, sep = '\t', header=False)
    pValue_featureOccurenceInLoop(
        file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_loops="data/genome/genesnotCGP.%s.bed" % loop_group, #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
        name="notCGP%s.All75." % loop_group, 
        genome_file="data/genome/dicty.chrom.sizes",
        pic_path="results/pics/paper",
        resolution=2000, fraction=0.5)
# %%
#.query('start > 3000')
tpm_df.query('CGP == "no"').loc[:, ['chrom', 'start', 'end']].sort_values(by=['chrom', 'start']).query('start > 3000').reset_index(drop=True).to_csv("data/genome/genesnotCGP.bed", index=False, sep = '\t', header=False)
pValue_featureOccurenceInLoop(
    file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
    time=None,
    mode=mode,
    N_shuffle=100,
    file_loops="data/genome/genesnotCGP.bed", #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
    name="notCGP.All50.", 
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000, fraction=0.5)
# %%
pValue_featureOccurenceInLoop(
    file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
    time=None,
    mode=mode,
    N_shuffle=1000,
    file_loops="results/convPairs.Mar14.bed", #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
    name="CGP.All50.", 
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000, fraction=0.5)
# %%
conv_pairs_TPM.loc[:,['chrom', 'start', 'end']].to_csv('results/convPairs.bed', index=False, sep = '\t', header=False)
pValue_featureOccurenceInLoop(
    file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed',
    time=None,
    mode=mode,
    N_shuffle=1000,
    file_loops="results/convPairs.newLA.bed", #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low2.thres_high8.with0HighToInf.step10.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
    name="LA-CGP.All50.", 
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000, fraction=0.5)

# %%
pValue_featureOccurenceInLoop(
    file_features="results/M_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed",
    time=None, 
    mode='whole',
    N_shuffle=1000,
    file_loops='results/long_loops/0AB_regular_loops.MegredAnchors.1bins_added.bed6', #'results/long_loops/loops_notFlamedAnchor.bed',# #'data/genome/genesConvPairs.byExpr.StAl.thr_low10.thres_high20.Anchors.bed', #'data/genome/genesConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZeroHighToInf.step%s.bed' % (thres_low, thres_high, step),
    name="M_enhInRegular.3bins", 
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000)
# %% suppl table 7
V_enhancer_df= bioframe.read_table('results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', schema='bed3')
genes_df = bioframe.read_table('data/genome/genes.bed', schema='bed')
enhancers_closet_gene = bioframe.closest(V_enhancer_df, genes_df)
enhancers_closet_gene.loc[enhancers_closet_gene.distance > 0, 'name_'] = np.nan

enhancers_closet_gene.loc[:,['chrom', 'start', 'end', 'name_']].to_csv('results/pics/paper/!Suppl_table/SupplTab7-ELE_withGeneID.bed', sep="\t", index=False,header=False)
# %%
# %% GO for genes with enhancers

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import textwrap
# from go import go_it

from custom_functions import create_rnaseq_df
from goatools.go_enrichment import GOEnrichmentStudy
fig, axs = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[20, 1]}, figsize=(20,18))

for time in ['V', 'S', 'M']:
    df_enhancers = bioframe.read_table('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % time, schema='bed3')
    # df_enhancers = bioframe.read_table('results/enhancers_merged.bed', schema='bed3')
    gene_withEnh = bioframe.overlap(tpm_df, df_enhancers, how='inner')['name'].tolist()

    pop = tpm_df.name.tolist() #list(assoc_dicty_dic.keys()) #

    study = gene_withEnh
    methods = ["bonferroni", "sidak", "holm", "fdr"]

    g = GOEnrichmentStudy(pop, assoc_dicty_dic_filt, go,
                            propagate_counts=True,
                            alpha=0.05,
                            methods=['fdr_bh'])
    g_res = g.run_study(study, prt=None)
    # g.print_results(g_res, min_ratio=None, pval=0.01)
    goea_results_sig = [r for r in g_res if (r.p_fdr_bh < 0.001 and r.depth > 3 and r.study_count > 40)]
    GO_df = go_it(g, study, GO_items)
    # g.wr_xlsx("results/genesWithMergedEnh.10-3.filt20.xlsx", goea_results_sig)
    # g.wr_tsv("results/%s_genesWithMergedEnh.10-3.filt20.tsv" % time, goea_results_sig)
    
    # prepare for plotting
    GO_df.GO = GO_df.GO.astype("category")
    # GO_df.GO = GO_df.GO.cat.set_categories(sorter)
    GO_df.p_corr = GO_df.p_corr.astype(float)
    df = GO_df[GO_df['GO'].notna()]
    df['log10_p'] = -np.log10(df.p_corr)
    df = df.query("n_genes > 20 and p_corr < 0.001 and depth > 3")

    cmap = mpl.cm.Reds_r
    norm = mpl.colors.Normalize(vmin = df.log10_p.min(), vmax = df.log10_p.max())

    mapper = cm.ScalarMappable(norm = norm, cmap = cm.Reds_r)

    cbl = mpl.colorbar.ColorbarBase(axs[1], cmap = cmap, norm = norm, orientation = 'vertical')

    # plt.figure(figsize = (18,6))

    ax = sns.barplot(data = df, x = 'n_genes', y = 'term',
                    palette = mapper.to_rgba(df.log10_p), ax = axs[0])

    ax.set_yticklabels([textwrap.fill(e, 22) for e in df['term']])
    plt.savefig("results/pics/paper/barplot_go_GenesWith%sEnhancers.pdf" % time, format="pdf", dpi=300)

    plt.show()
    plt.clf()
# cut -f1 results/genesWithEnh.10-3.tsv > results/genesWithEnh.10-3.geneNames.tsv
# python3 scripts/functions/revigo_access.py results/genesWithEnh.10-3.geneNames.tsv results/Revigo_BP_TreeMap_genesWithEnh.tsv

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import textwrap
import go_func
from go_func import go_it
from custom_functions import create_rnaseq_df
for time in ['V', 'S', 'M']:
    df_enhancers = bioframe.read_table('results/%s_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed' % time, schema='bed3')
    gene_withEnh = bioframe.overlap(tpm_df, df_enhancers, how='inner')['name'].tolist()

    pop = tpm_df.name.tolist() #list(assoc_dicty_dic.keys()) #

    study = gene_withEnh
    methods = ["bonferroni", "sidak", "holm", "fdr"]

    g = GOEnrichmentStudy(pop, assoc_dicty_dic_filt, go,
                            propagate_counts=True,
                            alpha=0.05,
                            methods=['fdr_bh'])
    # g_res = g.run_study(study, prt=None)
    # g.print_results(g_res, min_ratio=None, pval=0.01)
    # goea_results_sig = [r for r in g_res if (r.p_fdr_bh < 0.001 and r.depth > 3 and r.study_count > 40)]
    GO_df = go_it(g, study, GO_items)
    revigo_BP_Table = pd.read_table('results/%s_go/Revigo_BP_Table.tsv' % time)
    revigo_BP_TreeMap = pd.read_table('results/%s_go/Revigo_BP_TreeMap.tsv' % time, comment="#")
    sorter = revigo_BP_TreeMap.query('Frequency < 30').TermID.tolist()

    

    fig, axs = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[20, 1]}, figsize=(20,18))

    GO_df.GO = GO_df.GO.astype("category")
    GO_df.GO = GO_df.GO.cat.set_categories(sorter)
    GO_df.p_corr = GO_df.p_corr.astype(float)
    df = GO_df[GO_df['GO'].notna()]
    df['log10_p'] = -np.log10(df.p_corr)
    # df = df.loc[df.n_genes > 20,:]

    cmap = mpl.cm.Reds_r
    norm = mpl.colors.Normalize(vmin = df.log10_p.min(), vmax = df.log10_p.max())

    mapper = cm.ScalarMappable(norm = norm, cmap = cm.Reds_r)

    cbl = mpl.colorbar.ColorbarBase(axs[1], cmap = cmap, norm = norm, orientation = 'vertical')

    # plt.figure(figsize = (18,6))

    ax = sns.barplot(data = df, x = 'n_genes', y = 'term',
                    palette = mapper.to_rgba(df.log10_p), ax = axs[0])

    ax.set_yticklabels([textwrap.fill(e, 22) for e in df['term']])
    plt.savefig("results/pics/paper/barplot_go_GenesWith%sEnhancers.pdf" % time, format="pdf", dpi=300)

    plt.show()
    plt.clf()
