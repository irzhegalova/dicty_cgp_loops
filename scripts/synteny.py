# %%
import re 
import bioframe
import pandas as pd
import numpy as np
import seaborn as sns

# %%
import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import pValue_featureOccurenceInLoop

# %% D_pur
d_pur = bioframe.read_table('data/genome/d_pur/purpureum.gff3', schema='gff', comment='#')
d_pur = d_pur.loc[~d_pur.feature.isna(),:]
d_pur[['start', 'end']] = d_pur[['start', 'end']].astype(int)
d_pur_genes = d_pur.query('feature == "gene"').sort_values(['chrom', 'start']).reset_index(drop=True)
d_pur_genes['names'] = d_pur_genes['attributes'].str.split(";", expand=True)[1].str.split('Name=', expand=True)[1]
d_pur_genes = d_pur_genes.loc[:,['chrom','start', 'end','names']]
d_pur_genes['id'] = d_pur_genes.index

# %% Discoideum
d_disc = bioframe.read_table('data/genome/dicty_2.7.chr.gff', schema='gff', comment='#')
d_disc
d_disc = d_disc.loc[~d_disc.feature.isna(),:]
d_disc[['start', 'end']] = d_disc[['start', 'end']].astype(int)
d_disc_genes = d_disc.query('feature == "gene"').sort_values(['chrom', 'start']).reset_index(drop=True)
d_disc_genes['names'] = d_disc_genes['attributes'].str.split(";", expand=True)[1].str.split('Name=', expand=True)[1]
d_disc_genes = d_disc_genes.loc[:,['chrom','start', 'end','names']]
d_disc_genes['id'] = d_disc_genes.index
# %%
orthologs_df = pd.read_table("data/genome/d_pur/purpureum_ortholog.txt", header=None)
orthologs_df.columns = ['dpu', 'ddi']
orthologs_df

orthologs_df_withDDI = orthologs_df.merge(d_disc_genes, how='inner', left_on='ddi', right_on='names', suffixes=['', '_ddi'])
orthologs_df_withDDI_DPU = orthologs_df_withDDI.merge(d_pur_genes, how='inner', left_on='dpu', right_on='names', suffixes=['_ddi', '_dpu']).sort_values(['id_ddi']).reset_index(drop=True)

# %%
thres = 1
deltaDDI_ToKeep = []
for i in range(orthologs_df_withDDI_DPU.shape[0]):
    deltaDDI_vec = np.array(orthologs_df_withDDI_DPU['id_ddi'][i] - orthologs_df_withDDI_DPU['id_ddi'])
    deltaDDI_min = np.min(np.abs(deltaDDI_vec[np.nonzero(deltaDDI_vec)]))
    if deltaDDI_min <= thres:
        deltaDDI_ToKeep.append(i)

orthologs_df_withDDI_DPU_dropped = orthologs_df_withDDI_DPU.iloc[deltaDDI_ToKeep,:].sort_values(['id_dpu']).reset_index(drop=True)

# %%
deltaDPU_ToKeep = []
for i in range(orthologs_df_withDDI_DPU_dropped.shape[0]):
    deltaDPU_vec = np.array(orthologs_df_withDDI_DPU_dropped['id_dpu'][i] - orthologs_df_withDDI_DPU_dropped['id_dpu'])
    deltaDPU_min = np.min(np.abs(deltaDPU_vec[np.nonzero(deltaDPU_vec)]))
    if deltaDPU_min <= thres:
        deltaDPU_ToKeep.append(i)
# %%
df_synteny = orthologs_df_withDDI_DPU_dropped.iloc[deltaDPU_ToKeep,:].sort_values(['id_ddi']).reset_index(drop=True)
print(df_synteny.loc[:,['id_ddi', 'names_ddi']].head())


# %%
synteny_dic = {101: [101]}
for i in range(1, df_synteny.shape[0]): #synteny_dic
    print(i)
    assigned_bool = False
    for key in synteny_dic.keys():
        deltaDDI_vec = np.array(df_synteny['id_ddi'][i] - synteny_dic[key])
        deltaDDI_min = np.min(np.abs(deltaDDI_vec[np.nonzero(deltaDDI_vec)]))
        if deltaDDI_min <= thres:
            # synteny_dic[df_synteny['id_ddi'][i]] = str(key)
            assigned_bool = True
            break
    if assigned_bool:
        synteny_dic[key].append(df_synteny['id_ddi'][i])
    else:
        synteny_dic[df_synteny['id_ddi'][i]] = [df_synteny['id_ddi'][i]]

d = {}
for key in synteny_dic.keys():
    for item in synteny_dic[key]:
      d[item] = key  

# %%
df_synteny_cl = df_synteny.replace({"id_ddi": d})

df_synteny_cl = bioframe.merge(df_synteny_cl ,min_dist=4000, cols=['chrom_ddi', 'start_ddi', 'end_ddi'], on=['id_ddi'])
df_synteny_cl.loc[:,['chrom_ddi', 'start_ddi', 'end_ddi']].to_csv('results/synteny_manualMerge.bed',
           sep="\t", index=False,
           header=False)

df_synteny_cl1 = pd.concat([df_synteny_cl.loc[:, ['chrom_ddi', 'start_ddi']], df_synteny_cl.loc[:, ['chrom_ddi', 'end_ddi']]])
df_synteny_cl1.loc[df_synteny_cl1['start_ddi'].isna(), 'start_ddi'] = df_synteny_cl1.loc[df_synteny_cl1['start_ddi'].isna(), 'end_ddi']
df_synteny_cl1['end_ddi'] = df_synteny_cl1['start_ddi'] + 1
df_synteny_cl1.sort_values(['chrom_ddi', 'start_ddi'], inplace=True)
df_synteny_cl1[['start_ddi', 'end_ddi']] = df_synteny_cl1[['start_ddi', 'end_ddi']].astype(int)
df_synteny_cl1.to_csv('results/synteny_manualMerge.anchors.bed', sep="\t", index=False, header=False)

# %% synteny symap
synteny_df_old = pd.read_table('/home/fox/tools/symap_5/data/seq_results/d_disc_to_dicties_hits7/final/block.txt').sort_values(['grp1', 'start1'])
synteny_df_old['length'] = synteny_df_old.end1 - synteny_df_old.start1
synteny_df = pd.read_table('/home/fox/tools/symap_5/data/seq_results/d_disc_to_dicties_hits2/final/block.txt').sort_values(['grp1', 'start1'])
synteny_df['length'] = synteny_df.end1 - synteny_df.start1
synteny_df = synteny_df.loc[(synteny_df["#gene1"] < 7) | (synteny_df["#gene2"] < 7), :]
synteny_df = pd.concat([synteny_df, synteny_df_old])
synteny_df_smartFilter = synteny_df.loc[:,['grp1', 'start1', 'end1']]
synteny_df_smartFilter.columns = ['chrom', 'start', 'end']
synteny_df_smartFilter.to_csv('results/synteny_smallSymap.SmartFiltered.bed', sep="\t", index=False, header=False)
# synteny_df_cl = bioframe.merge(synteny_df, cols=['grp1', 'start1', 'end1'],)
# synteny_df.loc[:,['grp1', 'start1', 'end1']].to_csv('results/synteny_smallSymap.bed', sep="\t", index=False, header=False)


# %%
df_synteny_cl1 = pd.concat([synteny_df_smartFilter.loc[:, ['chrom', 'start']], synteny_df_smartFilter.loc[:, ['chrom', 'end']]])
df_synteny_cl1.loc[df_synteny_cl1['start'].isna(), 'start'] = df_synteny_cl1.loc[df_synteny_cl1['start'].isna(), 'end']
df_synteny_cl1['end'] = df_synteny_cl1['start'] + 1
df_synteny_cl1.sort_values(['chrom', 'start'], inplace=True)
df_synteny_cl1[['start', 'end']] = df_synteny_cl1[['start', 'end']].astype(int)
df_synteny_cl1.to_csv('results/synteny_smallSyMap.SmartFiltered.borders.bed', sep="\t", index=False, header=False)
# df_synteny_cl1 = pd.concat([synteny_df.loc[:, ['grp1', 'start1']], synteny_df.loc[:, ['grp1', 'end1']]])
# df_synteny_cl1.loc[df_synteny_cl1['start1'].isna(), 'start1'] = df_synteny_cl1.loc[df_synteny_cl1['start1'].isna(), 'end1']
# df_synteny_cl1['end1'] = df_synteny_cl1['start1'] + 1
# df_synteny_cl1.sort_values(['grp1', 'start1'], inplace=True)
# df_synteny_cl1[['start1', 'end1']] = df_synteny_cl1[['start1', 'end1']].astype(int)
# df_synteny_cl1.to_csv('results/synteny_smallSyMap.filtered.borders.bed', sep="\t", index=False, header=False)

# %% acnhors from SyMap
orhologous_df = pd.read_table('/home/fox/tools/symap_5/data/seq_results/d_disc_to_dicties/final/anchors.txt')
orhologous_df['chrom'] = orhologous_df.block.str.split('.', expand=True)[0].tolist()
orhologous_df.loc[:,['chrom', 'start1', 'end1']].to_csv('results/anchors_smallSymap.bed', sep="\t", index=False, header=False)
# %%
import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import pValue_featureOccurenceInLoop
for cl in range(6):
    loops_kmeans = 'data/genome/loops_withGroup' + str(cl) + '.bed3'
# loops_kmeans = 'data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed'
    pValue_featureOccurenceInLoop(file_loops=loops_kmeans, 
        time=None,
        mode='inside-1bin',
        N_shuffle=1000,
        file_features='results/synteny_smallSymap.SmartFiltered.bed',
        name='synteny_smallSyMap.filtered_loopWithGroup%s' % cl)
# %%
mode = 'inside-1bin'
pValue_featureOccurenceInLoop(file_loops='results/0AB_loops_notBMA.bed', #'results/0AB_loops_thresBMA.bed', 
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_features='data/genome/Blocks_of_conserved_gene_order.bed', #'results/synteny_smallSymap.SmartFiltered.bed',
        name='synteny_notBMA.%s' % mode)
# %%
for mode in ['anchors_3bins', 'inside-1bin']: #'whole', , 
    pValue_featureOccurenceInLoop(file_loops=loops_kmeans, 
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_features='results/synteny_smallSyMap.filtered.borders.bed', #'results/0AB_loops_thresBMA.bed',
        name='SyntenysmallSymapFilteredBorders_loop_mode_%s' % (mode))
# %%
for mode in ['whole']: #'whole', , 
    pValue_featureOccurenceInLoop(file_features='results/V_enhancers_idrMergeIdr.chipseqUnionThenInteresectATAC.bed', 
        time=None,
        mode=mode,
        N_shuffle=1000,
        file_loops='results/synteny_smallSymap.SmartFiltered.bed', #'results/0AB_loops_thresBMA.bed',
        name='SyntenysmallSymapSmartFilteredBorders_enhancer_mode_%s' % (mode))
    

# %% higlass
# clodius aggregate bedfile \
# --chromsizes-filename data/genome/dicty.chrom.sizes \
# results/synteny_smallSymap.bed

# clodius aggregate bedfile \
# --chromsizes-filename data/genome/dicty.chrom.sizes \
# results/synteny_Symap.bed

# %%
import bioframe
df_windows = bioframe.read_table('data/genome/dicty_w2k.bed', schema='bed3')

# loops_df = bioframe.read_table('data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed' % Timing, schema='bed3')
loops_df_cov = bioframe.coverage(df_windows, synteny_df_smartFilter)
loops_df_cov['frac'] = loops_df_cov.coverage / loops_df_cov.end
print(np.median(loops_df_cov['frac']))

# %%
arms = bioframe.read_chromsizes('./data/genome/dicty.chrom.sizes', filter_chroms=False, as_bed=True) 
arms = arms[['chrom', 'end']]

bw = pyBigWig.open('bw/synteny_cov.bw', "w")
bw.addHeader(list(arms.itertuples(index=False, name=None)), maxZooms=0)
chroms = loops_df_cov.chrom.tolist()
starts = loops_df_cov.start.tolist()
ends = loops_df_cov.end.tolist()
values0 = loops_df_cov.frac.tolist()
bw.addEntries(chroms, starts, ends=ends, values=values0)
bw.close()
# %%
for mode in ['median']:
    # plot_around_loop("bw/synteny_cov.bw",
    #                 "SyntenyCov_AroundLoops_pybbi.%s.pdf" % mode,
    #                 fill=True, mode=mode,
    #                 window=20000, nbins=30,
    #                 how_far_from_edge=10,
    #                 ymin=0, ymax=0.001,
    #                 vmin=0, vmax=0.005)
    plot_around_loop("bw/synteny_cov.bw",
                    "SyntenyCov_AroundLoopsWithGroups_pybbi.%s.pdf" % mode,
                    fill=True, mode=mode,
                    window=20000, nbins=30,
                    how_far_from_edge=10,
                    ymin=0, ymax=0.001,
                    vmin=0, vmax=0.005, bed_list=['data/genome/loops_withGroup%s.bedpe' % i for i in range(6)] + ['results/0AB_loops_thresBMA.bed']
    )
    
    
# %%
loops_df_notBMA = bioframe.read_table('results/0AB_loops_notBMA.bed', schema='bed3')
loops_df_notBMA['group'] = 'notBMA'
loops_df_thresBMA = bioframe.read_table('results/0AB_loops_thresBMA.bed', schema='bed3')
loops_df_thresBMA['group'] = 'BMA'
loops_df = pd.concat([loops_df_thresBMA, loops_df_notBMA])
loops_df['length'] = loops_df.end - loops_df.start
synteny_df = bioframe.read_table('data/genome/Blocks_of_conserved_gene_order.bed', schema='bed3')
syntenyMerged_df = bioframe.merge(synteny_df)
# %%
loops_df_withCov = bioframe.coverage(loops_df, syntenyMerged_df) #, how='left', return_overlap=True
loops_df_withCov['cov_synteny'] = loops_df_withCov.coverage / loops_df_withCov.length
sns.boxplot(loops_df_withCov, x='group', y='cov_synteny')
# %%
# %% use coolpuppy to create pics for syntheny block
flank = 20000
ConvPair_notLoopAnchor = bioframe.read_table('data/genome/Blocks_of_conserved_gene_order.bed', schema='bed')#.query('score == "none"')
pup = coolpup.pileup(clr, ConvPair_notLoopAnchor, #groupby=['expr_group1', 'anchor_group2'], #['group1'], #'length_group2'
rescale=True, rescale_size=int(1+flank * 2// resolution),  rescale_flank=10,
local=True,
features_format='bed', view_df=df_chromsizes, nproc=4, 
#by_strand=True, 
flip_negative_strand=True, 
expected_df=expected, 
flank=flank, min_diag=2)

plotpup.plot(pup,
             score=False, 
             cmap='coolwarm', #'viridis', #'coolwarm', #
             scale='log',
             #sym=True,
             #vmax=1.2, vmin=0.5,
             vmax=1.1, vmin=0.9,
             #vmin=0.0015, vmax=0.025,
             #vmin=0.005, vmax=0.025,
             height=5)
plt.savefig('results/pics/paper/av_synthenyBlock.flank20k.pdf', dpi=100, bbox_inches='tight')
plt.show()


# %% synteny blocks
gene_blocks_df = pd.read_excel('data/genome/Blocks_of_conserved_gene_order.xlsx', sheet_name='list1')
gene_blocks_df['block'].replace(' ', '_', regex=True, inplace=True)
gene_blocks_dic = {}
gene_blocks_chr_dic = {}
for i in range(gene_blocks_df.shape[0]):
    if gene_blocks_df.block[i] in gene_blocks_dic:
        if gene_blocks_df.block[i] == gene_blocks_df.block[i-1]:
            gene_blocks_dic[gene_blocks_df.block[i]].append(gene_blocks_df.dd_stop[i])
    else:
        gene_blocks_dic[gene_blocks_df.block[i]] = [gene_blocks_df.dd_start[i]]
        gene_blocks_chr_dic[gene_blocks_df.block[i]] = 'chr' + str(gene_blocks_df.loc[i, 'dd_chr'])

gene_blocks_list = gene_blocks_df['block'].unique().tolist()
gene_blocks_tmp = []

for block in gene_blocks_list:
    gene_blocks_tmp.append([gene_blocks_chr_dic[block], np.min(gene_blocks_dic[block]), np.max(gene_blocks_dic[block]), block])
gene_blocks_bed = pd.DataFrame(gene_blocks_tmp)
gene_blocks_bed.columns = ['chrom', 'start', 'end', 'name']

# gene_blocks_bed.to_csv('data/genome/Blocks_of_conserved_gene_order.bed', sep="\t", index=False, header=False)

# %%
gene_blocks_df = pd.read_excel('data/genome/Blocks_of_conserved_gene_order.xlsx', sheet_name='list1')
gene_blocks_df['block'].replace(' ', '_', regex=True, inplace=True)
gene_blocks_df['chrom'] = 'chr'+ gene_blocks_df['dd_chr'].astype(str)
gene_blocks_df['start'] = gene_blocks_df.dd_start.astype(int)
gene_blocks_df['end'] = gene_blocks_df.dd_stop.astype(int)
gene_blocksMerged_df = bioframe.merge(gene_blocks_df, min_dist=4000, on=['block']).sort_values(['chrom', 'start'])
# gene_blocksMerged_df.to_csv('data/genome/Blocks_of_conserved_gene_order_splited.bed', sep="\t", index=False, header=False)
gene_blocksMergedAnchor_df = pd.concat([gene_blocksMerged_df.loc[:, ['chrom', 'start']], gene_blocksMerged_df.loc[:, ['chrom', 'end']]])
gene_blocksMergedAnchor_df.loc[gene_blocksMergedAnchor_df['start'].isna(), 'start'] = gene_blocksMergedAnchor_df.loc[gene_blocksMergedAnchor_df['start'].isna(), 'end']
gene_blocksMergedAnchor_df['end'] = gene_blocksMergedAnchor_df['start'] + 1
gene_blocksMergedAnchor_df.sort_values(['chrom', 'start'], inplace=True)
gene_blocksMergedAnchor_df[['start', 'end']] = gene_blocksMergedAnchor_df[['start', 'end']].astype(int)
gene_blocksMergedAnchor_df.to_csv('data/genome/Blocks_of_conserved_gene_order_splited.anchors.bed', sep="\t", index=False, header=False)
# bedtools genomecov -i data/genome/Blocks_of_conserved_gene_order_splited.anchors.bed -g data/genome/dicty.chrom.sizes -bga > data/genome/Blocks_of_conserved_gene_order_splited.anchors.bg
# bedGraphToBigWig data/genome/Blocks_of_conserved_gene_order_splited.anchors.bg data/genome/dicty.chrom.sizes data/genome/Blocks_of_conserved_gene_order_splited.anchors.bw

# bedtools genomecov -i data/genome/Blocks_of_conserved_gene_order_splited.bed -g data/genome/dicty.chrom.sizes -bga > data/genome/Blocks_of_conserved_gene_order_splited.bg
# bedGraphToBigWig data/genome/Blocks_of_conserved_gene_order_splited.bg data/genome/dicty.chrom.sizes data/genome/Blocks_of_conserved_gene_order_splited.bw
# %%
mode='mean'
plot_around_loop('data/genome/Blocks_of_conserved_gene_order_splited.bw', 'syntenySplited_AroundLoops_pybbi.%s.pdf' % (mode), fill=True, mode=mode, window=20000, nbins=30, how_far_from_edge=10, ymin=0, ymax=0.9, vmin=-0, vmax=0.9)

# %%
synteny_df = bioframe.read_table('data/genome/Blocks_of_conserved_gene_order_splited.bed', schema='bed3')

df_loops = load_BedInMode('data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed', '0', 'anchors_3bins')
synteny_atLoopsAcnhors = bioframe.overlap(synteny_df, df_loops, how='left', return_overlap=True, suffixes=('', '_anchors'))
synteny_atLoopsAcnhors['group'] = 'outside'
synteny_atLoopsAcnhors['frac_anchors'] = (synteny_atLoopsAcnhors['overlap_end'] - synteny_atLoopsAcnhors['overlap_start']) / (synteny_atLoopsAcnhors['end'] - synteny_atLoopsAcnhors['start'])
synteny_atLoopsAcnhors.loc[synteny_atLoopsAcnhors['frac_anchors'] > 0.75, 'group'] = 'anchors'
synteny_atLoopsAcnhors = synteny_atLoopsAcnhors.drop(['overlap_start', 'overlap_end'],axis=1)

df_loops = load_BedInMode('data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed', '0', 'inside-1bin')
synteny_atLoops = bioframe.overlap(synteny_atLoopsAcnhors, df_loops, return_overlap=True, how='left', suffixes=('', '_inside'))
synteny_atLoops['frac'] = (synteny_atLoops['overlap_end'] - synteny_atLoops['overlap_start']) / (synteny_atLoops['end'] - synteny_atLoops['start']) 
synteny_atLoops.loc[synteny_atLoops['frac'] > 0.99, 'group'] = 'inside'

synteny_atLoops['length'] = synteny_atLoops['end'] - synteny_atLoops['start']
sns.boxplot(synteny_atLoops, x='group', y='length')
# %%
gene_blocks_bed_tmp = pd.DataFrame(gene_blocks_tmp)
resolution = 1
gene_blocks_bed_tmp.columns = ['chrom', 'start', 'end', 'name']
data = {'chrom': gene_blocks_bed_tmp.chrom,
        'start': gene_blocks_bed_tmp.start,
        'end': gene_blocks_bed_tmp.start + resolution}
df_bed_left = pd.DataFrame(data)
df_bed_left.to_csv('data/genome/Blocks_of_conserved_gene_order.start.bed', sep="\t", index=False, header=False)

data = {'chrom': gene_blocks_bed_tmp.chrom,
        'start': gene_blocks_bed_tmp.end - resolution,
        'end': gene_blocks_bed_tmp.end}
df_bed_right = pd.DataFrame(data)
df_bed_right.to_csv('data/genome/Blocks_of_conserved_gene_order.end.bed', sep="\t", index=False, header=False)

gene_blocks_bed = pd.concat([df_bed_left, df_bed_right])

# gene_blocks_bed_tmp['start'] = gene_blocks_bed_tmp['end'] - 1
gene_blocks_bed.to_csv('data/genome/Blocks_of_conserved_gene_order_tmp.bed', sep="\t", index=False, header=False)
# %%
pValue_featureOccurenceInLoop(file_loops='data/genome/Blocks_of_conserved_gene_order.bed', #'results/long_loops/%sAB_loops_leftFlames0.8.bed',
                              time=None,
                              mode='whole',
                              N_shuffle=1000,
                              file_features='results/VM_common_enhancers.bed',
                              name='VM_common_enh_occurence_ syntheny_block')
# %%
mode = 'inside'
pValue_featureOccurenceInLoop(file_loops='data/loops_quantifyChromosight/%sAB_consecutive_1binsAdded.bed', #'data/loops_quantifyChromosight/%sAB_conconsecutive_0binsAdded.bed', #"data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed",
                              time='0',
                              mode=mode,
                              N_shuffle=1000,
                              file_features='data/genome/Blocks_of_conserved_gene_order.bed',
                              name='syntheny_block_occurence_StALLoops_mode%s' % mode)
# %%
syntheny_df = bioframe.read_table("data/genome/dicty_noRTE_no_pseudogenes.gff", schema='gff')
#('data/genome/genes.bed', schema='bed3') #data/genome/Blocks_of_conserved_gene_order.bed
syntheny_df_cov = bioframe.coverage(df_chromsizes, syntheny_df)
syntheny_df_cov['frac'] = syntheny_df_cov.coverage / syntheny_df_cov.end
np.mean(syntheny_df_cov['frac'].tolist())

# %%
genesInsyntheny = bioframe.count_overlaps(syntheny_df, tpm_df, return_input=True)
genesInsyntheny['count'].sum()
# !bedtools intersect -a data/genome/Blocks_of_conserved_gene_order.bed -b data/genome/genes_noRTE_no_pseudogenes.bed -wb -F 0.5 | cut -f8 |sort | uniq | wc -l
# %%
syntheny_df = bioframe.read_table('data/loops_quantifyChromosight/0AB_consecutive_1binsAdded.bed', schema='bed3')
syntheny_df_cov = bioframe.coverage(df_chromsizes, syntheny_df)
syntheny_df_cov['frac'] = syntheny_df_cov.coverage / syntheny_df_cov.end
syntheny_df_cov['frac']

# %%
syntheny_df = bioframe.read_table('data/genome/Blocks_of_conserved_gene_order.bed', schema='bed3')
syntheny_df_complement = bioframe.complement(syntheny_df, df_chromsizes)
syntheny_df_complement.to_csv('data/genome/NotBlocks_of_conserved_gene_order.bed', sep="\t", index=False, header=False)


# %%
df_loops = bioframe.read_table('data/genome/Blocks_of_conserved_gene_order.bed', schema='bed3')
tmp1 = df_loops.loc[:, ['chrom', 'start']]
tmp2 = df_loops.loc[:, ['chrom', 'end']]
tmp2.columns = ['chrom', 'start']
list_nucl = pd.concat([tmp1, tmp2], axis=0).sort_values(by=['chrom', 'start']).reset_index()
list_nucl['end'] = list_nucl['start'] + 1
list_nucl.loc[:, ['chrom', 'start', 'end']].to_csv('data/genome/Blocks_of_conserved_gene_order.anchors.bed', sep="\t", index=False, header=False)