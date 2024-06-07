# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import cooler

import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import *
import functions

# %%
def process_convergence_group(group):
    """Processing the genes in the same bin"""
    N_pos = np.sum(group.strand_num==1)
    N_neg = np.sum(group.strand_num==0)
    if N_pos==N_neg:
        return 0
    elif N_pos>N_neg:
        return 1
    elif N_pos<N_neg:
        return -1


# %%
for binsize, window_size_bp in [[2000, 6000]#,
                                    #[1000, 50000],
                                     #[2000, 10000]#,
#                                     [2000, 50000],
                                    #[2000, 6000],
                                    #[2000, 8000]
                                   ]:

    window_size = window_size_bp//binsize

    # Reading genes
    df_genes = pd.read_csv(f'data/genome/dicty_fixedSourceName.gff', sep='\t', header=None, comment='#')
    df_genes.columns = ['chrom', 'source', 'type', 'start', 'end', '_', 'strand', '__', 'annot']
    df_genes = df_genes.query('type=="mRNA" and chrom!=["chrM","chr2F", "chr3F", "chrBF"]').reset_index(drop=True)
    df_genes.loc[:, 'length'] = (df_genes.end - df_genes.start)
    df_chromsizes = pd.read_csv('data/genome/dicty.chrom.sizes', sep='\t', header=None, )

    df_chromsizes.columns = ['chrom', 'length']

    # Creation of windows
    df_windows = pd.DataFrame({
        'chrom': [r.chrom for i, r in df_chromsizes.iterrows() for x in np.arange(0, r.length, binsize)],
        'bgn': [x for i, r in df_chromsizes.iterrows() for x in np.arange(0, r.length, binsize)],
        'end': [min(x + binsize, r.length) for i, r in df_chromsizes.iterrows() for x in
                np.arange(0, r.length, binsize)],
        'bin': [x // binsize for i, r in df_chromsizes.iterrows() for x in np.arange(0, r.length, binsize)]
    })

    # Assign genes to bins
    df_genes.loc[:, 'midpoint_bin'] = (df_genes.end + df_genes.start) / 2 // binsize
    df_genes.loc[:, 'newpoint_bin'] = df_genes.apply(lambda x: x.start if x.strand == '-' else x.end, axis=1) // binsize
    df_genes.loc[:, 'strand_num'] = (df_genes.strand == '+').astype(int)

for binsize, window_size_bp in [[2000, 6000]]:
    # Create draft dataframe with convergence score
    df_convergence = pd.DataFrame(df_genes.groupby(['chrom', 'midpoint_bin']) \
                                  .apply(process_convergence_group)).reset_index()
    df_convergence.columns = ['chrom', 'bin', 'value']

    df_convergence_windowized = pd.merge(df_windows, df_convergence, on=['chrom', 'bin'], how='outer').fillna(0)

    # Comuting convergence score by formulae
    v = []
    for i, r in df_convergence_windowized.iterrows():

        df_tmp = df_convergence_windowized.query(f"chrom=='{r.chrom}'").set_index('bin')

        if r.bin - window_size < 0 or r.bin + window_size > max(df_tmp.index):
            v.append(0)
            continue

        count = 0
        for j in range(-window_size, window_size):
            count += np.sign(j) * ((np.abs(j) - window_size) / window_size) * np.sum(df_tmp.loc[r.bin + j, 'value'])

        v.append(count)

    df_convergence_windowized.loc[:, 'convergence_score'] = v
    name = '../data/genome/dicty.'

#         df_convergence_windowized.to_csv(name + str(window_size_bp//1000) + 'k.' +
#                                          'binsize' + str(binsize//1000) + '.bedgraph', sep="\t", header=False,
#                                          index=False)


# %% D'Artagnan or gene batteries - MS ideas
binsize = 1000
expr_thresholds = np.quantile(tpm_df['0AB'], q=[0, 0.25, 0.75, 1]).tolist() #q=[0, 0.25, 0.5, 0.75, 1]
for e in range(len(expr_thresholds)-1): #[4, 8, 10, 20, 40]: #
    for strand in ['+', '-']:
        tpm_df = create_rnaseq_df(need_coord=True)
        tpm_df = tpm_df.query('chrom!=["chrM", "chr2F", "chr3F", "chrBF", "chrR"]').reset_index(drop=True)
        tpm_df.loc[:, 'expr_bool'] = ((tpm_df['0AB'] >= expr_thresholds[e]) & (tpm_df['0AB'] < expr_thresholds[e+1]) & (tpm_df.strand == strand)).astype(int)
        tpm_df.loc[:, 'midpoint_bin'] = (tpm_df.end + tpm_df.start) / 2 // binsize
        if strand == '+': #TODO
            tpm_df.loc[:, 'startpoint_bin'] = tpm_df.start // binsize
        elif strand == '-':
            tpm_df.loc[:, 'startpoint_bin'] = tpm_df.end // binsize


        for binsize, window_size_bp in [#[1000, 4000],
                                        #[1000, 10000],
                                        #[1000, 6000],
                                        [1000, 3000]
                                        ]:
            window_size = window_size_bp // binsize

            df_chromsizes = pd.read_csv('data/genome/dicty.chrom.sizes', sep='\t', header=None, )
            df_chromsizes.columns = ['chrom', 'length']

            # Creation of windows
            df_windows = pd.DataFrame({
                'chrom': [r.chrom for i, r in df_chromsizes.iterrows() for x in np.arange(0, r.length, binsize)],
                'bgn': [x for i, r in df_chromsizes.iterrows() for x in np.arange(0, r.length, binsize)],
                'end': [min(x + binsize, r.length) for i, r in df_chromsizes.iterrows() for x in
                        np.arange(0, r.length, binsize)],
                'bin': [x // binsize for i, r in df_chromsizes.iterrows() for x in np.arange(0, r.length, binsize)]
            })

            # Create draft dataframe with convergence score
            df_convergence_expr = pd.DataFrame(tpm_df.groupby(['chrom', 'startpoint_bin'])
                                               ['expr_bool'].sum()).reset_index()
            df_convergence_expr.columns = ['chrom', 'bin', 'value']

            df_convergence_windowized_expr = pd.merge(df_windows, df_convergence_expr,
                                                      on=['chrom', 'bin'], how='outer').fillna(0)
            df_convergence_windowized_expr.bin = df_convergence_windowized_expr.bin.astype(int)

            # Comuting convergence score by formulae
            v = []
            for i, r in df_convergence_windowized_expr.iterrows():

                df_tmp = df_convergence_windowized_expr.query(f"chrom=='{r.chrom}'").set_index('bin')

                if r.bin - window_size < 0 or r.bin + window_size > max(df_tmp.index):
                    v.append(0)
                    continue

                count = 0
                for j in range(-window_size, window_size):
                    # count += np.sign(j) * ((np.abs(j) - window_size) / window_size) * np.sum(df_tmp.loc[r.bin + j, 'value'])
                    count += ((window_size - np.abs(j)) / window_size) * np.sum(
                        df_tmp.loc[r.bin + j, 'value']) # peaks in the middle

                v.append(count)

            df_convergence_windowized_expr.loc[:, 'expr_score_' + strand] = v
            name = 'dicty.expr_score_inTheMiddle.byQuantile.'
            long_name = name + str(window_size_bp//1000) + 'k.' + 'binsize' + str(binsize//1000) + "." + strand + '.quart=' + str(e)

            # df_convergence_windowized_expr.to_csv('bw/' + long_name + '.bedgraph',
            #                                       sep="\t",
            #                                       header=False,
            #                                       index=False)

            bw = pyBigWig.open('bw/' + long_name + ".bw", "w")
            bw.addHeader(list(df_chromsizes.itertuples(index=False, name=None)), maxZooms=0)
            chroms = df_convergence_windowized_expr.chrom.tolist()
            starts = df_convergence_windowized_expr.bgn.tolist()
            ends = df_convergence_windowized_expr.end.tolist()
            values0 = df_convergence_windowized_expr.loc[:, 'expr_score_' + strand].tolist() #expr_score_' + strand
            bw.addEntries(chroms, starts, ends=ends, values=values0)
            bw.close()

            for mode in ['mean', 'median']: #

                plot_around_loop('bw/' + long_name + ".bw",
                                 long_name + '.' + mode + ".pdf", nbins=40, mode=mode,
                                 ymin=0, ymax=1, vmin=0, vmax=2)

    # plot_around_loop("bw/%s_merged.TPM_log2.HK.plus.bw",
    #                  "0_TPM_log2.HK.plus_AroundElLoops_pybbi.mean.pdf", nbins=40, mode='mean',
    #                  ymin=-2, ymax=5, vmin=-3, vmax=7)

