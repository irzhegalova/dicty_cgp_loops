# %% import standard python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
os.chdir('../../')

# import libraries for biological data analysis
import cooler
import bioframe
import cooltools

from packaging import version
if version.parse(cooltools.__version__) < version.parse('0.5.0'):
    raise AssertionError("tutorials rely on cooltools version 0.5.0 or higher,"+
                         "please check your cooltools version and update to the latest")

# %% load chromsizes
df_chromsizes = pd.read_csv('data/genome/dicty.chrom.sizes', sep='\t', header=None, )
df_chromsizes.columns = ['chrom', 'end']
df_chromsizes.loc[:, 'start'] = 0
df_chromsizes["name"] = df_chromsizes["chrom"]
df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]

# %%
# Open cool file with Hi-C data:
clr = cooler.Cooler('../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/2000')
# clr = cooler.Cooler('data/ibg_knockout/DMSO.null.mapq_30.500.mcool::/resolutions/2000')
# clr = cooler.Cooler('data/ibg_knockout/coolers_library_group/Flavopiridol.null.mapq_30.500.mcool::/resolutions/2000')
# clr = cooler.Cooler('data/ibg_knockout/coolers_library_group/Triptolide.null.mapq_30.500.mcool::/resolutions/2000')
# Set up selected data resolution:
resolution = clr.binsize

expected = cooltools.expected_cis(clr, view_df=df_chromsizes, nproc=2,
                                  chunksize=1_000_000)

# loop_file = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed"
loop_file = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.filtered.bed"

loop = bioframe.read_table(loop_file, schema='bed').query(f'chrom in {clr.chromnames}')


# %% updated version of the loop file levels generation
# loop_list = []
# i = 0
# while i < loop.shape[0]:
#     loop_start = loop['start'][i]
#     loop_end1 = loop['end'][i]
#     loop_start2 = loop['start'][i+1]
#     loop_end2 = loop['end'][i+1]
#     if loop_start2 - loop_end1 > 8000:
#         loop_end = loop_start2
#         i += 1
#     elif loop_start2 - loop_end2 < 8000:
#     vec = [loop['chrom'][i], loop_start, loop_end]
flank = 2_000
c = 0
# select_vec = [loop.start[i+1] - loop.end[i] < 8000 for i in range(loop.shape[0]-1)]
# select_vec.append(True)
loop_filtered = loop#.loc[select_vec,:].reset_index()
for n in range(0,4): #[1:2]
    vec = loop_filtered['end']
    left_vec = loop_filtered['start'][0:len(vec)-n]
    right_vec = [vec[i] for i in range(n, len(vec))]
    # df = pd.DataFrame({'chrom': loop['chrom'][0:len(vec)-n], 'x':left_vec, 'y':right_vec})
    # df.head()
    paired_sites = pd.DataFrame({'chrom1': loop_filtered['chrom'][0:len(vec)-n],
                                 'start1': left_vec,
                                 'end1': left_vec,
                                 'chrom2': loop_filtered['chrom'][0:len(vec)-n],
                                 'start2': right_vec,
                                 'end2': right_vec})
    paired_sites = paired_sites[paired_sites['chrom1'] == paired_sites['chrom2']]
    paired_sites['end1'] += 2000
    paired_sites['start2'] -= 2000

    # create the stack of snips:
    oe_stack = cooltools.pileup(clr, paired_sites, view_df=df_chromsizes,
                             expected_df=expected, flank=flank)

    av_middle = np.nansum(np.nansum(oe_stack, axis = 0), axis = 0)
    paired_sites['0AB'] = av_middle
    paired_sites = paired_sites.loc[paired_sites.start2 - paired_sites.end1 < 60000,:]
    # paired_sites.to_csv(
    #     'results/DMSO_n+%s_loopStrength.tsv' % (n),
    #     sep="\t", index=False,
    #     header=True)

# %%
sns.set_theme(style="whitegrid")

f, axs = plt.subplots(1, 4,
figsize=(18, 10))

flank = 20_000
ticks_pixels = np.linspace(0, flank*2//resolution,5)
ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)

c = 0
for n in range(0,4):
    vec = loop['end']
    left_vec = loop['start'][0:len(vec)-n]
    right_vec = [vec[i] for i in range(n, len(vec))]
    # df = pd.DataFrame({'chrom': loop['chrom'][0:len(vec)-n], 'x':left_vec, 'y':right_vec})
    # df.head()
    paired_sites = pd.DataFrame({'chrom1': loop['chrom'][0:len(vec)-n],
                                 'start1': left_vec,
                                 'end1': left_vec,
                                 'chrom2': loop['chrom'][0:len(vec)-n],
                                 'start2': right_vec,
                                 'end2': right_vec})
    paired_sites = paired_sites[paired_sites['chrom1'] == paired_sites['chrom2']]
    paired_sites['end1'] += 2000
    paired_sites['start2'] -= 2000
    if n >= 1:
        loop_distance = 60000*n
        paired_sites = paired_sites.loc[paired_sites.start2 - paired_sites.end1 < loop_distance,:]


    # create the stack of snips:
    stack = cooltools.pileup(clr, paired_sites, view_df=df_chromsizes,
                             expected_df=expected, flank=flank)

    oe_mtx = np.nanmedian(stack, axis=2)

    ax = axs[c]

    im = ax.imshow(
        np.log2(oe_mtx),
        vmax = 0.9,
        vmin = -0.35,
        cmap='coolwarm',
        interpolation='none')
    which_middle = oe_mtx.shape[0] // 2
    enr = np.nanmedian(
        np.nanmedian(np.log2(oe_mtx)[which_middle - 1:which_middle + 2,
                     which_middle - 1:which_middle + 1],
                         axis=0),
            axis=0)

    ax.text(
        s=np.round(enr, 3),
        y=0.95,
        x=0.05,
        ha="left",
        va="top",
        size="small",
        transform=ax.transAxes,
        )

    ax.set(title="N + " + str(n),
       xticks=ticks_pixels,
       xticklabels=ticks_kbp,
       yticks=ticks_pixels,
       yticklabels=ticks_kbp,
       xlabel='relative position, kbp',
       ylabel='relative position, kbp')
    ax.set_xticks([])
    ax.set_yticks([])
    c += 1

cb_ax = f.add_axes([0.92, 0.1, 0.02, 0.8])
cbar = f.colorbar(im, cax=cb_ax, pad=2, label = 'log2 median obs/exp')
plt.savefig("results/pics/paper/Fig2f_av_loops_N_test.pdf", format="pdf", dpi=300)
plt.show()
# %%
