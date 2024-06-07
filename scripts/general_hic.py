#%% cis trans ratio
import cooltools.api.coverage as cl_cov
import cooler
import matplotlib.pyplot as plt
import seaborn as sns

# load
stages = ['0AB', '2AB', '5AB', '8AB']
resolution = 2000 #int(1e5) #2000 #int(1e5) #16000#

coolpath = {  # access: coolpath[stage]
    stage: f'../dicty_hic_distiller/subsampled/{stage}.filtered.mcool::/resolutions/{resolution}'
    for stage in stages
}

for stage in stages:
    clr = cooler.Cooler(coolpath[stage])
    cistrans_value = cl_cov.coverage(clr, ignore_diags=0)
    print(sum(cistrans_value[0]) / (sum(cistrans_value[1]) - sum(cistrans_value[0])))

# %% subsample
for i in range(0, 8, 1):
    # create path to file
    in_1 = os.path.join(cooler_dir, cooler_files[i]) + '::/resolutions/500'
    # load merged cool file back to Python
    clr = cooler.Cooler(in_1)
    # create array with chromosomes needed to be extracted
    c_sub = clr.chromnames[0:6]
    # create bin df
    c_bin_df = pd.DataFrame(columns=clr.bins().columns)
    # create pixels df
    c_pix_df = pd.DataFrame(columns=clr.pixels().columns)
    # iterate through chromosomes to extract pixels
    for chrom in c_sub:
        c_bin = clr.bins().fetch(chrom)
        c_pix = clr.pixels().fetch(chrom)
        c_bin_df = c_bin_df.append(c_bin)
        c_pix_df = c_pix_df.append(c_pix)
    r = clr.offset('chrM')
    c_pix_df_fixed1 = \
        c_pix_df[c_pix_df.bin2_id < r]
    c_pix_df_fixed = c_pix_df_fixed1[c_pix_df_fixed1.bin1_id != c_pix_df_fixed1.bin2_id]
    c_pix_df_fixed.sort_values(by=['bin1_id', 'bin2_id'])

    # create path to save
    name = cooler_files[i].split(".")[0]
    filtered_pixels_path = '~/projects/dicty_hic_loop_study/dicty_hic_distiller/REPLICAS/pixels/' \
                           + name + '.filtered.txt'
    #file to tsv
    c_pix_df_fixed.to_csv(filtered_pixels_path,
                      sep='\t', mode='w', index=None, header=False)
# %% bash
# creating new cools with chr 1-6
# . ~/.bashrc
#    cooler load -f coo \
#     /home/izhegalova/projects/dicty_hic_loop_study/data/genome/dicty.chrom.sizes:500 \
#         ${pix_path}/${pix} \
#         ${cool_path}/${cool} \
#         --metadata /home/izhegalova/projects/dicty_hic_loop_study/dicty_hic_distiller/cool_info.json; \
!pix_path="/home/izhegalova/projects/dicty_hic_loop_study/dicty_hic_distiller/REPLICAS/pixels";\
    cool_path="/home/izhegalova/projects/dicty_hic_loop_study/dicty_hic_distiller/REPLICAS/filtered";\
    for pix in $(ls ${pix_path} | grep '.txt'); \
    do prefix="${pix%.*.*}"; \
    echo ${prefix}; \
    cool="${prefix}.filtered.cool"; \
    cooler balance ${cool_path}/${cool} -p 32 --ignore-diags 2; \
    cooler coarsen ${cool_path}/${cool} -p 32 -k 4 -o ${cool_path}/${prefix}.filtered.2kb.cool; \
    cooler balance ${cool_path}/${prefix}.filtered.2kb.cool -p 32 --ignore-diags 2; \
    done;

# %%
filtered_cooler_dir = "~/projects/dicty_hic_loop_study/dicty_hic_distiller/REPLICAS/filtered"
filtered_cooler_files=[]
#select only files in the directory
for fname in os.listdir(filtered_cooler_dir):
    cooler_path = os.path.join(filtered_cooler_dir, fname)
    if os.path.isdir(cooler_path):
        # skip directories
        continue
    else:
        if ".zero-filled" not in fname:
            if "tsv" not in fname:
                if ".2kb" not in fname:
                    filtered_cooler_files.append(fname)
filtered_cooler_files.sort()

info_list=[]
info_list_after=[]
for file in filtered_cooler_files:
    cooler_file = os.path.join(filtered_cooler_dir, file)
    # load initial cool file
    clr = cooler.Cooler(cooler_file)
    # count contacts in order to get
    info_list.append(clr.info['sum'])
print(np.min(info_list))

import numpy as np
from cooltools import sample
sampled_cooler_dir="/home/izhegalova/projects/dicty_hic_loop_study/dicty_hic_distiller/REPLICAS/subsampled"
target = np.min(info_list)
sampled_info_list = []
for file in filtered_cooler_files:
    sampled_cooler_file=os.path.join(sampled_cooler_dir, file)
    filtered_cooler_file=os.path.join(filtered_cooler_dir, file)
    sample.sample_cooler(filtered_cooler_file, sampled_cooler_file, 22080000) #manual input of min number of reads
    # sampled_clr = cooler.Cooler(sampled_cooler_file)
    # sampled_info_list.append(sampled_clr.info['sum'])
    # c_sub = sampled_clr.chromnames[0:6]
    # for chrom in c_sub:
    #     sampled_c_bin = sampled_clr.bins().fetch(chrom)
    # cooler.balance_cooler(sampled_clr, tol=1e-05, min_nnz=0,
    #                       min_count=0, mad_max=0, cis_only=False,
    #                       trans_only=False, ignore_diags=False,
    #                       max_iters=200, rescale_marginals=True,
    #                       use_lock=False, blacklist=None, x0=None,
    #                       store=True, store_name='weight')
    
# %% compute IS - bash
!window=10000; \
    project_dir="~/projects/dicty_hic_loop_study"; \
    cooler_dir="%s/dicty_hic_distiller/subsampled" % project_dir; \
    cool_matrixes=$(ls ${cooler_dir} | grep AB.filtered.2kb.cool | sort); \
    for file in ${cool_matrixes}; \
    do prefix="${file%.*.*.*}"; \
        cooler balance ${cooler_dir}/${file} -p 40 --ignore-diags 2 ; \
        cooltools diamond-insulation ${cooler_dir}/${file} ${window} | awk '$4 == "False"' | awk '$5 != "nan"' | awk -v OFS='\t' '{ print $1, $2, $3, $5 }' > ${project_dir}/data/IS/${prefix}.$((${window} / 1000))k.IS.cooltools.bedgraph; \
done

# %% chromosight
chromosight detect --threads 4 --min-dist 6000 --max-dist 45000 --pearson 0.2 \
dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/2000 \
results/0AB_chromosight
    
# %% Loop Anchor Index - Fig 2b
loops_5bins = {}
loops_3bins = {}
loops_1bins = {}
for time1 in [0, 2, 5, 8]:
      pathToFile = 'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed' % time1
      loops_1bins[time1] = load_BedInMode(pathToFile, mode='anchors')
      #bioframe.read_table(pathToFile, schema='bedpe')
      pathToFile_3bins = 'data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed' % time1
      # loops_3bins[time1] = bioframe.read_table(pathToFile_3bins, schema='bedpe'
      loops_3bins[time1] = load_BedInMode(pathToFile_3bins, mode='anchors_3bins')
      loops_5bins[time1] = load_BedInMode(pathToFile_3bins, mode='anchors_5bins')

# Paired comparison
percent = []
for time1 in [0, 2, 5, 8]: 
    LoopAnchorIntersect = []
    for time2 in [0, 2, 5, 8]:
        my_hit = bioframe.overlap(loops_1bins[time1], loops_5bins[time2],how='inner').loc[:,['chrom','start', 'end']].drop_duplicates()
        if time1 != time2:
            LoopAnchorIntersect.append(my_hit.shape[0]/loops_1bins[time1].drop_duplicates().shape[0])
        else:
            LoopAnchorIntersect.append(my_hit.shape[0]/loops_1bins[time1].drop_duplicates().shape[0])
    percent.append(LoopAnchorIntersect)

# pairs_Timing_spread


# plotting the heatmap
hm = sns.heatmap(data=percent,
                annot=True, 
                cmap='coolwarm', 
                vmin=0.5, vmax=1)
plt.savefig("results/pics/paper/LAI_1_in_5.pdf", dpi=100, bbox_inches='tight')
plt.show()
plt.clf()

# %% consecutive standAlone assignment
for time in [0,2,5,8]:
    df_loops = bioframe.read_table('data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bed' % time, schema='bed3')
    loops_df_withDist = bioframe.closest(df_loops)
    loops_df_withDist.query('distance < 2000').to_csv('data/loops_quantifyChromosight/%sAB_consecutive_1binsAdded.bed' % time, sep='\t', index=False, header=False)
    loops_df_withDist.query('distance >= 2000').to_csv('data/loops_quantifyChromosight/%sAB_standAlone_1binsAdded.bed' % time, sep='\t', index=False, header=False)

# %%
loop = pd.read_table('results/0AB_chromosight_quantify.tsv')
loop_filtered = loop.query('score > 0.2 and qvalue < 0.05').reset_index(drop=True)
loop_filtered.loc[:,['chrom1', 'start1', 'end2']].to_csv(
    "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.bed",
                    sep="\t", index=False,
                    header=False)

loop_filtered.to_csv("data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.bedpe", sep="\t", index=False, header=False)

#plot Fig2c
python3 scripts/modules/plot_hierarchy_loop.py