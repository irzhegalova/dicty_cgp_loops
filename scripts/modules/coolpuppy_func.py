import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess

# import libraries for biological data analysis
from coolpuppy import coolpup
from coolpuppy import plotpup
from matplotlib.colors import LogNorm
# from plotpup import plotpup

import sys

sys.path.append("/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/")
from custom_functions import *
import cooltools
import cooler
from cooltools import insulation
import pyBigWig
import bioframe

# %%
resolution = 500
cooler_path = "../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
clr = cooler.Cooler(cooler_path + str(resolution))

# chromsizes
tmp = {"chrom": clr.chromsizes.index.to_list(), "end": [3, 4]}
df_chromsizes = clr.chromsizes.reset_index()
df_chromsizes.columns = ["chrom", "end"]
df_chromsizes.loc[:, "start"] = 0
df_chromsizes["name"] = (
    df_chromsizes["chrom"]
    + ":"
    + df_chromsizes["start"].astype(str)
    + "-"
    + df_chromsizes["end"].astype(str)
)
df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
df_chromsizes["end"] = clr.chromsizes.tolist()
df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

expected = cooltools.expected_cis(
    clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=2
)  # , median=True


# %%
def compute_rescaleSize_zoo(
    cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/",
    resolution=2000,
    nthreads=4,
    mode="bedpe",
    chromsizes_path="data/genome/dicty.chrom.sizes",
    bedpe_path="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    rescale=False,
    local=False,
    score=True,
    by_strand=False,
    coef_flank=3,
    ignore_diags=2,
    rescale_flank=1.0,
    vmax=2.5,
    vmin=0.5,
    organism="nonhuman",
    plot_name="",
):
    clr = cooler.Cooler(cooler_path + str(resolution))

    df_chromsizes = clr.chromsizes.reset_index()  # pd.DataFrame(data=tmp)
    # df_chromsizes = pd.read_csv(chromsizes_path, sep='\t', header=None, )
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

    df_bedpe = pd.read_table(bedpe_path, header=None)  # .iloc[0:20,:]
    if mode == "bed":
        if df_bedpe.shape[1] == 3:
            print("bed is loaded")
        elif df_bedpe.shape[1] == 6:
            df_bedpe.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
            df_bedpe = df_bedpe[["chrom1", "start1", "end2"]]
            print("bed is created")
        df_bedpe.columns = ["chrom", "start", "end"]
        paired_sites = df_bedpe[df_bedpe["end"] - df_bedpe["start"] > resolution * 4]
        median_size = np.median(paired_sites.end - paired_sites.start)
    elif mode == "bedpe":
        if df_bedpe.shape[1] == 3:
            df_bedpe.columns = ["chrom1", "start1", "end2"]
            df_bedpe["end1"] = df_bedpe["start1"] + 2000
            df_bedpe["start2"] = df_bedpe["end2"] - 2000
            df_bedpe = df_bedpe[
                ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
            ]
        elif df_bedpe.shape[1] == 6:
            df_bedpe.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        paired_sites = df_bedpe[df_bedpe["start2"] - df_bedpe["end1"] > resolution * 2]
        median_size = np.median(paired_sites.start2 - paired_sites.end1)
    else:
        raise AssertionError("Not bedpe?")
    if organism == "human":
        expected = pd.read_csv(
            "GSE63525_GM12878_insitu_primary.5kb.expected.tsv", sep="\t"
        )
    else:
        expected = cooltools.expected_cis(
            clr,
            view_df=df_chromsizes,
            nproc=nthreads,
            chunksize=1_000_000,
            ignore_diags=0,
        )

    flank = int((median_size * coef_flank // resolution) * resolution)
    if rescale:
        cc = coolpup.CoordCreator(
            features=paired_sites,
            # flank=flank,
            mindist=resolution * 3,
            local=local,
            resolution=resolution,
            features_format=mode,
            rescale_flank=rescale_flank,
        )

        pu = coolpup.PileUpper(
            clr,
            cc,
            view_df=df_chromsizes,
            expected=expected,
            control=False,
            rescale=True,
            rescale_size=int(1 + flank * 2 // resolution),
            nproc=nthreads,
            ignore_diags=ignore_diags,
        )
    else:
        cc = coolpup.CoordCreator(
            features=paired_sites,
            local=local,
            flank=flank,
            mindist=resolution * 3,
            resolution=resolution,
            features_format=mode,
        )
        # , rescale_flank=1)
        pu = coolpup.PileUpper(
            clr,
            cc,
            view_df=df_chromsizes,
            expected=expected,
            control=False,
            # rescale=True, rescale_size=81,
            nproc=nthreads,
            ignore_diags=ignore_diags,
        )
    pup = pu.pileupsWithControl(nproc=nthreads)

    plotpup.make_heatmap_grid(
        pup,  # cols='separation',
        score=score,
        cmap="coolwarm",
        scale="log",
        sym=False,
        height=3,
        vmax=vmax,
        vmin=vmin,
    )
    plt.savefig(plot_name, dpi=100, bbox_inches="tight")
    plt.show()
    # return pup


# Functions to help with plotting
def pcolormesh_45deg(ax, matrix_c, start=0, resolution=1, *args, **kwargs):
    start_pos_vector = [start + resolution * i for i in range(len(matrix_c) + 1)]
    import itertools

    n = matrix_c.shape[0]
    t = np.array([[1, 0.5], [-1, 0.5]])
    matrix_a = np.dot(
        np.array(
            [
                (i[1], i[0])
                for i in itertools.product(start_pos_vector[::-1], start_pos_vector)
            ]
        ),
        t,
    )
    x = matrix_a[:, 1].reshape(n + 1, n + 1)
    y = matrix_a[:, 0].reshape(n + 1, n + 1)
    im = ax.pcolormesh(x, y, np.flipud(matrix_c), *args, **kwargs)
    im.set_rasterized(True)
    return im


from matplotlib.ticker import EngFormatter

bp_formatter = EngFormatter("b")


def format_ticks(ax, x=True, y=True, rotate=True):
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis="x", rotation=45)


def compute_is(
    cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/",
    resolution=2000,
    chrom="chr2",
    start=50_000,
    organism="dicty",
    blacklist=False,
    blacklist_path="data/hic_zoo/dm6-blacklist.v2.bed",
    long_name=None,
):
    clr = cooler.Cooler(cooler_path + str(resolution))
    windows = [
        40 * resolution,
        80 * resolution,
    ]  # [3 * resolution, 5 * resolution, 10 * resolution, 25 * resolution]
    insulation_table = insulation(clr, windows, verbose=True)
    # df_chromsizes = pd.read_csv(path_chromsizes, sep='\t', header=None, )
    df_chromsizes = clr.chromsizes.reset_index()  # pd.DataFrame(data=tmp)
    df_chromsizes.columns = ["chrom", "length"]
    for window in windows:
        insulation_table["z_value_log2_insulation_score_" + str(window)] = (
            insulation_table["log2_insulation_score_" + str(window)]
            - np.nanmean(
                insulation_table["log2_insulation_score_" + str(window)][
                    np.isfinite(
                        insulation_table["log2_insulation_score_" + str(window)]
                    )
                ]
            )
            / np.nanstd(
                insulation_table["log2_insulation_score_" + str(window)][
                    np.isfinite(
                        insulation_table["log2_insulation_score_" + str(window)]
                    )
                ]
            )
        )
        if long_name is None:
            long_name = organism + ".IS."
        long_name = (
            long_name + str(window // 1000) + "k.bin" + str(resolution // 1000) + "k"
        )
        bw = pyBigWig.open("bw/" + long_name + ".bw", "w")
        bw.addHeader(list(df_chromsizes.itertuples(index=False, name=None)), maxZooms=0)
        chroms = insulation_table.chrom.tolist()
        starts = insulation_table.start.tolist()
        ends = insulation_table.end.tolist()
        values0 = insulation_table.loc[
            :, "log2_insulation_score_" + str(window)
        ].tolist()
        bw.addEntries(chroms, starts, ends=ends, values=values0)
        bw.close()
        if blacklist:
            bashCommand = (
                "multiBigwigSummary bins -b bw/"
                + long_name
                + ".bw -bs 1000 -p 4 -o \
            bw/"
                + long_name
                + ".1kb.npz \
            --outRawCounts bw/"
                + long_name
                + ".blacklisted.tsv \
            --blackListFileName "
                + blacklist_path
            )
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            # return (output)

            # import time
            # time.sleep(10)

            is_1kb = pd.read_table("bw/" + long_name + ".blacklisted.tsv")
            is_1kb.columns = ["chrom", "start", "end", "value"]
            is_1kb = is_1kb.sort_values(by=["chrom", "start"])
            bw = pyBigWig.open("bw/" + long_name + ".blacklisted.bw", "w")
            df_chromsizes = clr.chromsizes.reset_index()  # .sort_values(by='name')
            df_chromsizes.columns = ["chrom", "end"]
            bw.addHeader(
                list(df_chromsizes.itertuples(index=False, name=None)), maxZooms=0
            )
            chroms = is_1kb.chrom.tolist()
            starts = is_1kb.start.tolist()
            ends = is_1kb.end.tolist()
            values0 = is_1kb.value.tolist()
            bw.addEntries(chroms, starts, ends=ends, values=values0)
            bw.close()


def plot_ConvPairs(
    cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/",
    resolution=500,
    nthreads=4,
    mode="bedpe",
    chromsizes_path="data/genome/dicty.chrom.sizes",
    bedpe_path="data/genome/genesConvPairs.byExpr.byAnchors.3.bed",
    ignore_diags=2,
    rescale_flank=10,
    vmax=2.5,
    vmin=0.5,
    flank=50000,
    group_name="both high",
    expected=None,
):
    clr = cooler.Cooler(cooler_path + str(resolution))

    df_chromsizes = clr.chromsizes.reset_index()
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

    df_bedpe = (
        bioframe.read_table(bedpe_path, schema="bed")
        .query(f"chrom in {clr.chromnames}")
        #.query('name == "high & low" and score == "loop anchor"')
    )
    if df_bedpe.shape[0] == 0:
        return [np.nan]
    # df_bedpe.chrom = df_bedpe.chrom.astype(str)
    df_bedpe.start = df_bedpe.start.astype(np.int64)
    df_bedpe.end = df_bedpe.end.astype(np.int64)

    paired_sites = df_bedpe  # [df_bedpe['end'] - df_bedpe['start'] > 3 * resolution][df_bedpe['end'] - df_bedpe['start'] < np.quantile(df_bedpe['end'] - df_bedpe['start'], 0.99)]
    # median_size = np.median(paired_sites.end - paired_sites.start)

    # expected = cooltools.expected_cis(clr, view_df=df_chromsizes,
    #                                 nproc=4,
    #                                 chunksize=1_000_000,
    #                                 ignore_diags=2) #, median=True

    pup = coolpup.pileup(
        clr,
        paired_sites,  # groupby=['name1', 'score2'], #['group1'], #'length_group2'
        rescale=True,
        rescale_size=int(1 + flank * 2 // resolution),
        rescale_flank=10,
        local=True,
        ooe=True,
        features_format="bed",
        view_df=df_chromsizes,
        nproc=4,
        # by_strand=True,
        flip_negative_strand=True,
        expected_df=expected,
        flank=flank,
        min_diag=2,
    )

    mtx = pup.data[
        0
    ]  # [pup.index[(pup.name1 == group_name) & (pup.score2 == "loop anchor")][0]]

    to_plot = mtx[20:76, 106:116]  # [20:80,105:115] #np.triu(np.rot90(, 3), 10)
    return to_plot


def plot_pairs(
    cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/",
    resolution=500,
    nthreads=4,
    mode="bedpe",
    chromsizes_path="data/genome/dicty.chrom.sizes",
    bedpe_path="data/genome/genesConvPairs.byExpr.byAnchors.3.bed",
    ignore_diags=2,
    rescale_flank=10,
    vmax=2.5,
    vmin=0.5,
    flank=50000,
    group_name="both high",
):
    clr = cooler.Cooler(cooler_path + str(resolution))

    df_chromsizes = clr.chromsizes.reset_index()
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

    df_bedpe = bioframe.read_table(bedpe_path, schema="bed").query(
        f"chrom in {clr.chromnames}"
    )
    # df_bedpe.chrom = df_bedpe.chrom.astype(str)
    df_bedpe.start = df_bedpe.start.astype(np.int64)
    df_bedpe.end = df_bedpe.end.astype(np.int64)
    paired_sites = df_bedpe
    # paired_sites = df_bedpe[df_bedpe["end"] - df_bedpe["start"] > 4 * resolution][df_bedpe["end"] - df_bedpe["start"] < np.quantile(df_bedpe["end"] - df_bedpe["start"], 0.99)]
    # median_size = np.median(paired_sites.end - paired_sites.start)

    expected = cooltools.expected_cis(
        clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000, ignore_diags=2
    )

    pup = coolpup.pileup(
        clr,
        paired_sites,
        groupby=["name1", "score2"],  # ['group1'], #'length_group2'
        rescale=True,
        rescale_size=int(1 + flank * 2 // resolution),
        rescale_flank=rescale_flank,
        local=True,
        features_format="bed",
        view_df=df_chromsizes,
        nproc=4,
        # by_strand=True,
        flip_negative_strand=True,
        expected_df=expected,
        flank=flank,
        min_diag=2,
    )

    return pup


def find_exprThres(
    conv=True,
    thres_low=1,
    thres_high=8,
    length_high=False,
    length_low=False,
    step=10,
    intervals_high=False,
    intervals_low=False,
    return_df=False,
):
    tpm_df = create_rnaseq_df(need_coord=True)
    # tpm_thres = np.quantile(tpm_df["0AB"], [0.33, 0.66]).tolist()
    tpm_df = tpm_df  # .query('end - start < 20000 or end - start > 500')

    tpm_df_toMerge = (
        tpm_df[["name", "0AB", "2AB", "5AB", "8AB"]]
        .set_index("name")
        .rename(columns={"0AB": "0AB-", "2AB": "2AB-", "5AB": "5AB-", "8AB": "8AB-"})
    )

    # create conv pairs
    genes_minus = bioframe.read_table("data/genome/genes.minus.bed", schema="bed")
    genes_plus = bioframe.read_table("data/genome/genes.plus.bed", schema="bed")

    genes_plus_pairs = bioframe.closest(
        genes_plus,
        genes_minus,
        ignore_upstream=True,
        ignore_overlaps=True,
        direction_col="strand",
        suffixes=("+", "-"),
    )

    genes_minus_pairs = bioframe.closest(
        genes_minus,
        genes_plus,
        ignore_upstream=True,
        ignore_overlaps=True,
        direction_col="strand",
        suffixes=("-", "+"),
    )

    conv_pairs = pd.merge(
        genes_minus_pairs.loc[:, ["name-", "name+"]],
        genes_plus_pairs,
        on=["name-", "name+"],
        how="inner",
    )
    # get bin where middle points (equal distance to both genes ends) are located
    conv_pairs = conv_pairs.query("distance < 8000")
    conv_pairs["middleInter_bin"] = (
        (conv_pairs["end+"] + np.round(conv_pairs["distance"] / 2, 0)) // 2000 * 2000
    )
    conv_pairs["middleInter_bin_500"] = (
        (conv_pairs["end+"] + np.round(conv_pairs["distance"] / 2, 0)) // 500 * 500
    )
    conv_pairs["middleInter_bin_end"] = conv_pairs["middleInter_bin"] + 2000
    conv_pairs["middleInter_bin_end_500"] = conv_pairs["middleInter_bin"] + 500

    conv_pairs["start"] = conv_pairs[["start+", "start-", "end+", "end-"]].min(axis=1)
    conv_pairs["end"] = conv_pairs[["start+", "start-", "end+", "end-"]].max(axis=1)

    conv_pairs_TPM = (
        tpm_df_toMerge.join(conv_pairs.set_index("name-"), how="inner")
        .reset_index()
        .rename(columns={"index": "name-"})
    )
    tpm_df_toMerge = (
        tpm_df[["name", "0AB", "2AB", "5AB", "8AB"]]
        .set_index("name")
        .rename(columns={"0AB": "0AB+", "2AB": "2AB+", "5AB": "5AB+", "8AB": "8AB+"})
    )
    conv_pairs_TPM = (
        tpm_df_toMerge.join(conv_pairs_TPM.set_index("name+"), how="inner")
        .reset_index()
        .rename(columns={"index": "name+"})
    )

    is_table = bioframe.read_table(
        "bw/IS/0AB.20k.IS.cooltools.bedgraph", schema="bedGraph"
    )
    # bigWigToBedGraph bw/dicty.IS.5k.bw bw/dicty.IS.5k.bg
    # is_table_500 = bioframe.read_table("bw/dicty.IS.5k.bg", schema="bedGraph")

    conv_pairs_TPM_IS = pd.merge(
        conv_pairs_TPM,
        is_table,
        left_on=["chrom+", "middleInter_bin"],
        right_on=["chrom", "start"],
    )
    conv_pairs_TPM_IS["log2_0AB+"] = np.log2(conv_pairs_TPM_IS["0AB+"] + 0.01)
    conv_pairs_TPM_IS["log2_0AB-"] = np.log2(conv_pairs_TPM_IS["0AB-"] + 0.01)

    conv_pairs_TPM_IS["start"] = conv_pairs_TPM_IS[
        ["start+", "start-", "end+", "end-"]
    ].min(axis=1)
    conv_pairs_TPM_IS["end"] = conv_pairs_TPM_IS[
        ["start+", "start-", "end+", "end-"]
    ].max(axis=1)

    conv_pairs_TPM_IS = assign_expr_groups(
        conv_pairs_TPM_IS,
        thres_high=thres_high,
        thres_low=thres_low,
        step=step,
        intervals_low=intervals_low,
        intervals_high=intervals_high,
    )

    if length_high:
        conv_pairs_TPM_IS["length-"] = (
            conv_pairs_TPM_IS["end-"] - conv_pairs_TPM_IS["start-"]
        )
        conv_pairs_TPM_IS["length+"] = (
            conv_pairs_TPM_IS["end+"] - conv_pairs_TPM_IS["start+"]
        )
        length_high = conv_pairs_TPM_IS.query(
            'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "+"'
        )["length+"].tolist()
        length_high.extend(
            conv_pairs_TPM_IS.query(
                'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "-"'
            )["length-"].tolist()
        )
        return np.median(length_high)

    if length_low:
        conv_pairs_TPM_IS["length-"] = (
            conv_pairs_TPM_IS["end-"] - conv_pairs_TPM_IS["start-"]
        )
        conv_pairs_TPM_IS["length+"] = (
            conv_pairs_TPM_IS["end+"] - conv_pairs_TPM_IS["start+"]
        )

        length_low = conv_pairs_TPM_IS.query(
            'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "+"'
        )["length-"].tolist()
        length_low.extend(
            conv_pairs_TPM_IS.query(
                'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "-"'
            )["length+"].tolist()
        )
        return np.median(length_low)

    if intervals_low:
        suffix = "notZero"
    else:
        suffix = "with0"
    if not intervals_high:
        suffix += "HighToInf"
    if conv:
        prefix = "Conv"
    else:
        prefix = "Div"

    if return_df:
        return conv_pairs_TPM_IS
    else:
        conv_pairs_TPM_IS.loc[
            :, ["chrom+", "start", "end", "expr_group", "anchor_group", "strand+"]
        ].to_csv(
            "data/genome/genes%sPairs.byExpr.byAnchors.thr_low%s.thres_high%s.%s.step%s.bed"
            % (prefix, thres_low, thres_high, suffix, step),
            index=False,
            sep="\t",
            header=False,
        )  # .query('expr_group == "high & low" and anchor_group == "loop anchor"')
        # conv_pairs_right = conv_pairs_TPM_IS.loc[conv_pairs_TPM_IS['strand+'] == '+', :]
        # data = {'chrom': conv_pairs_right['chrom+'],
        #         'start': conv_pairs_right.end + 10000,
        #         'end': conv_pairs_right.end + 10001,
        #         'expr_group': conv_pairs_right.expr_group,
        #         'anchor_group': conv_pairs_right.anchor_group,
        #         'strand+': conv_pairs_right['strand+']}
        # df_bed_right = pd.DataFrame(data)

        # conv_pairs_toFlip = conv_pairs_TPM_IS.loc[conv_pairs_TPM_IS['strand+'] == '-', :]
        # data = {'chrom': conv_pairs_toFlip['chrom+'],
        #         'start': conv_pairs_toFlip.start - 10001,
        #         'end': conv_pairs_toFlip.start - 10000,
        #         'expr_group': conv_pairs_toFlip.expr_group,
        #         'anchor_group': conv_pairs_toFlip.anchor_group,
        #         'strand+': conv_pairs_toFlip['strand+']}
        # df_bed_toFlip = pd.DataFrame(data)

        # # Create DataFrame
        # df_bed = pd.concat([df_bed_toFlip, df_bed_right])
        # df_bed.to_csv("data/genome/downstream20kb_fromConvPairs.byExpr.byAnchors.thr_low%s.thres_high%s.bed" % (thres_low, thres_high),
        # index=False, sep = '\t', header=False)


def assign_expr_groups(
    conv_pairs_TPM_IS=None,
    thres_high=None,
    thres_low=None,
    step=10,
    intervals_low=False,
    intervals_high=False,
):
    # assign expression groups
    conv_pairs_TPM_IS["expr_group"] = "both low"

    if not intervals_low and not intervals_high:
        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB+"] > thres_high),
            "expr_group",
        ] = "both high"
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] < 2) & (conv_pairs_TPM_IS['0AB+'] < 2), 'expr_group'] = 'both low'
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
        conv_pairs_TPM_IS.loc[
            (
                (conv_pairs_TPM_IS["0AB-"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB+"] > thres_high)
            )
            | (
                (conv_pairs_TPM_IS["0AB+"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB-"] > thres_high)
            ),
            "expr_group",
        ] = "high & low"

        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB+"] <= thres_low)
            & (conv_pairs_TPM_IS["0AB-"] > thres_high),
            "strand+",
        ] = "-"
    elif intervals_high and not intervals_low:
        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB+"] > thres_high),
            "expr_group",
        ] = "both high"
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] < 2) & (conv_pairs_TPM_IS['0AB+'] < 2), 'expr_group'] = 'both low'
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
        conv_pairs_TPM_IS.loc[
            (
                (conv_pairs_TPM_IS["0AB-"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB-"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB+"] > thres_high)
                & (conv_pairs_TPM_IS["0AB+"] < thres_high + step)
            )
            | (
                (conv_pairs_TPM_IS["0AB+"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB-"] > thres_high)
                & (conv_pairs_TPM_IS["0AB-"] < thres_high + step)
            ),
            "expr_group",
        ] = "high & low"

        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB+"] <= thres_low)
            & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
            & (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB-"] < thres_high + step),
            "strand+",
        ] = "-"
    elif not intervals_high and intervals_low:
        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB+"] > thres_high),
            "expr_group",
        ] = "both high"
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] < 2) & (conv_pairs_TPM_IS['0AB+'] < 2), 'expr_group'] = 'both low'
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
        conv_pairs_TPM_IS.loc[
            (
                (conv_pairs_TPM_IS["0AB-"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB-"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB+"] > thres_high)
            )
            | (
                (conv_pairs_TPM_IS["0AB+"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB-"] > thres_high)
            ),
            "expr_group",
        ] = "high & low"

        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB+"] <= thres_low)
            & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
            & (conv_pairs_TPM_IS["0AB-"] > thres_high),
            "strand+",
        ] = "-"
    elif intervals_high and intervals_low:
        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB+"] > thres_high),
            "expr_group",
        ] = "both high"
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] < 2) & (conv_pairs_TPM_IS['0AB+'] < 2), 'expr_group'] = 'both low'
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
        conv_pairs_TPM_IS.loc[
            (
                (conv_pairs_TPM_IS["0AB-"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB-"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB+"] > thres_high)
                & (conv_pairs_TPM_IS["0AB+"] < thres_high + step)
            )
            | (
                (conv_pairs_TPM_IS["0AB+"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB-"] > thres_high)
                & (conv_pairs_TPM_IS["0AB-"] < thres_high + step)
            ),
            "expr_group",
        ] = "high & low"

        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB+"] <= thres_low)
            & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
            & (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB-"] < thres_high + step),
            "strand+",
        ] = "-"

    # conv_pairs_TPM_IS_high.loc[:,['chrom+', 'middleInter_bin', 'middleInter_bin_end', 'name+', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.highExpr.binned.bed" ,
    #     index=False, sep = '\t', header=False)

    # conv_pairs_TPM_IS_high.loc[:,['chrom+', 'start', 'end', 'name+', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.highExpr.bed",
    # index=False, sep = '\t', header=False)

    # conv_pairs_TPM_IS.loc[:,['chrom+', 'start', 'end', 'expr_group', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.byExpr.bed",
    # index=False, sep = '\t', header=False)

    loop_path = "data/loops_quantifyChromosight/0AB_mergedAnchors.q=02.0bins_added.bed"
    # loop_path = 'data/loops_quantifyChromosight/0_8_common_anchors.q=02.1bins_added.bed'
    loop_anchors = bioframe.read_table(loop_path, schema="bed3")

    tmp = conv_pairs_TPM_IS[["chrom+", "start", "end"]]
    tmp["start"] = tmp["start"].astype(int)
    tmp["end"] = tmp["end"].astype(int)
    tmp.columns = ["chrom", "start", "end"]
    ConvPairs_inAnchors_ind = bioframe.overlap(
        tmp, loop_anchors, how="inner", return_input=False, return_index=True
    )["index"].tolist()
    ConvPairs_inAnchors_indUniq = np.unique(ConvPairs_inAnchors_ind)
    conv_pairs_TPM_IS["anchor_group"] = "none"
    conv_pairs_TPM_IS.loc[:, "anchor_group"][
        ConvPairs_inAnchors_indUniq
    ] = "loop anchor"
    return conv_pairs_TPM_IS


def create_DivPairs_exprThres(
    thres_low=1,
    thres_high=8,
    intervals_low=False,
    intervals_high=False,
    length_high=False,
    length_low=False,
    step=10,
    return_df=False,
):
    tpm_df = create_rnaseq_df(need_coord=True)
    # tpm_thres = np.quantile(tpm_df["0AB"], [0.33, 0.66]).tolist()
    tpm_df = tpm_df#.query("end - start < 10000 or end - start > 1000")

    tpm_df_toMerge = (
        tpm_df[["name", "0AB", "2AB", "5AB", "8AB"]]
        .set_index("name")
        .rename(columns={"0AB": "0AB-", "2AB": "2AB-", "5AB": "5AB-", "8AB": "8AB-"})
    )

    # create conv pairs
    genes_minus = bioframe.read_table("data/genome/genes.minus.bed", schema="bed")
    genes_plus = bioframe.read_table("data/genome/genes.plus.bed", schema="bed")

    genes_plus_pairs = bioframe.closest(
        genes_plus,
        genes_minus,
        ignore_downstream=True,
        ignore_overlaps=True,
        direction_col="strand",
        suffixes=("+", "-"),
    )

    genes_minus_pairs = bioframe.closest(
        genes_minus,
        genes_plus,
        ignore_downstream=True,
        ignore_overlaps=True,
        direction_col="strand",
        suffixes=("-", "+"),
    )

    conv_pairs = pd.merge(
        genes_minus_pairs.loc[:, ["name-", "name+"]],
        genes_plus_pairs,
        on=["name-", "name+"],
        how="inner",
    )
    # get bin where middle points (equal distance to both genes ends) are located
    conv_pairs = conv_pairs.query("distance < 8000")
    # conv_pairs['middleInter_bin'] = (conv_pairs['end+'] + np.round(conv_pairs['distance']/2,0))//2000 * 2000
    # conv_pairs['middleInter_bin_500'] = (conv_pairs['end+'] + np.round(conv_pairs['distance']/2,0))//500 * 500
    # conv_pairs['middleInter_bin_end'] = conv_pairs['middleInter_bin'] + 2000
    # conv_pairs['middleInter_bin_end_500'] = conv_pairs['middleInter_bin'] + 500

    conv_pairs["start"] = conv_pairs[["start+", "start-", "end+", "end-"]].min(axis=1)
    conv_pairs["end"] = conv_pairs[["start+", "start-", "end+", "end-"]].max(axis=1)

    conv_pairs_TPM = (
        tpm_df_toMerge.join(conv_pairs.set_index("name-"), how="inner")
        .reset_index()
        .rename(columns={"index": "name-"})
    )
    tpm_df_toMerge = (
        tpm_df[["name", "0AB", "2AB", "5AB", "8AB"]]
        .set_index("name")
        .rename(columns={"0AB": "0AB+", "2AB": "2AB+", "5AB": "5AB+", "8AB": "8AB+"})
    )
    conv_pairs_TPM = (
        tpm_df_toMerge.join(conv_pairs_TPM.set_index("name+"), how="inner")
        .reset_index()
        .rename(columns={"index": "name+"})
    )

    # is_table = bioframe.read_table('bw/IS/0AB.20k.IS.cooltools.bedgraph', schema='bedGraph')
    # #bigWigToBedGraph bw/dicty.IS.5k.bw bw/dicty.IS.5k.bg
    # is_table_500 = bioframe.read_table('bw/dicty.IS.5k.bg', schema='bedGraph')

    # conv_pairs_TPM_IS = pd.merge(conv_pairs_TPM, is_table, left_on=['chrom+', 'middleInter_bin'], right_on=['chrom', 'start'])
    conv_pairs_TPM_IS = conv_pairs_TPM
    conv_pairs_TPM_IS["log2_0AB+"] = np.log2(conv_pairs_TPM_IS["0AB+"] + 0.01)
    conv_pairs_TPM_IS["log2_0AB-"] = np.log2(conv_pairs_TPM_IS["0AB-"] + 0.01)

    conv_pairs_TPM_IS["start"] = conv_pairs_TPM_IS[
        ["start+", "start-", "end+", "end-"]
    ].min(axis=1)
    conv_pairs_TPM_IS["end"] = conv_pairs_TPM_IS[
        ["start+", "start-", "end+", "end-"]
    ].max(axis=1)

    # assign expression groups
    conv_pairs_TPM_IS["expr_group"] = "base"

    if not intervals_low:
        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB+"] > thres_high),
            "expr_group",
        ] = "both high"
        conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] <= thres_low) & (conv_pairs_TPM_IS['0AB+'] <= thres_low), 'expr_group'] = 'both low'
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
        conv_pairs_TPM_IS.loc[
            (
                (conv_pairs_TPM_IS["0AB-"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB+"] > thres_high)
            )
            | (
                (conv_pairs_TPM_IS["0AB+"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB-"] > thres_high)
            ),
            "expr_group",
        ] = "high & low"

        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB+"] <= thres_low)
            & (conv_pairs_TPM_IS["0AB-"] > thres_high),
            "strand+",
        ] = "-"
    else:
        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB+"] > thres_high),
            "expr_group",
        ] = "both high"
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] < 2) & (conv_pairs_TPM_IS['0AB+'] < 2), 'expr_group'] = 'both low'
        # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
        conv_pairs_TPM_IS.loc[
            (
                (conv_pairs_TPM_IS["0AB-"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB-"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB+"] > thres_high)
                & (conv_pairs_TPM_IS["0AB+"] < thres_high + step)
            )
            | (
                (conv_pairs_TPM_IS["0AB+"] <= thres_low)
                & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
                & (conv_pairs_TPM_IS["0AB-"] > thres_high)
                & (conv_pairs_TPM_IS["0AB-"] < thres_high + step)
            ),
            "expr_group",
        ] = "high & low"

        conv_pairs_TPM_IS.loc[
            (conv_pairs_TPM_IS["0AB+"] <= thres_low)
            & (conv_pairs_TPM_IS["0AB+"] > thres_low - step)
            & (conv_pairs_TPM_IS["0AB-"] > thres_high)
            & (conv_pairs_TPM_IS["0AB-"] < thres_high + step),
            "strand+",
        ] = "-"

    # conv_pairs_TPM_IS_high.loc[:,['chrom+', 'middleInter_bin', 'middleInter_bin_end', 'name+', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.highExpr.binned.bed" ,
    #     index=False, sep = '\t', header=False)

    # conv_pairs_TPM_IS_high.loc[:,['chrom+', 'start', 'end', 'name+', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.highExpr.bed",
    # index=False, sep = '\t', header=False)

    # conv_pairs_TPM_IS.loc[:,['chrom+', 'start', 'end', 'expr_group', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.byExpr.bed",
    # index=False, sep = '\t', header=False)

    loop_path = "data/loops_quantifyChromosight/0AB_mergedAnchors.q=02.0bins_added.bed"
    # loop_path = 'data/loops_quantifyChromosight/0_8_common_anchors.q=02.1bins_added.bed'
    loop_anchors = bioframe.read_table(loop_path, schema="bed3")

    tmp = conv_pairs_TPM_IS[["chrom+", "start", "end"]]
    tmp["start"] = tmp["start"].astype(int)
    tmp["end"] = tmp["end"].astype(int)
    tmp.columns = ["chrom", "start", "end"]
    ConvPairs_inAnchors_ind = bioframe.overlap(
        tmp, loop_anchors, how="inner", return_input=False, return_index=True
    )["index"].tolist()
    ConvPairs_inAnchors_indUniq = np.unique(ConvPairs_inAnchors_ind)
    conv_pairs_TPM_IS["anchor_group"] = "none"
    conv_pairs_TPM_IS.loc[:, "anchor_group"][
        ConvPairs_inAnchors_indUniq
    ] = "loop anchor"

    if length_high:
        conv_pairs_TPM_IS["length-"] = (
            conv_pairs_TPM_IS["end-"] - conv_pairs_TPM_IS["start-"]
        )
        conv_pairs_TPM_IS["length+"] = (
            conv_pairs_TPM_IS["end+"] - conv_pairs_TPM_IS["start+"]
        )
        length_high = conv_pairs_TPM_IS.query(
            'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "+"'
        )["length+"].tolist()
        length_high.extend(
            conv_pairs_TPM_IS.query(
                'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "-"'
            )["length-"].tolist()
        )
        return np.median(length_high)

    if length_low:
        conv_pairs_TPM_IS["length-"] = (
            conv_pairs_TPM_IS["end-"] - conv_pairs_TPM_IS["start-"]
        )
        conv_pairs_TPM_IS["length+"] = (
            conv_pairs_TPM_IS["end+"] - conv_pairs_TPM_IS["start+"]
        )

        length_low = conv_pairs_TPM_IS.query(
            'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "+"'
        )["length-"].tolist()
        length_low.extend(
            conv_pairs_TPM_IS.query(
                'expr_group == "high & low" and anchor_group == "loop anchor" and `strand+` == "-"'
            )["length+"].tolist()
        )
        return np.median(length_low)
    if return_df:
        return conv_pairs_TPM_IS
    if not intervals_low:
        conv_pairs_TPM_IS.loc[
            :, ["chrom+", "start", "end", "expr_group", "anchor_group", "strand+"]
        ].to_csv(
            "data/genome/genesDivPairs.byExpr.byAnchors.thr_low%s.thres_high%s.bed"
            % (thres_low, thres_high),
            index=False,
            sep="\t",
            header=False,
        )
        # .query('expr_group == "high & low" and anchor_group == "loop anchor"')

        conv_pairs_right = conv_pairs_TPM_IS.loc[conv_pairs_TPM_IS["strand+"] == "+", :]
        data = {
            "chrom": conv_pairs_right["chrom+"],
            "start": conv_pairs_right.end + 10000,
            "end": conv_pairs_right.end + 10001,
            "expr_group": conv_pairs_right.expr_group,
            "anchor_group": conv_pairs_right.anchor_group,
            "strand+": conv_pairs_right["strand+"],
        }
        df_bed_right = pd.DataFrame(data)

        conv_pairs_toFlip = conv_pairs_TPM_IS.loc[
            conv_pairs_TPM_IS["strand+"] == "-", :
        ]
        data = {
            "chrom": conv_pairs_toFlip["chrom+"],
            "start": conv_pairs_toFlip.start - 10001,
            "end": conv_pairs_toFlip.start - 10000,
            "expr_group": conv_pairs_toFlip.expr_group,
            "anchor_group": conv_pairs_toFlip.anchor_group,
            "strand+": conv_pairs_toFlip["strand+"],
        }
        df_bed_toFlip = pd.DataFrame(data)

        # Create DataFrame
        df_bed = pd.concat([df_bed_toFlip, df_bed_right])
        df_bed.to_csv(
            "data/genome/downstream20kb_fromDivairs.byExpr.byAnchors.thr_low%s.thres_high%s.bed"
            % (thres_low, thres_high),
            index=False,
            sep="\t",
            header=False,
        )
    else:
        conv_pairs_TPM_IS.loc[
            :, ["chrom+", "start", "end", "expr_group", "anchor_group", "strand+"]
        ].to_csv(
            "data/genome/genesDivPairs.byExpr.byAnchors.thr_low%s.thres_high%s.notZero.step%s.bed"
            % (thres_low, thres_high, step),
            index=False,
            sep="\t",
            header=False,
        )
        # query('expr_group == "high & low" and anchor_group == "loop anchor"')

    # the rest of code
    # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > thres) & (conv_pairs_TPM_IS['0AB+'] > thres), 'expr_group'] = 'both high'
    # # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] < 2) & (conv_pairs_TPM_IS['0AB+'] < 2), 'expr_group'] = 'both low'
    # # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB-'] > 2) & (conv_pairs_TPM_IS['0AB+'] > 2) & (conv_pairs_TPM_IS['0AB-'] < 8) & (conv_pairs_TPM_IS['0AB+'] < 8), 'expr_group'] = 'both av'
    # conv_pairs_TPM_IS.loc[((conv_pairs_TPM_IS['0AB-'] <= thres) & (conv_pairs_TPM_IS['0AB+'] > thres)) | ((conv_pairs_TPM_IS['0AB+'] <= thres) & (conv_pairs_TPM_IS['0AB-'] > thres)), 'expr_group'] = 'high & low'

    # conv_pairs_TPM_IS.loc[(conv_pairs_TPM_IS['0AB+'] <= thres) & (conv_pairs_TPM_IS['0AB-'] > thres), 'strand+'] = '-'

    # # conv_pairs_TPM_IS_high.loc[:,['chrom+', 'middleInter_bin', 'middleInter_bin_end', 'name+', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.highExpr.binned.bed" ,
    # #     index=False, sep = '\t', header=False)

    # # conv_pairs_TPM_IS_high.loc[:,['chrom+', 'start', 'end', 'name+', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.highExpr.bed",
    # # index=False, sep = '\t', header=False)

    # # conv_pairs_TPM_IS.loc[:,['chrom+', 'start', 'end', 'expr_group', 'score+', 'strand+']].to_csv("data/genome/genesConvPairs.byExpr.bed",
    # # index=False, sep = '\t', header=False)

    # #loop_path = 'data/loops_quantifyChromosight/0AB_mergedAnchors.q=02.0bins_added.bed'
    # loop_path = 'data/loops_quantifyChromosight/0_8_common_anchors.q=02.1bins_added.bed'
    # loop_anchors = bioframe.read_table(loop_path, schema='bed3')

    # tmp = conv_pairs_TPM_IS[['chrom+', 'start', 'end']]
    # tmp['start'] = tmp['start'].astype(int)
    # tmp['end'] = tmp['end'].astype(int)
    # tmp.columns = ['chrom', 'start', 'end']
    # ConvPairs_inAnchors_ind = bioframe.overlap(tmp, loop_anchors, how='inner', return_input=False, return_index=True)['index'].tolist()
    # ConvPairs_inAnchors_indUniq = np.unique(ConvPairs_inAnchors_ind)
    # conv_pairs_TPM_IS['anchor_group'] = 'none'
    # conv_pairs_TPM_IS.loc[:,'anchor_group'][ConvPairs_inAnchors_indUniq] = 'loop anchor'

    # conv_pairs_TPM_IS.loc[:,['chrom+', 'start', 'end', 'expr_group', 'anchor_group', 'strand+']].to_csv("data/genome/genesDivPairs.byExpr.byAnchors.thr%s.bed" % (thres),
    # index=False, sep = '\t', header=False)

def compute_pixel(paired_sites, clr, expected, df_chromsizes, flank=10000, nthreads=4, remove_na=True, cell_cycle=False, pad=1, use_max=False, use_oe=True, use_localExpected=False, use_mean=False):
    """
    Call _create_stack and then select the "main" pixel either by searching in the window or using mean etc
    use median by default
    
    Returns
    -------
    [123]D (?) array with float middle pixel
    """
    
    resolution = clr.binsize
    stack = cooltools.pileup(clr, paired_sites, view_df=df_chromsizes, expected_df=expected, flank=flank)
    if not use_oe:
        if use_localExpected:
            which_middle = flank // resolution
            if use_max:
                max_pixel = np.amax(
                    np.amax(stack[which_middle - 1:which_middle + 2, which_middle - 1:which_middle + 2, :], axis=0),
                    axis=0)
                # add local norm
                max_pixel_norm = max_pixel / np.nanmean(np.nanmean(stack, axis=0), axis=0)
                if remove_na:
                    max_pixel_norm = max_pixel_norm[~np.isnan(max_pixel_norm)]
                return (max_pixel_norm)
            else:
                middle_pixel = stack[which_middle, which_middle, :]
                # add local norm
                middle_pixel_norm = middle_pixel / np.nanmean(np.nanmean(stack, axis=0), axis=0)
                if remove_na:
                    middle_pixel = middle_pixel[~np.isnan(middle_pixel)]
                return (middle_pixel_norm)

    which_middle = flank // resolution
    if use_max:
        max_pixel = np.amax(
            np.amax(stack[which_middle - pad:which_middle + 1+pad, which_middle - pad:which_middle + 1+pad, :], axis=0),
            axis=0)
        if remove_na:
            max_pixel = max_pixel[~np.isnan(max_pixel)]
        return (max_pixel)
    elif use_mean:
        max_pixel = np.nanmean(
            np.nanmean(stack[which_middle - pad:which_middle + 1+pad, which_middle - pad:which_middle + 1+pad, :], axis=0),
            axis=0)
        if remove_na:
            max_pixel = max_pixel[~np.isnan(max_pixel)]
        return (max_pixel)
    else:
        middle_pixel = np.nanmedian(
            np.nanmedian(stack[which_middle - pad:which_middle + 1+pad, which_middle - pad:which_middle + 1+pad, :], axis=0),
            axis=0)
        if remove_na:
            middle_pixel = middle_pixel[~np.isnan(middle_pixel)]
        return (middle_pixel)
    
def add_heatmap_custom(
    data,
    flank,
    rescale,
    rescale_flank,
    n,
    max_coordinates,
    height=1,
    aspect="auto",
    color=None,
    cmap="coolwarm",
    norm=LogNorm(0.5, 2),
    plot_ticks=False,
    stripe=False,
    font_scale=1,
    ax=None
):
    """
    Adds the array contained in data.values[0] to the current axes as a heatmap of stripes
    """
    if len(data) > 1:
        raise ValueError(
            "Multiple pileups for one of the conditions, ensure unique correspondence for each col/row combination"
        )
    elif len(data) == 0:
        return
    flank = int(flank)
    # ax = plt.gca()
    if stripe:
        extents = [-flank / max_coordinates[1], flank / max_coordinates[1], -int(n), 0]
        ax.set_ylim(-max_coordinates[2], 0)
    else:
        extents = [
            -flank / max_coordinates[1],
            flank / max_coordinates[1],
            -flank / max_coordinates[1],
            flank / max_coordinates[1],
        ]
        ax.set_xlim(-(flank / max_coordinates[1]), (flank / max_coordinates[1]))
        ax.set_ylim(-(flank / max_coordinates[1]), (flank / max_coordinates[1]))
    ax.imshow(
        data.values[0],
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation="none",
        extent=extents,
    )
    if plot_ticks:
        # ax.tick_params(
        #     axis="both",
        #     which="major",
        #     labelsize=font_scale * (4.94 + height),
        #     width=1 + (height / 2),
        #     length=1 + height,
        # )

        # if not rescale.any():
        ticks_pixels = np.linspace(
            -max_coordinates[0] / max_coordinates[1],
            max_coordinates[0] / max_coordinates[1],
            3,
        )
        ticks_kbp = ((ticks_pixels * max_coordinates[1]) // 1000).astype(int)
        # ax.xticks(ticks_pixels.tolist(), ticks_kbp.tolist())
        ax.set_xticks(ticks_pixels.tolist())
        ax.set_xticklabels(ticks_kbp.tolist())
        
        if not stripe:
            # ax.yticks(ticks_pixels.tolist(), ticks_kbp.tolist())
            ax.set_yticks(ticks_pixels.tolist())
            ax.set_yticklabels(ticks_kbp.tolist())
        else:
            ticks_n = np.floor(np.linspace(0, -max_coordinates[2], 5)).astype(int)
            ax.yticks(ticks_n, np.append("", np.negative(ticks_n)[1:]))
    else:
        ax.set_yticks([])
        ax.set_xticks([])

def subtract_pups(pup1, pup2):
    """
    Subtract two pups and get the resulting pup. Requires that the pups have identical shapes, resolutions, flanks, etc. If pups contain stripes, these will only be subtracted if stripes have identical coordinates.
    """
    drop_columns = [
        "control_n",
        "control_num",
        "n",
        "num",
        "clr",
        "chroms",
        "minshift",
        "expected_file",
        "maxshift",
        "mindist",
        "maxdist",
        "subset",
        "seed",
        "data",
        "horizontal_stripe",
        "vertical_stripe",
        "cool_path",
        "features",
        "outname",
        "coordinates",
    ]
    pup1 = pup1.reset_index(drop=True)
    pup2 = pup2.reset_index(drop=True)
    drop_columns = list(set(drop_columns) & set(pup1.columns))
    div_pup = pup1.drop(columns=drop_columns)
    for col in div_pup.columns:
        assert np.all(
            np.sort(pup1[col]) == np.sort(pup2[col])
        ), f"Cannot divide these pups, {col} is different between them"
    div_pup["data"] = np.subtract(pup1["data"], pup2["data"])
    # div_pup["data"] = pup1["data"] / pup2["data"]
    div_pup["clrs"] = str(pup1["clr"]) + "-" + str(pup2["clr"])
    div_pup["n"] = pup1["n"] + pup2["n"]
    if set(["vertical_stripe", "horizontal_stripe"]).issubset(pup1.columns):
        if np.all(np.sort(pup1["coordinates"]) == np.sort(pup2["coordinates"])):
            div_pup["coordinates"] = pup1["coordinates"]
            for stripe in ["vertical_stripe", "horizontal_stripe"]:
                div_pup[stripe] = pup1[stripe] / pup2[stripe]
                div_pup[stripe] = div_pup[stripe].apply(
                    lambda x: np.where(np.isin(x, [np.inf, np.nan]), 0, x)
                )
        else:
            logging.info("Stripes cannot be subtracted, coordinates differ between pups")
    return div_pup