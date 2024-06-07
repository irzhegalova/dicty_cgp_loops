# %%
from sklearn.cluster import KMeans
import warnings
import pandas as pd
import numpy as np
# correct class for using uncorrected values
from goatools.godag_plot import plot_gos, plot_results, plot_goid2goobj, GODagSmallPlot, OboToGoDagSmall
from goatools.semantic import semantic_similarity, resnik_sim,  lin_sim, TermCounts, get_info_content

from pybedtools import BedTool
# Import the OBO parser from GOATools
from goatools import obo_parser
from itertools import combinations, product
import wget
import os
import sys

sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from custom_functions import create_rnaseq_df

def compute_bma(termcounts, go, gene1='DDB_G0283757', gene2='DDB_G0292268', threshold=None):
    """
    Compute the Best Match Average (BMA) score between two genes based on their Gene Ontology (GO) term associations.
    
    Args:
        termcounts (dict): A dictionary containing the counts of each GO term in the dataset.
        go (dict): A dictionary representing the Gene Ontology (GO) hierarchy.
        gene1 (str, optional): The ID of the first gene. Defaults to 'DDB_G0283757'.
        gene2 (str, optional): The ID of the second gene. Defaults to 'DDB_G0292268'.
        threshold (float, optional): A threshold value used to filter out GO terms with low information content. Defaults to None.
    
    Returns:
        float: The Best Match Average (BMA) score between the two genes.
    """
    # Get the list of GO terms associated with gene1 and gene2
    a = list(assoc_dicty_dic[gene1])
    b = list(assoc_dicty_dic[gene2])
    
    # Create dictionaries to store the similarity scores for each GO term
    a_dic = {}
    b_dic = {}
    
    # Get the cartesian product of the GO terms from gene1 and gene2
    c = list(product(a, b))
    
    # Calculate the similarity score for each pair of GO terms
    for go1, go2 in c:
        # Check if the similarity score should be calculated based on a threshold
        if threshold is not None:
            IC1 = get_info_content(go1, termcounts)
            IC2 = get_info_content(go2, termcounts)
            if (IC1 < threshold) or (IC2 < threshold):
                continue
        
        # Calculate the Lin similarity score for the pair of GO terms
        sim_l = lin_sim(go1, go2, go, termcounts)
        
        # Prevent nans by replacing None with np.nan
        if sim_l is None:
            sim_l = np.nan
        
        # Store the similarity score in the dictionaries
        if go1 not in a_dic:
            a_dic[go1] = [sim_l]
        else:
            a_dic[go1].append(sim_l)
        
        if go2 not in b_dic:
            b_dic[go2] = [sim_l]
        else:
            b_dic[go2].append(sim_l)
    
    # Calculate the maximum similarity score for each GO term in a_dic and b_dic
    a_sum = []
    b_sum = []
    for a_i in a_dic.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a_sum.append(np.nanmax(a_dic[a_i]))
    
    for b_i in b_dic.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            b_sum.append(np.nanmax(b_dic[b_i]))
    
    # Calculate the Best Match Average (BMA) score
    bma = (np.nansum(a_sum) + np.nansum(b_sum)) / (len(a_dic) + len(b_dic))
    
    return bma

class GODagSmallPlot_uncorrected(GODagSmallPlot):
    @staticmethod
    def _init_pval_name(**kws):
        """Initialize pvalue attribute name."""
        if 'pval_name' in kws:
            return kws['pval_name']
        if 'goea_results' in kws:
            goea = kws['goea_results']
            if goea:
                return "p_uncorrected"#.format(M=goea[0].method_flds[0].fieldname)


def plot_gos_uncorrected(output_path, go_ids, obo_dag, *args, **kwargs):
    """Plot paths from GO IDs.

    Args:
        output_path (str): The path to the output PNG file.
        go_ids (list): A list of GO IDs.
        obo_dag (OboDag): The OboDag object.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Raises:
        KeyError: If the 'engine' keyword argument is missing.

    Returns:
        None
    """
    # Check if 'engine' is provided, otherwise use 'pydot' as default
    engine = kwargs.get('engine', 'pydot')

    # Create a small GO DAG from the given GO IDs and OboDag
    go_dag_small = OboToGoDagSmall(go_ids=go_ids, obodag=obo_dag).godag

    # Create a plotter for the small GO DAG
    go_dag_plot = GODagSmallPlotUncorrected(go_dag_small, *args, **kwargs)

    # Plot the GO DAG to the output PNG file
    go_dag_plot.plot(output_path, engine)


loops_path = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bed"
# loops_path = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.filtered.bed"
loops_bed = BedTool(loops_path)
loops_df = pd.read_table(loops_path, header=None)
loops_df.columns = ["chrom", "start", "end"]
for group in ['1', '2', '3', '4', 'Unicellular', 'Multicellular']:
    feature = BedTool("data/genome/pseudogenes_group" + group + ".bed")
    covByPseudogenes = loops_bed.coverage(feature).to_dataframe()
    loops_df["covByGroup"+str(group)]  = covByPseudogenes.loc[:, 'thickStart'].tolist()
loops_df['loop_id'] = loops_df.index
loops_dropped = loops_df[['covByGroup1', 'covByGroup2', 'covByGroup3',
       'covByGroup4']]


kmeans = KMeans(n_clusters=6, random_state=0).fit(loops_dropped)
kmeans.labels_
loops_df['kmeans_cl'] = kmeans.labels_

# GO terms for clustered loops
go_dicty_dic = {}
assoc_dicty_dic = {}
go_dicty_df = pd.read_table("data/genome/gene_assoc.Dec5_2022.gaf",#"data/gene_association.dictyBase",
                            header=None,
                            comment="!", on_bad_lines='skip')
go_dicty_df.columns = ['DB', 'Gene', 'Gene_symbol', 'Qualifier',
                       'GO_id', 'DB:Reference',
                       'Evidence Code', 'With_from', 'Aspect',
                       'Gene_name', 'Gene_synonym',
                       'DBobject_type', 'taxon', 'Date',
                       'Assigned_by', 'Annp_extension',
                       'Gene_Product_Form_ID']
for i in range(go_dicty_df.shape[0]):
    gene_name = go_dicty_df.loc[i,'Gene']
    # if tpm_df.query('name in @gene_name').shape[0] > 0:
    # append term dict
    if go_dicty_df.GO_id[i] not in go_dicty_dic.keys():
        go_dicty_dic[go_dicty_df.GO_id[i]] = set()
    go_dicty_dic[go_dicty_df.GO_id[i]].add(go_dicty_df.Gene[i])
    # append gene dict
    if gene_name not in assoc_dicty_dic.keys():
        assoc_dicty_dic[gene_name] = set()
    assoc_dicty_dic[gene_name].add(str(go_dicty_df.loc[i, 'GO_id']))

go_terms_list = list(go_dicty_dic.keys())
go_dicty_dic_filtered = {}
for key in go_terms_list:
    if (len(go_dicty_dic[key]) > 20) & (len(go_dicty_dic[key]) < 200):
        go_dicty_dic_filtered[key] = go_dicty_dic[key]

# APPEND GENE DICT
assoc_dicty_dic_filt = {}
for i in range(go_dicty_df.shape[0]):
    gene_name = go_dicty_df.loc[i, 'Gene']
    if go_dicty_df.loc[i, 'GO_id'] in go_dicty_dic_filtered.keys():
        if gene_name not in assoc_dicty_dic_filt.keys():
            assoc_dicty_dic_filt[gene_name] = set()
        assoc_dicty_dic_filt[gene_name].add(str(go_dicty_df.loc[i, 'GO_id']))

go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
data_folder = os.getcwd() + '/data'

# Check if we have the ./data directory already
if (not os.path.isfile(data_folder)):
    # Emulate mkdir -p (no error if folder exists)
    try:
        os.mkdir(data_folder)
    except OSError as e:
        if (e.errno != 17):
            raise e
else:
    raise Exception('Data path (' + data_folder + ') exists as a file. '
                                                  'Please rename, remove or change the desired location of the data path.')

# Check if the file exists already
if (not os.path.isfile(data_folder + '/go-basic.obo')):
    go_obo = wget.download(go_obo_url, data_folder + '/go-basic.obo')
else:
    go_obo = data_folder + '/go-basic.obo'
go = obo_parser.GODag(go_obo)
go_id = 'GO:0008270'
go_term = go[go_id]

# create accos_dic with namespaces
assoc_dicty_namespace_dic = {}
short_names = {'molecular_function': 'MF',
               'cellular_component': 'CC',
               'biological_process': 'BP'}
for sh in short_names.values():
    assoc_dicty_namespace_dic[sh] = {}
for i in range(go_dicty_df.shape[0]):
    gene_name = go_dicty_df.loc[i,'Gene']
    # if tpm_df.query('name in @gene_name').shape[0] > 0:
    # append gene dict
    go_id = go_dicty_df.loc[i, 'GO_id']
    short_go0 = short_names[go[go_id].namespace]
    if go_dicty_df.loc[i, 'GO_id'] in go_dicty_dic_filtered.keys():
        if gene_name not in assoc_dicty_namespace_dic[short_go0].keys():
            assoc_dicty_namespace_dic[short_go0][gene_name] = set()
        assoc_dicty_namespace_dic[short_go0][gene_name].add(str(go_dicty_df.loc[i, 'GO_id']))

from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
tpm_df = create_rnaseq_df(need_coord=True)
pop = tpm_df.name.tolist()
goeaobj = GOEnrichmentStudyNS(pop, assoc_dicty_namespace_dic, go,
                              propagate_counts=True,
                              alpha=0.05,
                              methods=['fdr_bh'])
GO_items = []

temp = goeaobj.ns2objgoea['BP'].assoc
for item in temp:
    GO_items += temp[item]

temp = goeaobj.ns2objgoea['CC'].assoc
for item in temp:
    GO_items += temp[item]

temp = goeaobj.ns2objgoea['MF'].assoc
for item in temp:
    GO_items += temp[item]

def go_it(goeaobj, test_genes, GO_items):
    print(f'input genes: {len(test_genes)}')

    goea_results_all = goeaobj.run_study(test_genes)
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
    GO = pd.DataFrame(list(map(lambda x: [x.GO, x.goterm.name, x.goterm.namespace, x.goterm.depth, '%.2e' % x.p_uncorrected,
                                          #x.p_uncorrected,
                                          '%.2e' % x.p_fdr_bh, \
                                          x.ratio_in_study[0], x.ratio_in_study[1], GO_items.count(x.GO),
                                          list(x.study_items), \
                                          ], goea_results_all)),
                      columns=['GO', 'term', 'class', 'depth', 'p', 'p_corr', 'n_genes', \
                               'n_study', 'n_go', 'study_genes'])

    GO = GO.query('n_genes > 1 and depth > 2 and n_go < 1000')
    return GO

# %%
