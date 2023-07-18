# STEP 1: importing all needed modules, setting up jobs and variables for pathway

# Necessary Modules
import os, glob, re, pickle
from functools import partial
from collections import OrderedDict
import operator as op
from cytoolz import compose
import pandas as pd
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyscenic.export import export2loom, add_scenic_metadata
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.rss import regulon_specificity_scores
from pyscenic.plotting import plot_binarization, plot_rss

# Maximum Number of jobs
sc.settings.njobs = 32

# Directory Pathway Variables
RESOURCES_FOLDERNAME = "/home/linl5/project/Mouse_Tutorial/resources"
RESULTS_FOLDERNAME = "/home/linl5/project/Mouse_Tutorial/results"
FIGURES_FOLDERNAME = "/home/linl5/project/Mouse_Tutorial/figures"
AUXILLIARIES_FOLDERNAME = "/home/linl5/project/Mouse_Tutorial/auxilliaries"

# INPUT: Data Files Pathway Variable 

# Ranking databases. Downloaded from cisTargetDB: https://resources.aertslab.org/cistarget/ (July 5th 2023)
RANKING_DBS_FNAMES = list(map(lambda fn: os.path.join(AUXILLIARIES_FOLDERNAME, fn),
                       ['mm9-500bp-upstream-10species.mc9nr.genes_vs_motifs.rankings.feather',
                       'mm9-tss-centered-10kb-10species.mc9nr.genes_vs_motifs.rankings.feather',
                        'mm9-tss-centered-5kb-10species.mc9nr.genes_vs_motifs.rankings.feather']))

# Motif annotations. Downloaded from cisTargetDB: https://resources.aertslab.org/cistarget/ (July 5th 2023)
MOTIF_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDERNAME, 'motifs-v9-nr.mgi-m0.001-o0.0.tbl')

# Downloaded expression matrix from GEO (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361). (July 5th 2023); organized by cell ID and Gene matrix, values are counts of read in that cell
COUNTS_MTX_FNAME = os.path.join(RESOURCES_FOLDERNAME, 'GSE60361_C1-3005-Expression.txt')

# Downloaded metadata from http://linnarssonlab.org/cortex/ 
METADATA_FNAME = os.path.join(RESOURCES_FOLDERNAME, 'expression_mRNA_17-Aug-2014.txt')

# parsing Motif annotation file for list for unique transcription factors (from step 3)
MM_TFS_FNAME = os.path.join(RESULTS_FOLDERNAME, 'mm_tfs.txt')

# Naming DATASET
DATASET_ID = 'GSE60361'

# OUTPUT: File Pathway Variables
# Output: Expression matrix after preprocessing
COUNTS_QC_MTX_FNAME = os.path.join(RESULTS_FOLDERNAME, 'GSE60361.qc.counts.csv')

# Scoring after cistarget
ADJACENCIES_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.adjacencies.tsv'.format(DATASET_ID))

# Motif List
MOTIFS_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.motifs.csv'.format(DATASET_ID))

REGULONS_DAT_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.regulons.dat'.format(DATASET_ID))
AUCELL_MTX_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.auc.csv'.format(DATASET_ID))
BIN_MTX_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.bin.csv'.format(DATASET_ID))
THR_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.thresholds.csv'.format(DATASET_ID))
ANNDATA_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.h5ad'.format(DATASET_ID))
LOOM_FNAME = os.path.join(RESULTS_FOLDERNAME, '{}.loom'.format(DATASET_ID))


# STEP 2: Pasring Motif Annotation file to construct a TF list

# read motif annotation file from imported pathway
pd_motifs = pd.read_csv(MOTIF_ANNOTATIONS_FNAME, sep='\t')

### A LOT OF WARNINGS ###
print("\n\n\n\n\n")

# parses through pd_motifs to isolate the column with transcription factor name, output number of tfs, expected on tutorial is 1721
mm_tfs = pd_motifs.gene_name.unique()
print("NUMBERS OF TRANSCRIPTION FACTOR: " + str(len(mm_tfs)) + "\n")

# writing out the list of transcription factors to resource file
with open(MM_TFS_FNAME, 'wt') as f:
    f.write('\n'.join(mm_tfs) + '\n')



# STEP 3: Parsing Expression matrix

# read expression matrix from imported pathway
df_counts = pd.read_csv(COUNTS_MTX_FNAME, sep='\t', index_col=0)

# writing out shape of 
print("SHAPE OF EXPRESSION MATRIX: " + str(df_counts.shape) + "\n")



# STEP 4: Prepping Metadata

#reads a CSV file, selects the first 9 rows and transpose
df_metadata = pd.read_csv(METADATA_FNAME, sep='\t', index_col=1, nrows=9).drop(columns=['Unnamed: 0']).T.reset_index() 
df_metadata.columns.name = ''
df_metadata.age = df_metadata.age.astype(int)

#displaying the dataframe object of metadata file
print("INFORMATION CONTRAINED IN METADATA:\n")
print(df_metadata.head())
print("\n")



# STEP 5: Creating Anndata from expression matrix and metadata file

#Initailizes a AnnData object from Scanpy library 
adata = sc.AnnData(X=df_counts.T.sort_index())

#AnnData object storing the observation data (metadata) for each cell of the expression matrix. 
adata.obs = df_metadata.set_index('cell_id').sort_index()

### Warning about variables are not unique, but i did use it? ###
#Using the make_unique method from scanpy to ensure that our gene names (stored in var_name) is unique
adata.var_names_make_unique()

#preprocessing and filters our data such that each cell must have expressed at least 200 genes, and each gene must exist in at least 3 cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

#Storing raw (unprocessed and not normalized) data
adata.raw = adata 

#Converts the filtered and processed anndata with metadata back into a DataFrame called df_counts_qc. Used for SCENIC later.
df_counts_qc = adata.to_df()

#printing expression matrix values before normalization
print("EXPRESSION MATRIX BEFORE NORMALIZATION:\n")
Norm_expresion = adata.X
print(Norm_expresion)
print("\n")

#Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization. Count is set to median of all total cell count
sc.pp.normalize_total(adata)

#Printing expression matrix post normalization
print("EXPRESSION MATRIX AFTER NORMALIZATION TO MEDIAN OF TOTAL GENE COUNT IN CELL:\n")
Norm_expresion = adata.X
print(Norm_expresion)
print("\n")

#applying the natural logarithm to (count value + 1)
sc.pp.log1p(adata)

#Printing expression matrix After log-transformation on normalized data
print("EXPRESSION MATRIX AFTER LOG-TRANSFORMATION OF NORMALIZED DATA:\n")
Norm_expresion = adata.X
print(Norm_expresion)
print("\n")

#Printing variable x observations with metadata
print("INFORMATION ABOUT ANNDATA OBJECT WITH CELL X GENES, METADATA ATTRIBUTES, UNSTRUCTURED DATA (USER DEFINED): \n")
print(adata)
print("\n")



# STEP 6:Applying highly variable genes method and plotting 

# applying hvg method to access if genes are hvg
# default bin size is 20, min dispersion is 0.5
sc.pp.highly_variable_genes(adata)

# plotting HVG plots
sc.pl.highly_variable_genes(adata)

#Filtering out genes that aren't HVG
adata = adata[:, adata.var['highly_variable']]

#Printing anndata after HVG filter
print("INFORMATION ABOUT ANNDATA POST HIGHLY VARIABLE GENE FILTER, MIN_DISPERSION = 0.5, BIN = 20: \n")
print(adata)
print("\n")



# STEP 7: Applying PCA
sc.tl.pca(adata, svd_solver='arpack')

#Priting data inforamtion post-PCA
print("INFORMATION ABOUT ANNDATA POST PCA:\n")
print(adata)
print("\n")



# STEP 8: running T-SNE
sc.tl.tsne(adata)

#plotting T-SNE
sc.set_figure_params(frameon=False, dpi=150, fontsize=8)
sc.pl.tsne(adata, color=['level1class', 'sex', 'age', 'Gad1' ], 
           title=['GSE60361 - Cell types', 'Sex', 'Age', 'Gad1'], ncols=2, use_raw=False)


# STEP PRE-A (SCENIC SUBSAMPLING)

print("\n\n\n\n\n")
print("STARTING SCENIC WORK FLOW\n")
print("TEST: WITH SUBSAMPLING OF EXPRESSION DATA\n")

# Set the seed for reproducibility
seed_state = 1
np.random.seed(seed_state)

# Subsample the cell population of DataFrame with a specified seed for scenic grn boost run
subsampled_cell_df_counts_qc = df_counts_qc.sample(frac=0.01, random_state=seed_state)

# Specify the number of columns (GENES) to subsample
n_columns_subsampled = 100

# Get the total number of columns in the original DataFrame
n_total_columns = subsampled_cell_df_counts_qc.shape[1]

# Generate random genes indices for the subsampled columns
subsampled_column_indices = np.random.choice(range(n_total_columns), size=n_columns_subsampled, replace=False)

# Subset the DataFrame based on the random column indices
subsampled_cell_feature_df_counts_qc = subsampled_cell_df_counts_qc.iloc[:, subsampled_column_indices]

#check printing subsampled_cell_feature_df_counts_qc
print("Subsample Expression matrix:\n")
print(subsampled_cell_feature_df_counts_qc)
print("\n")

#output subsampled expression matrix that will be used to run with part A of scenic
subsampled_cell_feature_df_counts_qc.to_csv(COUNTS_QC_MTX_FNAME)



# STEP A: running SCENIC GRNBoost

# Read the CSV file into a DataFrame
df = pd.read_csv(COUNTS_QC_MTX_FNAME)

# Get the shape of the DataFrame
rows, columns = df.shape

# Print the number of rows and columns
print("Number of rows:", rows, "\n")
print("Number of columns:", columns, "\n")

#input file: in results folder is the dataframe file with raw counts (currently subsamples from step 8) and also mouse TF list
#output: csv file with three columns: the TF, its target, the importance of the TF on that target

import subprocess

#terminal command to run GENIE/GRN 

command = "pyscenic grn {} {} -o {} --num_workers 32".format(
    COUNTS_QC_MTX_FNAME,
    MM_TFS_FNAME,
    ADJACENCIES_FNAME
)

subprocess.run(command, shell=True)

# printing completion of STEP A SCENIC
print("SCENIC: PART A: SUCCESSFUL \n")



#STEP B: Regulon construction with cistarget/iRegulon

#Importing all the ranking files (each 1 gb) to one pathway variable
DBS_PARAM = ' '.join(RANKING_DBS_FNAMES)

# terminal command to run cistargetx
command = [
    'pyscenic',
    'ctx',
    ADJACENCIES_FNAME,
    DBS_PARAM,
    '--annotations_fname',
    MOTIF_ANNOTATIONS_FNAME,
    '--expression_mtx_fname',
    COUNTS_QC_MTX_FNAME,
    '--output',
    MOTIFS_FNAME,
    '--num_workers',
    '26'
]

# Run the command
subprocess.run(command)
