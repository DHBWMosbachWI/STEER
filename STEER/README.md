# STEER: Steered Training Data Generation for Learned Semantic Type Detection

This Repo contains the code and data for our data programming framework STEER.

## Data Preparation
### Public BI Bechnmark
Download the Public BI benchmark data set here:
https://github.com/cwida/public_bi_benchmark/tree/dev/master

### Public BI
Uses the data from Publlic BI Benchmark. As described in the paper. Publi BI only uses the tables and their columns which fits into the Sato's 78 semantic types. Which tables and columns are present in this corpus are specified in the file `STEER/STEER/data/extract/out/valid_headers/public_bi_type78.json`

### Public BI Num
Uses the data from Publlic BI Benchmark. As described in the paper. Publi BI Num extends Public BI by various numeric based semantic types and their table-colums. Which tables and columns are present in this corpus are specified in the file `STEER/STEER/data/extract/out/valid_headers/public_bi_num_public_bi.json`

### TURL-Copus
Download the used Turl data set here:
https://osf.io/bna2k/?view_only=a5e4ee1065704a64a54da605ce697374

These contains the original turl tables separately stored in CSV-Files. There are two version. One version (`tables.zip`) containing all tables with the semantic types as column header. Another version (`tables_with_headers.zip`) containing all tables with the original column headers.

### SportsDB
Download the used SportsDB data set here:
https://osf.io/bna2k/?view_only=a5e4ee1065704a64a54da605ce697374

## Set-up Environment Variables
Most python scripts use environment variables, which must be defined in the `.env` File. The following variables must be set:
```
WORKING_DIT => The path to the directory of STEER/STEER
SATO_DIR => The path to the directory of STEER/Sato
TYPENAME => "type78" for Public BI / "type_turl" for Turl
CORPUS => "public_bi" / "turl"
PUBLIC_BI_BENCHMARK => path to the raw data of the Public BI Benchmark. 
TURL => path to the raw data of the Turl data corpus provided by the link above.
TURL_DIR => unfortunately this is a redundancy to upper variables and must be eliminated in the future. So you have to define the same path as on "TURL" environment variable
PYTHON => your python commmand to run a script (e.g. python, python3, py)
```

## Valid Semantic Types
All used semantic types are defined in `data/extract/out/valid_types/types.json`. Here the list `type78` defines all types used for the Public BI benchmark and the list `type_turl` contains all 105 types used for the turl data corpus.
- `type78` => All semantic types used together with the corpus Public BI
- `type_turl` => All semantic types used together with the corpus TURL-Corpus
- `type_public_bi` => All semantic types used together with the corpus Public BI Num
- `type_sportDB` => All semantic types used together with the corpus SportsDB 

## Valid Headers
The assignment of semantic types to the columns for each data corpus can be found in `data/extract/out/valid_headers`. We have one CSV-File and one JSON-File for each data corpus, but both contain the same information of the assignment. The reason for this is that our system STEER works mostly with the JSON-File and the existing neuronal network Sato, which we used as prediction model currently, with the CSV-File.

## Labeled/Unlabeled/Test Split
To run the labeled/unlabeled/test split for the Public BI or for the Turl data corpus as explained in our paper, you have to run: 
`data/extract/data_split_labeled_unlabeled_test_absolut.py`

When executing the .py script, you must set corresponding required arguments:
```
python data_split_labeled_unlabeled_test_absolut.py --labeled_size [l_size] --test_size [t_size] --valid_headers [v_headers] --corpus [corp] --random_state [r_state]

# l_size: desired number of columns per semantic type in the labeled split (1-5 used in our paper)
# t_size: percentage size of the test split (default 0.2 (20%))
# v_headers: valid header CSV-File in data/extract/out/valid_headers/
# corp: name of the corpus "public_bi" for Public BI "turl" for Turl corpus
# random_state: the random state (seed) value. In the paper we uese 1-5    
```
The scripts stored `.json`-Files in the directory: `STEER/STEER/data/extract/out/labeled_unlabeled_test_split`
The file-names are encoded with the folloing syntax:
`<corpus>_<labeled_data_size>_<unlabeled_data_size>_<test_data_size>_<random_state>.json`

They split files, which specifies wich table-column is in which split, we used in our experiments should be already stored in the directory. So you dont have to execute the script to reproduce our experiment results. 

## Generate Additional Training Data with STEER
### Run all non-numeric LFs
To run all LF you have to do the follwing two steps. For the LF "Embedding Clustering" we provided a seperate .py script.  For the other LFs we have made one .py script which executes one LF after the other automatically.
- LF: Embedding Clustering

First, you must execute the script to calculate the embedded vector represantation for each column in `emb_clus/word_embedding/`:
```
python word_embedding.py --corpus [corp] --fraction [frac] --num_processes [n]

# corp: name of the corpus "public_bi" for Public BI "turl" for Turl corpus
# frac: fraction of the column-values to build the vector representation for one column.
# n: number of parallel processes
```

Second, run the following .py script in `emb_clus/without_knn/`. 
```
python run_cluster_n_classify.py
```

- LFs: [Column Headers, Value-Overlap, Value-Pattern]

Run the following .py script in `labeling_functions/`
```
python run_all_LFs.py
```

### Run numeric LF
The numeric LF is stored in `STEER/STEER/labeling_functions/numerics/normal_EMD`.
To execute the LF for all set-ups (all defined labeled data sizes and all diff. random states) you can execute the script `run_all_set_up.py`. Thereby the script `run_normal_EMD.py` is exectued with the different set-ups and label the numeric cols with semantic types. You can also run the script yourself and define your own input parameters.
```
python run_normal_EMD.py --labeled_data_size [l_d_s] --corpus [corpus] --validation_on [val_on] --gen_train_data [gen_train] --threshold_EMD_factor [thresh_emd] --random_state [ran_st] --pruning_mode [prun_mod]

# l_d_s: labeled data size as described in the paper
# corpus: name of the corpus for which do you want to labeled the unlabeled numeric cols 
# val_on: only for non generating training data mode (gen_train=False). Specifies on which split (unlabeled/test) you want to execute validation to measure the performance of the LF
# gen_train: Specifies if you want to generate training data by labeling the unlabeled columns. If not, the script runs just in validation mode (specifie val_on!)
# treshold_EMD_factor: Provice the threshold EMD as described in the paper. Research of the automatic determining EMD threshold is in `analyse_results.ipynb`
# ran_st: random state you want to choose for the execution
# prun_mod: with pruning, you can choose if the numeric LF uses the textual based semantic types of the same table as additional information during the labeling process (as described in the paper Algorithm 1) or not. Therby:
None: not using textual based semantic types
0: use textual based semantic types during the labeling. Only consider labeled numeric columns for the EMD comparison where in the coresponding table is at least one same labeled textual col like in the table of the unlabeled numeric column.
1: Same as in mode 0, but also compare to labeled numeric columns where the coresponding table has no textual columns with labeled textual semantic types.
```


### Combine the outputs of the LFs
To combine the outputs of the five different LFs as described in our paper, run the following .py script in `labeling_functions/`
```
python run_combine_LFs_labels.py
```
This script will output all generated training data which we can now use to train or re-train an learned model in `labeling_functions/combined_LFs/gen_training_data`

## Train/Retrain existing learned model
### Sato
#### Sato retrained
We re-traind the existing Sato model with the training data defined by different labeled data size (small training data set). To do this you have to run the following script in STEER/Sato/steer_integration/:
```
python run_exp_2.py
```
#### STEER on Sato
We re-trained the existing Sato model with the additional generated training data.
To do this you have to run the following script in STEER/Sato/steer_integration/:
```
python run_exp_4_without_knn_combinedLFs.py

#### Note
## Please set-up first the variables from codeline 5-24 before runining the script!
# BASEPATH: Path to the Sato dir
# TYPENAME: "type78" for Public BI / "type_turl" for TURL
# corpus: "public_bi" / "turl"

## You also have to set-up the variable in codeline 33 & 35 for specifying your filepaths
```
### TURL
#### TURL retrain
We finetuned the existing pre-trained TURL model with the training data defined by th different labeled data sizes (small training data set). To do this you have to run the following script in STEER/Turl/:
On TURL-Corpus:
```
fine_tune_CT_STEER.sh

# Note:
# You have to set the argument add_STEER_train_data to False or better set a comment # in front of the argument
```
On Public BI:
```
fine_tune_CT_STEER_PublicBI.sh

# Note:
# You have to set the argument add_STEER_train_data to False or better set a comment # in front of the argument
```
#### STEER on Turl
We finetuned the existing pre-trained TURL model with the additional generated training data by our STEER labeling framework. To do this you have to run the following script in STEER/Turl/:
On TURL-Corpus:
```
fine_tune_CT_STEER.sh

# Note:
# You have to set the argument add_STEER_train_data to True.
```
On Public BI:
```
fine_tune_CT_STEER_PublicBI.sh

# Note:
# You have to set the argument add_STEER_train_data to True.
```
#### Validate the models
We validate the different TURL models in the Jupyter-Notebook: `evaluate_tasks.ipynb`
There are two cells with the header:
```
##################################
## Evaluation TURL on TURL_Corpus
##################################

###############################
## Evaluation TURL on Public BI
###############################
```
Before you run the cells, you have to set the values of the given variables in the cell to specifie the set-up (e.g. labeled data size...)
