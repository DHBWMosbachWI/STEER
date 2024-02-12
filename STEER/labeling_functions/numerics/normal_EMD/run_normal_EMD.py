import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(os.environ["WORKING_DIR"])
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, StructType, StructField
from pyspark.sql.functions import udf, col, pandas_udf, PandasUDFType, collect_list, count, avg, lit, mean, stddev, monotonically_increasing_id, row_number
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.pandas as ps
from snorkel.labeling import PandasLFApplier
import configargparse
from snorkel.preprocess import preprocessor
from snorkel.labeling import labeling_function
from os.path import join
import copy
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from data_loader.utils import load_public_bi_table_by_cols
from scipy.stats import wasserstein_distance, kruskal

#import logging
import time
from datetime import datetime

# # create and register UDF-Function to calc EMD-Distance
# @udf(returnType=FloatType())
# def emd_UDF(col1, col2) -> FloatType:
#     return float(wasserstein_distance(col1, col2))


# conf = SparkConf()
# # conf.set("spark.executor.instances","2")
# # conf.set("spark.executor.cores","2")
# conf.set("spark.executor.memory", "150g")
# conf.set("spark.driver.memory", "150g")
# conf.set("spark.memory.offHeap.enabled", "true")
# conf.set("spark.memory.offHeap.size", "50g")
# #conf.set("spark.sql.execution.arrow.enabled", "true")
# conf.setMaster("local[*]")
# conf.setAppName("STEER")

# spark = SparkSession.builder.config(conf=conf).getOrCreate()

# spark.udf.register("emd_UDF", emd_UDF)

labeled_unlabeled_test_split_path = join(os.environ["WORKING_DIR"], "data",
                                         "extract", "out",
                                         "labeled_unlabeled_test_split")

valid_headers_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                          "valid_headers")

gen_train_data_path = join(os.environ["WORKING_DIR"], "labeling_functions", "combined_LFs",
                           "gen_training_data")

numeric_types = ["X1B",
                 "X2B",
                 "X3B",
                 "TB",
                 "HR",
                 "R",
                 "BB",
                 "AB",
                 "GIDP",
                 "HBP",
                 "H",
                 "SF",
                 "SH",
                 "SO",
                 "iBB",
                 "CS",
                 "SB",
                 "latitude",
                 "longitude",
                 "year"]

# LabelEncoder
with open(
        join(os.environ["WORKING_DIR"], "data", "extract", "out",
             "valid_types", "types.json")) as f:
    valid_types = json.load(f)[os.environ["TYPENAME"]]

label_enc = LabelEncoder()
label_enc.fit(valid_types)

PERCENTAGE_OF_OVERLAP_FOR_JOIN_CANDIDATES = 0.75
P_VALUE_CORRSTEERTION_ANALYZATION = 0.05
PERCENTAGE_THRESHOLD_UNIQUE_VALUES = 0.1

if __name__ == "__main__":

    parser = configargparse.ArgParser()

    parser.add("--labeled_data_size", type=float, default=1.0)
    parser.add("--unlabeled_data_size", type=float, default=79.0)
    parser.add("--test_data_size", type=float, default=20.0)
    parser.add("--corpus", type=str, default="public_bi")
    parser.add("--validation_on", type=str, default="test")
    parser.add("--gen_train_data", type=bool, default=False)
    parser.add("--n_worker", type=int, default=4)
    parser.add("--threshold_EMD_factor", type=float, default=0.01)
    parser.add("--max_group_size", type=int, default=3)
    parser.add("--random_state", type=int, default=2)
    parser.add("--table_frac", type=float, default=None)
    parser.add("--pruning_mode", type=int, default=None)
    # pruning-mode:
    #       None:   no pruning
    #       0:      pruning. Use only tables with at least one same labeled str col.
    #       1:      pruning. Use same tables as in Mode 0, but also with tables where no str cols are labeled
    #parser.add("--approach", type=int, default=4)

    # for absolut number of labeled train data
    parser.add("--absolute_numbers", type=bool, default=False)

    args = parser.parse_args()
    labeled_data_size = args.labeled_data_size
    unlabeled_data_size = args.unlabeled_data_size
    test_data_size = args.test_data_size
    validation_on = args.validation_on
    gen_train_data = args.gen_train_data
    corpus = args.corpus
    absolute_numbers = args.absolute_numbers
    n_worker = args.n_worker
    threshold_EMD_factor = args.threshold_EMD_factor
    max_group_size = args.max_group_size
    random_state = args.random_state
    table_frac = args.table_frac
    pruning_mode = args.pruning_mode
    approach = 1 # because this script is just for normal EMD 

    if absolute_numbers:
        unlabeled_data_size = "absolute"
        labeled_data_size = int(labeled_data_size)


    ### logging
    import logging
    logging.basicConfig(filename=join("logs", f"{corpus}_normal_EMD_pruning{pruning_mode}_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.log"), encoding="utf-8", level=logging.DEBUG, force=True)
    #logging.basicConfig(filename="test.log", encoding="utf-8", level=logging.DEBUG, force=True)
    logging.info(f"programm sarted: {datetime.now()}")
    start_time = time.time()

    #############
    # Load data
    #############

    # load labeled data from labeled, unlabeled, test split file
    with open(
            join(
                labeled_unlabeled_test_split_path,
                f"{corpus}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
            )) as f:
        labeled_unlabeled_test_split_file = json.load(f)
        labeled_data_ids = labeled_unlabeled_test_split_file[
            f"labeled{labeled_data_size}"]
        if gen_train_data:
            if absolute_numbers:
                unlabeled_data_ids = labeled_unlabeled_test_split_file[
                    f"unlabeled"]
            else:
                unlabeled_data_ids = labeled_unlabeled_test_split_file[
                    f"unlabeled{unlabeled_data_size}"]
            print(f"Unlabeled Data: {len(unlabeled_data_ids)}")
        if validation_on == "unlabeled":
            test_data_ids = labeled_unlabeled_test_split_file[
                f"{validation_on}"]
        else:
            test_data_ids = labeled_unlabeled_test_split_file[
                f"{validation_on}{test_data_size}"]

    print(f"Labeled Data: {len(labeled_data_ids)}")
    print(f"Test Data: {len(test_data_ids)}")

    # load the valid headers with real sem. types
    valid_header_file = f"{corpus}_{os.environ['TYPENAME']}.json"
    valid_headers = join(valid_headers_path, valid_header_file)
    with open(valid_headers, "r") as file:
        valid_headers = json.load(file)
    # transform valid header into df to make it joinable with word embeddings
    valid_header_df_data = []
    for table in valid_headers.keys():
        for column in valid_headers[table].keys():
            valid_header_df_data.append([
                table, column, table + "+" + column,
                valid_headers[table][column]["semanticType"]
            ])
    valid_header_df = pd.DataFrame(
        valid_header_df_data,
        columns=["table", "column", "dataset_id", "semanticType"])

    #############
    # Build LF
    #############
    @labeling_function()
    def normal_EMD(numeric_column_to_label):
        print("Numeric Column to label: " +
              numeric_column_to_label["dataset_id"])
        # load the table with the numeric column to label
        cols_to_load = [numeric_column_to_label["dataset_id"]]
        df_cols_to_load = pd.DataFrame({"col_num": [int(col.split(
            "+")[1].split("_")[1]) for col in cols_to_load], "col_header": cols_to_load}).sort_values(by="col_num")

        df_table_with_n_col_to_label = load_public_bi_table_by_cols(numeric_column_to_label["table"].split(
            "_")[0], numeric_column_to_label["table"], usecols=df_cols_to_load["col_num"].values, col_headers=df_cols_to_load["col_header"].values)

        df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]] = pd.to_numeric(
            df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]], errors="coerce")
        df_table_with_n_col_to_label.dropna(inplace=True)
        if len(df_table_with_n_col_to_label) == 0:
            return -1

        # search all already labeled numeric cols in the corpus
        already_labeled_numeric_cols = total_labeled_data_df.loc[total_labeled_data_df["semanticType"].isin(
            numeric_types)].drop_duplicates()

        # iterrate over all alread labeled numeric col and do the EMD measure
        results = []
        for index, row in already_labeled_numeric_cols.iterrows():

            df_table_with_labeled_numeric = load_public_bi_table_by_cols(row["table"].split(
                "_")[0], row["table"], usecols=[int(row["column"].split("_")[1])], col_headers=[row["dataset_id"]], frac=table_frac)

            df_table_with_labeled_numeric[row["dataset_id"]] = pd.to_numeric(
                df_table_with_labeled_numeric[row["dataset_id"]], errors="coerce")
            df_table_with_labeled_numeric.dropna(inplace=True)
            if len(df_table_with_labeled_numeric[row["dataset_id"]].to_list()) == 0:
                continue

            # EMD calc
            emd = wasserstein_distance(df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].to_list(
            ), df_table_with_labeled_numeric[row["dataset_id"]].to_list())
            print(f"EMD: {emd}")
            results.append([numeric_column_to_label["dataset_id"], numeric_column_to_label["semanticType"],
                            row["dataset_id"], row["semanticType"], emd, df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].mean(), df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].std()])

        df_results = pd.DataFrame(results, columns=[
                                  "unlabeled_col", "real_semantic_type", "labeled_col", "semantic_type", "EMD", "mean", "std"])  # .sort_values(by="EMD")
        df_results = df_results[pd.to_numeric(
            df_results['EMD'], errors='coerce').notnull()]
        df_results = df_results.sort_values(by="EMD")
        if gen_train_data:
            df_results.to_csv(join(os.environ["WORKING_DIR"], "labeling_functions", "numerics", "normal_EMD", "results",
                                f"{numeric_column_to_label['dataset_id']}_appr{approach}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), index=False)
        elif gen_train_data == False:
            df_results.to_csv(join(os.environ["WORKING_DIR"], "labeling_functions", "numerics", "normal_EMD", "results_test_data",
                                f"{numeric_column_to_label['dataset_id']}_appr{approach}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), index=False)
        predicted_semantic_type = -1
        if len(df_results) > 0:
            if df_results.iloc[0]["EMD"] >= (df_results.iloc[0]["std"] * threshold_EMD_factor):
                return -1
            if len(df_results) < 2:
                predicted_semantic_type = df_results.iloc[0]["semantic_type"]
                predicted_semantic_type = label_enc.transform(
                    [predicted_semantic_type])[0]
            if len(df_results) > 2:
                if ((df_results.iloc[0]["EMD"] == df_results.iloc[1]["EMD"]) & (df_results.iloc[0]["semantic_type"] != df_results.iloc[1]["semantic_type"])):
                    return -1
                predicted_semantic_type = df_results.iloc[0]["semantic_type"]
                predicted_semantic_type = label_enc.transform(
                    [predicted_semantic_type])[0]
        return predicted_semantic_type

    #############
    # Build LF
    #############
    @labeling_function()
    def normal_EMD_pruning(numeric_column_to_label):
        print("Numeric Column to label: " +
              numeric_column_to_label["dataset_id"])
        
        # 1. search for already labeled string types in the same table of the unlabeled numeric column that we want to label next
        string_already_labeled_cols_in_table = total_labeled_data_df[
            total_labeled_data_df["table"] == numeric_column_to_label["table"]]
        string_already_labeled_cols_in_table = string_already_labeled_cols_in_table.loc[
            ~string_already_labeled_cols_in_table["semanticType"].isin(numeric_types)]      

        # load the table with the numeric column to label
        cols_to_load = [numeric_column_to_label["dataset_id"]]
        string_cols_to_load = list(
            string_already_labeled_cols_in_table["dataset_id"].values)
        cols_to_load = cols_to_load + string_cols_to_load
        df_cols_to_load = pd.DataFrame({"col_num": [int(col.split(
            "+")[1].split("_")[1]) for col in cols_to_load], "col_header": cols_to_load}).sort_values(by="col_num")

        df_table_with_n_col_to_label = load_public_bi_table_by_cols(numeric_column_to_label["table"].split(
            "_")[0], numeric_column_to_label["table"], usecols=df_cols_to_load["col_num"].values, col_headers=df_cols_to_load["col_header"].values)

        df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]] = pd.to_numeric(
            df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]], errors="coerce")
        # df_table_with_n_col_to_label.dropna(inplace=True)
        # if len(df_table_with_n_col_to_label) == 0:
        #     return -1


        # 2. search tables with labeled numeric columns from the same domain
        # detect the same domain by comparing already labeled string columns
        # at least one same already labeled string column must be in the other table with the labales numerical column
        joinable_tables = total_labeled_data_df.loc[total_labeled_data_df["semanticType"].isin(
            string_already_labeled_cols_in_table["semanticType"])].drop_duplicates()
        joinable_tables_sorted = joinable_tables[joinable_tables["table"] != numeric_column_to_label["table"]].groupby(
            "table").count().sort_values(by=["semanticType"], ascending=False)


        # 3. Iterate over all founded tables + the tables with no labels inside.
        results = []
        for joinable_table, row in joinable_tables_sorted.iterrows():
            if joinable_table != "MLB_4":
                break
            # first check if there is a numeric column in the table which is already labeled with an numeric type
            numerics_in_joinable_table = total_labeled_data_df[
                total_labeled_data_df["table"] == joinable_table]
            numerics_in_joinable_table = numerics_in_joinable_table.loc[numerics_in_joinable_table["semanticType"].isin(
                numeric_types)]
            if len(numerics_in_joinable_table) == 0:
                continue
            #print(numerics_in_joinable_table)

            if len(numerics_in_joinable_table) > 0:
                # strings_in_joinable_table = joinable_tables[joinable_tables["table"] == joinable_table]

                # load the table, only with the numeric cols to measure emd against
                cols_to_load_for_joinable_table = list(
                    numerics_in_joinable_table["dataset_id"].values)

                df_cols_to_load_for_joinable_table = pd.DataFrame({"col_num": [int(col.split(
                    "+")[1].split("_")[1]) for col in cols_to_load_for_joinable_table], "col_header": cols_to_load_for_joinable_table}).sort_values(by="col_num")
                df_joinable_table = load_public_bi_table_by_cols(joinable_table.split(
                    "_")[0], joinable_table, usecols=df_cols_to_load_for_joinable_table["col_num"].values, col_headers=df_cols_to_load_for_joinable_table["col_header"].values, frac=table_frac)
                        
                #print(df_joinable_table.head(3))

                for i, numeric_col_in_joinable_table in numerics_in_joinable_table.iterrows():
                    print("EMD Calc:")
                    #print(numeric_col_in_joinable_table)
                    unlabeled_col = df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].dropna().to_list()
                    if len(unlabeled_col) == 0:
                        return -1
                    labeled_col = df_joinable_table[numeric_col_in_joinable_table["dataset_id"]].dropna().to_list()
                    print(len(labeled_col))
                    if len(labeled_col) == 0:
                        continue
                    
                    # EMD calc
                    emd = wasserstein_distance(unlabeled_col, labeled_col)
                    print(f"EMD: {emd}")

                    results.append([numeric_column_to_label["dataset_id"], numeric_column_to_label["semanticType"],numeric_col_in_joinable_table["dataset_id"], numeric_col_in_joinable_table["semanticType"], emd, df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].mean(), df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].std()])

        if pruning_mode == 1:
            for joinable_table in tables_with_no_strCol_labels:
                # first check if there is a numeric column in the table which is already labeled with an numeric type
                numerics_in_joinable_table = total_labeled_data_df[
                    total_labeled_data_df["table"] == joinable_table]
                numerics_in_joinable_table = numerics_in_joinable_table.loc[numerics_in_joinable_table["semanticType"].isin(
                    numeric_types)]
                if len(numerics_in_joinable_table) == 0:
                    continue
                #print(numerics_in_joinable_table)

                if len(numerics_in_joinable_table) > 0:
                    # strings_in_joinable_table = joinable_tables[joinable_tables["table"] == joinable_table]

                    # load the table, only with the numeric cols to measure emd against
                    cols_to_load_for_joinable_table = list(
                        numerics_in_joinable_table["dataset_id"].values)

                    df_cols_to_load_for_joinable_table = pd.DataFrame({"col_num": [int(col.split(
                        "+")[1].split("_")[1]) for col in cols_to_load_for_joinable_table], "col_header": cols_to_load_for_joinable_table}).sort_values(by="col_num")
                    df_joinable_table = load_public_bi_table_by_cols(joinable_table.split(
                        "_")[0], joinable_table, usecols=df_cols_to_load_for_joinable_table["col_num"].values, col_headers=df_cols_to_load_for_joinable_table["col_header"].values, frac=table_frac)
                            
                    #print(df_joinable_table.head(3))

                    for i, numeric_col_in_joinable_table in numerics_in_joinable_table.iterrows():
                        print("EMD Calc:")
                        #print(numeric_col_in_joinable_table)
                        unlabeled_col = df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].dropna().to_list()
                        if len(unlabeled_col) == 0:
                            return -1
                        labeled_col = df_joinable_table[numeric_col_in_joinable_table["dataset_id"]].dropna().to_list()
                        print(len(labeled_col))
                        if len(labeled_col) == 0:
                            continue
                        
                        # EMD calc
                        emd = wasserstein_distance(unlabeled_col, labeled_col)
                        print(f"EMD: {emd}")

                        results.append([numeric_column_to_label["dataset_id"], numeric_column_to_label["semanticType"],numeric_col_in_joinable_table["dataset_id"], numeric_col_in_joinable_table["semanticType"], emd, df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].mean(), df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]].std()])

        df_results = pd.DataFrame(results, columns=[
                                    "unlabeled_col", "real_semantic_type", "labeled_col", "semantic_type", "EMD", "mean", "std"])  # .sort_values(by="EMD")
        df_results = df_results[pd.to_numeric(
            df_results['EMD'], errors='coerce').notnull()]
        df_results = df_results.sort_values(by="EMD")

        if gen_train_data:
            df_results.to_csv(join(os.environ["WORKING_DIR"], "labeling_functions", "numerics", "normal_EMD", "results",
                                f"{numeric_column_to_label['dataset_id']}_appr{approach}_pruning{pruning_mode}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), index=False)
        elif gen_train_data == False:
            df_results.to_csv(join(os.environ["WORKING_DIR"], "labeling_functions", "numerics", "normal_EMD", "results_test_data",
                                f"{numeric_column_to_label['dataset_id']}_appr{approach}_pruning{pruning_mode}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), index=False)
        predicted_semantic_type = -1
        if len(df_results) > 0:
            if df_results.iloc[0]["EMD"] >= (df_results.iloc[0]["std"] * threshold_EMD_factor):
                return -1
            if len(df_results) < 2:
                predicted_semantic_type = df_results.iloc[0]["semantic_type"]
                predicted_semantic_type = label_enc.transform(
                    [predicted_semantic_type])[0]
            if len(df_results) > 2:
                if ((df_results.iloc[0]["EMD"] == df_results.iloc[1]["EMD"]) & (df_results.iloc[0]["semantic_type"] != df_results.iloc[1]["semantic_type"])):
                    return -1
                predicted_semantic_type = df_results.iloc[0]["semantic_type"]
                predicted_semantic_type = label_enc.transform(
                    [predicted_semantic_type])[0]

        return predicted_semantic_type

    if gen_train_data:
        # filter out unlabeled data from valid_headers
        unlabeled_data_df = valid_header_df.loc[
            valid_header_df["dataset_id"].isin(unlabeled_data_ids)]

        # load already labeled data
        labeled_data_df = valid_header_df.loc[valid_header_df["dataset_id"].isin(
            labeled_data_ids)]

        # load already generated labeled train data
        # gen_labeled_data_df = pd.read_csv(join(gen_train_data_path, f"public_bi_gen_training_data_all_combined_maj_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), names=[
        #                                   "table", "column", "dataset_id", "semanticType"])

        ### drop duplicate only on cols "table", "column", "dataset_id"!!! This must be fixed. Actually it cant happen that there are two duplicte sets with different semantic types!
        # total_labeled_data_df = pd.concat(
        #     [labeled_data_df, gen_labeled_data_df]).drop_duplicates(subset=["table", "column", "dataset_id"])
        total_labeled_data_df = labeled_data_df.drop_duplicates(subset=["table", "column", "dataset_id"])

        # only unlabaled columns of tyoe numeric
        numeric_unlabeled_data_df = unlabeled_data_df.loc[unlabeled_data_df["semanticType"].isin(
            numeric_types)]
        #numeric_unlabeled_data_df = numeric_unlabeled_data_df[678:]

        if pruning_mode == 1:
            # search tables with labeled numerical column but with no labeled string columns
            total_number_of_lab_num_cols_in_total_unlabeled_tables = 0
            tables_with_no_strCol_labels = []
            for idx, groups in enumerate(total_labeled_data_df.groupby("table")):
                # if idx > 2:
                #     break
                #print(groups[0])
                df = groups[1]
                #print(df)
                # search in table for a labeled num col
                df_lab_num_col = df[df["semanticType"].isin(numeric_types)]
                #print(df_lab_num_col)
                if len(df_lab_num_col) == 0:
                    #print("No labeled num col")
                    continue
                # search in the table for labeled string cols
                df_lab_str_col = df[~df["semanticType"].isin(numeric_types)]
                if len(df_lab_str_col) == 0:
                    total_number_of_lab_num_cols_in_total_unlabeled_tables += len(df_lab_num_col)
                    tables_with_no_strCol_labels.append(groups[0])
                    #print(df)

        # define LF to apply
        if pruning_mode == None:
            lfs = [normal_EMD]
        elif (pruning_mode == 0) or (pruning_mode == 1):
            lfs = [normal_EMD_pruning]


        # snorkel pandas applier for apply lfs to the data
        applier = PandasLFApplier(lfs=lfs)

        from multiprocessing import Pool
        from multiprocessing.pool import ThreadPool as Pool
        from functools import partial
        import numpy as np
        from tqdm.auto import tqdm

        def parallelize(data, func, num_of_processes=8):
            data_split = np.array_split(data, num_of_processes)
            pool = Pool(num_of_processes)
            #data = pd.concat(pool.map(func, data_split))
            data = np.concatenate(pool.map(func, data_split), axis=0)
            pool.close()
            pool.join()
            return data

        L_train = applier.apply(df=numeric_unlabeled_data_df)
        #L_train = parallelize(numeric_unlabeled_data_df,
        #                      applier.apply, n_worker)

        print(
            f"Length of labeled data: {len([x for x in L_train if x != -1])}")

        numeric_unlabeled_data_df["predicted_semantic_type"] = [
            label_enc.inverse_transform([x])[0] if x != -1 else "None"
            for x in L_train
        ]
        numeric_unlabeled_data_df.to_csv(join(
            os.environ["WORKING_DIR"], "labeling_functions",
            "numerics", "normal_EMD", "out", "results",
            f"{corpus}_normal_EMD_pruning{pruning_mode}_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
            index=False)

        # save gen train data
        class_reportable_data = numeric_unlabeled_data_df.drop(numeric_unlabeled_data_df[
            numeric_unlabeled_data_df["predicted_semantic_type"] == "None"].index)

        class_reportable_data[[
            "table", "column", "dataset_id", "predicted_semantic_type"
        ]].to_csv(join(
            os.environ["WORKING_DIR"], "labeling_functions", "numerics",
            "normal_EMD", "out", "gen_train_data",
            f"{corpus}_gen_training_data_pruning{pruning_mode}_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
                  index=False)

        cls_report = classification_report(
            class_reportable_data["semanticType"],
            class_reportable_data["predicted_semantic_type"],
            output_dict=True)

        # save classification_report
        with open(
                join(
                    os.environ["WORKING_DIR"], "labeling_functions", "numerics",
                    "normal_EMD", "out", "validation",
                    f"{corpus}_classification_report_unlabeled_pruning{pruning_mode}_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
                ), "w") as f:
            json.dump(cls_report, f)

    ########################################
    ## Validation only / no gen training data
    ########################################
    else:
        # filter out unlabeled data from valid_headers
        test_data_df = valid_header_df.loc[
            valid_header_df["dataset_id"].isin(test_data_ids)]

        # load already labeled data
        labeled_data_df = valid_header_df.loc[valid_header_df["dataset_id"].isin(
            labeled_data_ids)]

        # load already generated labeled train data
        # gen_labeled_data_df = pd.read_csv(join(gen_train_data_path, f"public_bi_gen_training_data_all_combined_maj_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), names=[
        #                                   "table", "column", "dataset_id", "semanticType"])

        ### drop duplicate only on cols "table", "column", "dataset_id"!!! This must be fixed. Actually it cant happen that there are two duplicte sets with different semantic types!
        # total_labeled_data_df = pd.concat(
        #     [labeled_data_df, gen_labeled_data_df]).drop_duplicates(subset=["table", "column", "dataset_id"])
        total_labeled_data_df = labeled_data_df.drop_duplicates(subset=["table", "column", "dataset_id"])

        # only unlabaled columns of tyoe numeric
        numeric_test_data_df = test_data_df.loc[test_data_df["semanticType"].isin(
            numeric_types)]
        #numeric_unlabeled_data_df = numeric_unlabeled_data_df[678:]

        if pruning_mode == 1:
            # search tables with labeled numerical column but with no labeled string columns
            total_number_of_lab_num_cols_in_total_unlabeled_tables = 0
            tables_with_no_strCol_labels = []
            for idx, groups in enumerate(total_labeled_data_df.groupby("table")):
                # if idx > 2:
                #     break
                #print(groups[0])
                df = groups[1]
                #print(df)
                # search in table for a labeled num col
                df_lab_num_col = df[df["semanticType"].isin(numeric_types)]
                #print(df_lab_num_col)
                if len(df_lab_num_col) == 0:
                    #print("No labeled num col")
                    continue
                # search in the table for labeled string cols
                df_lab_str_col = df[~df["semanticType"].isin(numeric_types)]
                if len(df_lab_str_col) == 0:
                    total_number_of_lab_num_cols_in_total_unlabeled_tables += len(df_lab_num_col)
                    tables_with_no_strCol_labels.append(groups[0])
                    #print(df)

        # define LF to apply
        if pruning_mode == None:
            lfs = [normal_EMD]
        elif (pruning_mode == 0) or (pruning_mode == 1):
            lfs = [normal_EMD_pruning]

        # snorkel pandas applier for apply lfs to the data
        applier = PandasLFApplier(lfs=lfs)

        from multiprocessing import Pool
        from multiprocessing.pool import ThreadPool as Pool
        from functools import partial
        import numpy as np
        from tqdm.auto import tqdm

        def parallelize(data, func, num_of_processes=8):
            data_split = np.array_split(data, num_of_processes)
            pool = Pool(num_of_processes)
            #data = pd.concat(pool.map(func, data_split))
            data = np.concatenate(pool.map(func, data_split), axis=0)
            pool.close()
            pool.join()
            return data

        L_train = applier.apply(df=numeric_test_data_df)
        #L_train = parallelize(numeric_unlabeled_data_df,
        #                      applier.apply, n_worker)

        print(
            f"Length of labeled data: {len([x for x in L_train if x != -1])}")
    

    ### logging
    logging.info(f"programm end: {datetime.now()}")
    logging.info(f"elapsed time (sec.): {time.time() - start_time}")