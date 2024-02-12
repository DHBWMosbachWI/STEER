import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(os.environ["WORKING_DIR"])

from scipy.stats import wasserstein_distance, kruskal
from data_loader.utils import load_public_bi_table_by_cols, load_sportsDB_soccer_table
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json
import copy
from os.path import join
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
import configargparse
from snorkel.labeling import PandasLFApplier
import pyspark.pandas as ps
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, pandas_udf, PandasUDFType, collect_list, count, avg, lit, mean, stddev, monotonically_increasing_id, row_number
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, StructType, StructField


# create and register UDF-Function to calc EMD-Distance


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

#spark = SparkSession.builder.config(conf=conf).getOrCreate()

#spark.udf.register("emd_UDF", emd_UDF)

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

numeric_types_sportsDB = [
    "age",
    "assists",
    "gamesPlayed",
    "goals",
    "goalsPlusAssistsPer90Min",
    "minutesPlayed",
    "nonPenaltyXGoalsPer90Min",
    "nonPenaltyXGoalsPlusAssists",
    "penaltiesAttempted",
    "penaltiesScored",
    "redCards",
    "xAssistsPer90Min",
    "xGoalsPer90Min",
    "xGoalsPlusAssistsPer90Min",
    "yellowCards"
  ]

# LabelEncoder
with open(
    join(os.environ["WORKING_DIR"], "data", "extract", "out",
         "valid_types", "types.json")) as f:
    valid_types = json.load(f)["type_sportsDB"]

label_enc = LabelEncoder()
label_enc.fit(valid_types)

PERCENTAGE_OF_OVERLAP_FOR_JOIN_CANDIDATES = 0.75
P_VALUE_CORRSTEERTION_ANALYZATION = 0.05
PERCENTAGE_THRESHOLD_UNIQUE_VALUES = 0.1

labeled_data_size = 4
unlabeled_data_size = "absolute"
test_data_size = 20.1
validation_on = "test"
gen_train_data = True
corpus = "sportsDB"
absolute_numbers = True
n_worker = 4
threshold_EMD_factor = 0.1
max_group_size = 4
random_state = 2
table_frac = None
approach = 1  # because this script is just for normal EMD


for labeled_data_size in [1,2,3,4,5]:
    for random_state in [1,2,3,4,5]:

        if absolute_numbers:
            unlabeled_data_size = "absolute"
            labeled_data_size = int(labeled_data_size)

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
        valid_header_file = f"{corpus}_type_sportsDB.json"
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

            df_table_with_n_col_to_label = load_sportsDB_soccer_table(
                numeric_column_to_label["table"], usecols=df_cols_to_load["col_num"].values, col_headers=df_cols_to_load["col_header"].values)

            df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]] = pd.to_numeric(
                df_table_with_n_col_to_label[numeric_column_to_label["dataset_id"]], errors="coerce")
            df_table_with_n_col_to_label.dropna(inplace=True)
            if len(df_table_with_n_col_to_label) == 0:
                return -1

            # search all already labeled numeric cols in the corpus
            already_labeled_numeric_cols = total_labeled_data_df.loc[total_labeled_data_df["semanticType"].isin(
                numeric_types_sportsDB)].drop_duplicates()

            # iterrate over all alread labeled numeric col and do the EMD measure
            results = []
            for index, row in already_labeled_numeric_cols.iterrows():

                df_table_with_labeled_numeric = load_sportsDB_soccer_table(row["table"], usecols=[int(
                    row["column"].split("_")[1])], col_headers=[row["dataset_id"]], frac=table_frac)

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
                df_results.to_csv(join(os.environ["WORKING_DIR"], "labeling_functions", "numerics", "normal_EMD", "results", corpus,
                                    f"{numeric_column_to_label['dataset_id']}_appr{approach}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), index=False)
            elif gen_train_data == False:
                df_results.to_csv(join(os.environ["WORKING_DIR"], "labeling_functions", "numerics", "normal_EMD", "results_test_data", corpus,
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
            total_labeled_data_df = labeled_data_df.drop_duplicates(
                subset=["table", "column", "dataset_id"])

            # only unlabaled columns of tyoe numeric
            numeric_unlabeled_data_df = unlabeled_data_df.loc[unlabeled_data_df["semanticType"].isin(
                numeric_types_sportsDB)]
            #numeric_unlabeled_data_df = numeric_unlabeled_data_df[678:]

            # define LF to apply
            lfs = [normal_EMD]

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
                f"{corpus}_normal_EMD_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
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
                f"{corpus}_gen_training_data_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
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
                        f"{corpus}_classification_report_unlabeled_{threshold_EMD_factor}_{table_frac}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
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
            gen_labeled_data_df = pd.read_csv(join(gen_train_data_path, f"public_bi_gen_training_data_all_combined_maj_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"), names=[
                "table", "column", "dataset_id", "semanticType"])

            ### drop duplicate only on cols "table", "column", "dataset_id"!!! This must be fixed. Actually it cant happen that there are two duplicte sets with different semantic types!
            total_labeled_data_df = pd.concat(
                [labeled_data_df, gen_labeled_data_df]).drop_duplicates(subset=["table", "column", "dataset_id"])

            # only unlabaled columns of tyoe numeric
            numeric_test_data_df = test_data_df.loc[test_data_df["semanticType"].isin(
                numeric_types_sportsDB)]
            #numeric_unlabeled_data_df = numeric_unlabeled_data_df[678:]

            # define LF to apply
            lfs = [normal_EMD]

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
