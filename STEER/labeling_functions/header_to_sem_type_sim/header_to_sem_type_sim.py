import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.append(os.environ["WORKING_DIR"])
from os.path import join
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
import copy
import json
import configargparse
from sklearn.metrics.pairwise import cosine_similarity
from snorkel.labeling import PandasLFApplier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report

COSINE_SIM_THRESHOLD = 0.9

labeled_unlabeled_test_split_path = join(os.environ["WORKING_DIR"], "data",
                                         "extract", "out",
                                         "labeled_unlabeled_test_split")

valid_headers_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                          "valid_headers")

# LabelEncoder
with open(
        join(os.environ["WORKING_DIR"], "data", "extract", "out",
             "valid_types", "types.json")) as f:
    valid_types = json.load(f)[os.environ["TYPENAME"]]

label_enc = LabelEncoder()
label_enc.fit(valid_types)

### load google use v5
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# embed = hub.load(
#     "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")


### table colum loader of raw data
def load_tablecolumn_header_turl(dataset_id: str):
    table_id = dataset_id.split("+")[0]
    column_id = dataset_id.split("+")[1].split("_")[1]
    header = pd.read_csv(join(os.environ["TURL"], "tables_with_headers",
                              table_id),
                         usecols=[int(column_id)]).columns[0]
    return header


from sql_metadata import Parser

def load_tablecolum_header_public_bi(dataset_id: str):
    table_id = dataset_id.split("+")[0]
    folder_id = table_id.split("_")[0]
    column_id = dataset_id.split("+")[1].split("_")[1]
    sql = open(join(os.environ["PUBLIC_BI_BENCHMARK"], folder_id, "tables", f"{table_id}.table.sql"), "r")
    header = Parser(sql.read()).columns[int(column_id)]
    return header 


df_valid_type_WE = pd.DataFrame({"valid_types": valid_types})
#df_valid_type_WE["valid_types_WE"] = df_valid_type_WE.apply(lambda row: embed([" ".join(row["valid_types"].split("."))]), axis=1)
if os.environ["TYPENAME"] == "type78":
    df_valid_type_WE["valid_types_WE"] = df_valid_type_WE.apply(
        lambda row: embed([row["valid_types"]]), axis=1)
elif os.environ["TYPENAME"] == "type_public_bi":
    df_valid_type_WE["valid_types_WE"] = df_valid_type_WE.apply(
        lambda row: embed([row["valid_types"]]), axis=1)
else: # split the semantic types that look like: american_football.football_team => football_team
    df_valid_type_WE["valid_types_WE"] = df_valid_type_WE.apply(
        lambda row: embed([row["valid_types"].split(".")[1]]), axis=1)


@labeling_function()
def header_emebedding_similarity(x):
    if os.environ["CORPUS"] == "turl":
        header = load_tablecolumn_header_turl(x["dataset_id"])
    elif os.environ["CORPUS"] == "public_bi":
        header = load_tablecolum_header_public_bi(x["dataset_id"])
    df_compare = copy.copy(df_valid_type_WE)
    header_embedded = embed([header])
    df_compare["cosine_sim"] = df_compare.apply(lambda row: cosine_similarity(
        row["valid_types_WE"], header_embedded)[0][0],
                                                axis=1)
    df_compare = df_compare.sort_values(by="cosine_sim", ascending=False)
    if df_compare.iloc[0]["cosine_sim"] >= COSINE_SIM_THRESHOLD:
        LABEL = label_enc.transform([df_compare.iloc[0]["valid_types"]])[0]
        return LABEL
    else:
        return -1


if __name__ == "__main__":

    parser = configargparse.ArgParser()

    parser.add("--labeled_data_size", type=float, default=1.0)
    parser.add("--unlabeled_data_size", type=float, default=79.0)
    parser.add("--test_data_size", type=float, default=20.0)
    parser.add("--corpus", type=str, default="public_bi")
    parser.add("--validation_on", type=str, default="test")
    parser.add("--n_worker", type=int, default=4)
    parser.add("--gen_train_data", type=bool, default=False)
    parser.add("--random_state", type=int, default=2)

    # for absolut number of labeled train data
    parser.add("--absolute_numbers", type=bool, default=False)

    args = parser.parse_args()
    labeled_data_size = args.labeled_data_size
    unlabeled_data_size = args.unlabeled_data_size
    test_data_size = args.test_data_size
    validation_on = args.validation_on
    n_worker = args.n_worker
    gen_train_data = args.gen_train_data
    corpus = args.corpus
    absolute_numbers = args.absolute_numbers
    random_state = args.random_state

    if absolute_numbers:
        unlabeled_data_size = "absolute"
        labeled_data_size = int(labeled_data_size)

    #############
    ## Load data
    #############

    # load labeled data from labeled, unlabeled, test split file and use labeled and test data for clustering
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
    valid_header_file = f"{corpus}_{os.environ['TYPENAME']}_valid.json"
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

    ####################################
    ## Validation Part on test data only
    ####################################
    # validation part is only test data because in generatin training data, validation is also done
    if gen_train_data == False:

        # filter out test data from valid headers
        test_data_df = valid_header_df.loc[valid_header_df["dataset_id"].isin(
            test_data_ids)]

        # define labeling functions to apply
        lfs = [header_emebedding_similarity]

        # snorkel pandas applier for apply lfs to the data
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=test_data_df)

        print(f"Length of labeled data: {len(L_train)}")

        test_data_df["predicted_semantic_type"] = [
            label_enc.inverse_transform([x])[0] if x != -1 else "None"
            for x in L_train
        ]

        # save lf results
        test_data_df.to_csv(join(
            os.environ["WORKING_DIR"], "labeling_functions",
            "header_to_sem_type_sim", "out", "results",
            f"{corpus}_header_to_sem_type_results_test_{COSINE_SIM_THRESHOLD}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
                            index=False)

        # do classification report

        cls_report = classification_report(
            test_data_df["semanticType"],
            test_data_df["predicted_semantic_type"],
            output_dict=True)

        # save classification_report
        with open(
                join(
                    os.environ["WORKING_DIR"], "labeling_functions",
                    "header_to_sem_type_sim", "out", "validation",
                    f"{corpus}_classification_report_test_{COSINE_SIM_THRESHOLD}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
                ), "w") as f:
            json.dump(cls_report, f)

    ###########################
    ## Generating training data
    ###########################

    if gen_train_data:
        # filter out unlabeled data from valid_headers
        unlabeled_data_df = valid_header_df.loc[
            valid_header_df["dataset_id"].isin(unlabeled_data_ids)]

        # define labeling functions to apply
        lfs = [header_emebedding_similarity]

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
        
        #L_train = applier.apply(df=unlabeled_data_df)
        L_train = parallelize(unlabeled_data_df, applier.apply, n_worker)

        print(
            f"Length of labeled data: {len([x for x in L_train if x != -1])}")

        unlabeled_data_df["predicted_semantic_type"] = [
            label_enc.inverse_transform([x])[0] if x != -1 else "None"
            for x in L_train
        ]

        # save lf results
        unlabeled_data_df.to_csv(join(
            os.environ["WORKING_DIR"], "labeling_functions",
            "header_to_sem_type_sim", "out", "results",
            f"{corpus}_header_to_sem_type_results_{COSINE_SIM_THRESHOLD}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
                                 index=False)

        # save gen train data
        class_reportable_data = unlabeled_data_df.drop(unlabeled_data_df[
            unlabeled_data_df["predicted_semantic_type"] == "None"].index)

        class_reportable_data[[
            "table", "column", "dataset_id", "predicted_semantic_type"
        ]].to_csv(join(
            os.environ["WORKING_DIR"], "labeling_functions",
            "header_to_sem_type_sim", "out", "gen_training_data",
            f"{corpus}_gen_training_data_{COSINE_SIM_THRESHOLD}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
                  index=False)

        cls_report = classification_report(
            class_reportable_data["semanticType"],
            class_reportable_data["predicted_semantic_type"],
            output_dict=True)

        # save classification_report
        with open(
                join(
                    os.environ["WORKING_DIR"], "labeling_functions",
                    "header_to_sem_type_sim", "out", "validation",
                    f"{corpus}_classification_report_unlabeled_{COSINE_SIM_THRESHOLD}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
                ), "w") as f:
            json.dump(cls_report, f)