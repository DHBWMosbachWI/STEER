import os
from os.path import join
import pickle as pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import configargparse
import json
from dotenv import load_dotenv

load_dotenv(override=True)

import sys

word_embedding_path = join(os.environ["WORKING_DIR"], "emb_clus",
                           "word_embedding", "out")

labeled_unlabeled_test_split_path = join(os.environ["WORKING_DIR"], "data",
                                         "extract", "out",
                                         "labeled_unlabeled_test_split")

valid_headers_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                          "valid_headers")

numeric_types_public_bi_num = ["X1B",
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

if __name__ == "__main__":

    parser = configargparse.ArgParser()
    parser.add("--word_embeddings",
               type=str,
               default="word_embeddings_all_df.p")
    parser.add("--valid_headers", type=str, default="public_bi_type78.json")
    parser.add("--distance_threshold", type=float, default=0.1)
    parser.add("--labeled_data_size", type=float, default=1.0)
    parser.add("--unlabeled_data_size", type=float, default=79.0)
    parser.add("--test_data_size", type=float, default=20.0)
    parser.add("--corpus", type=str, default="public_bi")
    parser.add("--validation_on", type=str, default="test")
    parser.add("--gen_train_data", type=bool, default=False)
    parser.add("--random_state", type=int, default=2)
    parser.add("--comment", type=str, default="")
    parser.add("--mode", type=str, default=False) # both, strings, numerics

    # for absolut number of labeled train data
    parser.add("--absolute_numbers", type=bool, default=False)

    args = parser.parse_args()
    word_embeddings = args.word_embeddings
    valid_headers_file = args.valid_headers
    distance_threshold = args.distance_threshold
    labeled_data_size = args.labeled_data_size
    unlabeled_data_size = args.unlabeled_data_size
    test_data_size = args.test_data_size
    corpus = args.corpus
    validation_on = args.validation_on
    gen_train_data = args.gen_train_data
    absolute_numbers = args.absolute_numbers
    random_state = args.random_state
    comment = args.comment
    mode = args.mode

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

    # load the word embeddings
    # word_embeddings = pd.read_pickle(join(word_embedding_path,
    #                                       word_embeddings))
    if "csv" in word_embeddings:
        word_embeddings = pd.read_csv(join(word_embedding_path, word_embeddings))
        if corpus == "sportsDB":
            word_embeddings["table"] = word_embeddings["table"].apply(lambda x: x.split(".csv")[0])
            word_embeddings["dataset_id"] = word_embeddings["table"]+"+"+word_embeddings["column"] 
            word_embeddings["word_embedding"] = word_embeddings.apply(lambda test: eval(test["word_embedding"]) if type(test["word_embedding"]) == str else None, axis=1)
    else:
        with open(join(join(word_embedding_path, word_embeddings)), "rb") as f:
            word_embeddings = pickle.load(f)
    #print(word_embeddings.head(3))
    # filter out the labeled and test data ids
    word_embeddings_labeled = word_embeddings.loc[
        word_embeddings["dataset_id"].isin(labeled_data_ids)]
    word_embeddings_labeled["already_labeled"] = True

    if gen_train_data:
        word_embeddings_unlabeled = word_embeddings.loc[
            word_embeddings["dataset_id"].isin(unlabeled_data_ids)]
        word_embeddings_unlabeled["already_labeled"] = False
        print(f"word embeddings unlabeled: {len(word_embeddings_unlabeled)}")

    word_embeddings_test = word_embeddings.loc[
        word_embeddings["dataset_id"].isin(test_data_ids)]
    word_embeddings_test["already_labeled"] = False

    print(f"word embeddings labeled: {len(word_embeddings_labeled)}")
    print(f"word embeddings test: {len(word_embeddings_test)}")
    

    if gen_train_data == False:

        ####################
        ## Validation Part
        ####################

        # combine the two dataframes
        word_embeddings = pd.concat(
            [word_embeddings_labeled, word_embeddings_test], ignore_index=True)
        print(len(word_embeddings))
        #print(word_embeddings)

        # load the valid headers with real sem. types
        valid_headers = join(valid_headers_path, valid_headers_file)
        with open(valid_headers, "r") as file:
            valid_headers = json.load(file)

        # transform valid header into df to make it joinable with clustering results
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

        # join word_embeddings and valid header df on "table"&"column"
        word_embeddings = word_embeddings.join(
            valid_header_df.set_index(["table", "column", "dataset_id"]),
            on=["table", "column", "dataset_id"]).reset_index(drop=True)
        #print(word_embeddings)

        ##############
        ## Clustering
        ##############

        # generate word_embedding vector into one numpy array for clusteirng algo
        X = word_embeddings["word_embedding"].tolist()
        print(f"Columns to cluster: {len(X)}")

        # Clustering
        print(
            f"Do Agglomerative clustering with distance_threshold {distance_threshold}"
        )
        clustering = AgglomerativeClustering(
            linkage="average",
            affinity="cosine",
            distance_threshold=distance_threshold,
            n_clusters=None).fit(X)
        print(f"Labels: {len(clustering.labels_)}")
        print(f"Clusters: {len(np.unique(clustering.labels_))}")

        word_embeddings["cluster_label"] = list(clustering.labels_)

        # drop column word embedding
        word_embeddings = word_embeddings.drop(columns=["word_embedding"])

        #print(word_embeddings)

        ##################
        ## Label clusters
        ##################

        word_embeddings["predicted_type"] = None
        # set predicted_type to the already labeled cols
        word_embeddings.loc[(word_embeddings["already_labeled"] == True),
                            "predicted_type"] = word_embeddings.loc[
                                (word_embeddings["already_labeled"] == True),
                                "semanticType"]
        #print(word_embeddings[word_embeddings["already_labeled"]].to_markdown())

        # iterate over all cluster labels and try to sign a label with the already labeled cols
        for index, cluster in enumerate(
                word_embeddings["cluster_label"].unique()):
            # if index >= 2:
            #     break
            df_cols_in_cluster_labeled = word_embeddings[
                (word_embeddings["cluster_label"] == cluster)
                & (word_embeddings["already_labeled"] == True)]
            #print(df_cols_in_cluster_labeled.to_markdown())
            # calc the semantic type of the cluster by majority vote
            if len(df_cols_in_cluster_labeled) > 0:
                sem_type_of_cluster = df_cols_in_cluster_labeled.groupby(
                    "semanticType").size().reset_index(
                        name="counts").sort_values(
                            by="counts", ascending=False)["semanticType"][0]
                #print(sem_type_of_cluster)
            else:
                sem_type_of_cluster = "None"
            # assign the calced sem. type to all cols in that cluster, which are not already labeled
            word_embeddings.loc[(word_embeddings["cluster_label"] == cluster)
                                &
                                (word_embeddings["already_labeled"] == False),
                                "predicted_type"] = sem_type_of_cluster
            #print(word_embeddings[word_embeddings["cluster_label"] == cluster])
        #print(word_embeddings.to_markdown())

        # store dataframe of the results
        store_result_path = join(os.environ["WORKING_DIR"], "emb_clus",
                                 "without_knn", "out")

        word_embeddings.to_csv(join(
            store_result_path,
            f"{corpus}_clustering_n_classify_results_{validation_on}_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}_{mode}_{comment}.csv"
        ),
                               index=False)

        #########################
        ## classification report
        #########################

        # report only on the test data
        #y_pred = word_embeddings[(word_embeddings["already_labeled"] == False) & (word_embeddings["predicted_type"] != "None")]["semanticType"]
        #y_real = word_embeddings[(word_embeddings["already_labeled"] == False) & (word_embeddings["predicted_type"] != "None")]["predicted_type"]

        y_pred = word_embeddings[(
            word_embeddings["already_labeled"] == False)]["predicted_type"] 
        y_real = word_embeddings[(
            word_embeddings["already_labeled"] == False)]["semanticType"]
        if len(y_pred) > 0:
            class_rep = classification_report(y_real, y_pred, output_dict=True)
            #print(class_rep)

            # store classification report
            store_result_path_validation = join(os.environ["WORKING_DIR"],
                                                "emb_clus", "without_knn",
                                                "out", "validation")

            with open(
                    join(
                        store_result_path_validation,
                        f"{corpus}_classification_report_{validation_on}_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}_{mode}_{comment}.json"
                    ), "w") as file:
                json.dump(class_rep, file)

    ###############################
    ## Generate training data part
    ###############################
    if gen_train_data:
        # combine the two dataframes
        # word_embeddings = pd.concat(
        #     [word_embeddings_labeled, word_embeddings_unlabeled], ignore_index=True)
        # print(len(word_embeddings))
        #print(word_embeddings)

        # load the valid headers with real sem. types
        valid_headers = join(valid_headers_path, valid_headers_file)
        with open(valid_headers, "r") as file:
            valid_headers = json.load(file)

        # transform valid header into df to make it joinable with clustering results
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

        # join word_embeddings and valid header df on "table"&"column"
        # word_embeddings = word_embeddings.join(
        #     valid_header_df.set_index(["table", "column", "dataset_id"]),
        #     on=["table", "column", "dataset_id"]).reset_index(drop=True)
        #print(word_embeddings)

        ##################################################
        ### Before clustering split
        ### the data corpus in to buckets
        ### because of memory issues while cluster
        ### each bucket should consist 50.000 datapoints
        ##################################################
        n_chunk = 50000
        list_word_embeddings_unlabeled = [
            word_embeddings_unlabeled[i:i + n_chunk]
            for i in range(0, len(word_embeddings_unlabeled), n_chunk)
        ]

        header, save_mode = False, "a"
        for index_list_we, word_embeddings_unlabeled in enumerate(
                list_word_embeddings_unlabeled):
            if index_list_we == 0:
                header = True
            else:
                header = False
            # combine the two dataframes
            word_embeddings = pd.concat(
                [word_embeddings_labeled, word_embeddings_unlabeled],
                ignore_index=True)
            print(f"Do Clustering with: {len(word_embeddings)}")
            #print(word_embeddings)

            # join word_embeddings and valid header df on "table"&"column"
            word_embeddings = word_embeddings.join(
                valid_header_df.set_index(["table", "column", "dataset_id"]),
                on=["table", "column", "dataset_id"]).reset_index(drop=True)
            if mode == "numerics":
                if corpus == "public_bi_num":
                    word_embeddings = word_embeddings.loc[word_embeddings["semanticType"].isin(
                        numeric_types_public_bi_num)]
                elif corpus == "sportsDB": 
                    word_embeddings = word_embeddings.loc[word_embeddings["semanticType"].isin(
                        numeric_types_sportsDB)]
                print(f"word embeddings unlabeled (numerics): {len(word_embeddings[word_embeddings['already_labeled'] == False])}")
            elif mode == "strings":
                if corpus == "public_bi_num":
                    word_embeddings = word_embeddings.loc[~word_embeddings["semanticType"].isin(numeric_types_public_bi_num)]
                    print(f"word embeddings unlabeled (numerics): {len(word_embeddings[word_embeddings['already_labeled'] == False])}")
                elif corpus == "sportsDB":
                    word_embeddings = word_embeddings.loc[~word_embeddings["semanticType"].isin(numeric_types_sportsDB)]
                    print(f"word embeddings unlabeled (numerics): {len(word_embeddings[word_embeddings['already_labeled'] == False])}")
            elif mode == "both":
                ## nothing to do here until yet in this mode
                pass
            ## do clustering with only a chunk of the whole data corpus

            ##############
            ## Clustering
            ##############
            #print(word_embeddings.head(5))
            #print(type(word_embeddings.iloc[0]["word_embedding"]))
            #sys.exit()
            # generate word_embedding vector into one numpy array for clusteirng algo
            word_embeddings.dropna(inplace=True)
            
            X = word_embeddings["word_embedding"].tolist()
            #X = word_embeddings["word_embedding"].to_numpy()
            print(f"Columns to cluster: {len(X)}")
            print(f"word embedding vec: {len(X[0])}")
            print(X[:10])
            
            
            # Clustering
            print(
                f"Do Agglomerative clustering with distance_threshold {distance_threshold}"
            )
            
            
            clustering = AgglomerativeClustering(
                linkage="average",
                affinity="cosine",
                distance_threshold=distance_threshold,
                n_clusters=None).fit(X)
            
            print(f"Labels: {len(clustering.labels_)}")
            print(f"Clusters: {len(np.unique(clustering.labels_))}")

            word_embeddings["cluster_label"] = list(clustering.labels_)

            # drop column word embedding
            word_embeddings = word_embeddings.drop(columns=["word_embedding"])

            #print(word_embeddings)

            ##################
            ## Label clusters
            ##################

            word_embeddings["predicted_type"] = None
            # set predicted_type to the already labeled cols
            word_embeddings.loc[(word_embeddings["already_labeled"] == True),
                                "predicted_type"] = word_embeddings.loc[(
                                    word_embeddings["already_labeled"] == True
                                ), "semanticType"]
            #print(word_embeddings[word_embeddings["already_labeled"]].to_markdown())

            # iterate over all cluster labels and try to sign a label with the already labeled cols
            for index, cluster in enumerate(
                    word_embeddings["cluster_label"].unique()):
                # if index >= 2:
                #     break
                df_cols_in_cluster_labeled = word_embeddings[
                    (word_embeddings["cluster_label"] == cluster)
                    & (word_embeddings["already_labeled"] == True)]
                #print(df_cols_in_cluster_labeled.to_markdown())
                # calc the semantic type of the cluster by majority vote
                if len(df_cols_in_cluster_labeled) > 0:
                    sem_type_of_cluster = df_cols_in_cluster_labeled.groupby(
                        "semanticType").size().reset_index(
                            name="counts").sort_values(
                                by="counts",
                                ascending=False)["semanticType"][0]
                    #print(sem_type_of_cluster)
                else:
                    sem_type_of_cluster = "None"
                # assign the calced sem. type to all cols in that cluster, which are not already labeled
                word_embeddings.loc[
                    (word_embeddings["cluster_label"] == cluster)
                    & (word_embeddings["already_labeled"] == False),
                    "predicted_type"] = sem_type_of_cluster
                #print(word_embeddings[word_embeddings["cluster_label"] == cluster])
            #print(word_embeddings.to_markdown())

            # store dataframe of the results
            # check if there is a existing file

            store_result_path = join(os.environ["WORKING_DIR"], "emb_clus",
                                     "without_knn", "out")

            # if file already exists => raise an error
            # if os.path.isfile(
            #         join(
            #             store_result_path,
            #             f"{corpus}_clustering_n_classify_results_gen_train_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
            #         )):
            #     print("There exists already a output file!!")
            #     raise RuntimeError

            word_embeddings.to_csv(join(
                store_result_path,
                f"{corpus}_clustering_n_classify_results_gen_train_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}_{mode}_{comment}.csv"
            ),
                                   header=header,
                                   mode=save_mode,
                                   index=False)

            ## store generated training data
            ## to do
            store_result_path = join(os.environ["WORKING_DIR"], "emb_clus",
                                     "without_knn", "out", "gen_training_data")

            # if file already exists => raise an error
            # if os.path.isfile(
            #         join(
            #             store_result_path,
            #             f"{corpus}_gen_training_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
            #         )):
            #     print("There exists already a output file!!")
            #     raise RuntimeError

            word_embeddings = word_embeddings[
                (word_embeddings["already_labeled"] == False)
                & (word_embeddings["predicted_type"] != "None")][[
                    "table", "column", "dataset_id", "predicted_type"
                ]].rename(
                    columns={"predicted_type": "predicted_semantic_type"})
            word_embeddings.to_csv(join(
                store_result_path,
                f"{corpus}_gen_training_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}_{mode}_{comment}.csv"
            ),
                                   header=header,
                                   mode=save_mode,
                                   index=False)
