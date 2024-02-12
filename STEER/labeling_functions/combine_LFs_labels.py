import os
import sys
from os.path import join
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np
import configargparse
from dotenv import load_dotenv

load_dotenv(override=True)

labeled_unlabeled_test_split_path = join(os.environ["WORKING_DIR"], "data",
                                         "extract", "out",
                                         "labeled_unlabeled_test_split")

valid_headers_path = join(os.environ["WORKING_DIR"], "data", "extract", "out", "valid_headers")

# LabelEncoder
with open(
        join(os.environ["WORKING_DIR"], "data", "extract", "out",
             "valid_types", "types.json")) as f:
    valid_types = json.load(f)[os.environ["TYPENAME"]]

label_enc = LabelEncoder()
label_enc.fit(valid_types)

distance_threshold = 1e-2
cosine_sim_threshold = 0.9

if __name__ == "__main__":

    parser = configargparse.ArgParser()

    parser.add("--labeled_data_size", type=float, default=1.0)
    parser.add("--unlabeled_data_size", type=float, default=79.0)
    parser.add("--test_data_size", type=float, default=20.0)
    parser.add("--corpus", type=str, default="public_bi")
    parser.add(
        "--gen_train_data",
        type=bool,
        default=False,
        help="False is for generating labels for the test data (for validation only); True for generate Labels for unlabeled data"
    )
    parser.add("--snorkel_label_model",
               type=str,
               default="maj",
               choices=['maj', 'lm'])
    parser.add("--random_state", type=int, default=2)

    # for absolut number of labeled train data
    parser.add("--absolute_numbers", type=bool, default=False)

    args = parser.parse_args()
    labeled_data_size = args.labeled_data_size
    unlabeled_data_size = args.unlabeled_data_size
    test_data_size = args.test_data_size
    corpus = args.corpus
    gen_train_data = args.gen_train_data
    absolute_numbers = args.absolute_numbers
    snorkel_label_model = args.snorkel_label_model
    random_state = args.random_state

    if absolute_numbers:
        unlabeled_data_size = "absolute"
        labeled_data_size = int(labeled_data_size)

    # basic paths of the output of the diff LFs
    source_1 = join(os.environ["WORKING_DIR"], "emb_clus")
    source_2 = join(os.environ["WORKING_DIR"], "labeling_functions")

    # load all types for the differnet LFs
    if corpus == "turl":
        check_elements_types = [
            "american_football.football_team",
            "automotive.model",
            "baseball.baseball_team",
            "film.film_genre",
            "ice_hockey.hockey_team",
            "location.us_county",
            "location.us_state",
            "music.genre",
            "soccer.football_team",
            "soccer.football_player",
            "sports.sports_league",
        ]
        regex_elements_in_col = [
            "aviation.aircraft_model", "internet.website",
            "award.award_category", "film.director",
            "american_football.football_player", "boats.ship_class",
            "cricket.cricket_player", "military.military_unit"
        ]
    elif corpus == "public_bi":
        #check_elements_types = ["gender", "language"]
        check_elements_types = []
        #regex_elements_in_col = ["description", "name"]
        regex_elements_in_col = []
    elif corpus == "public_bi_num":
        check_elements_types = []
        regex_elements_in_col = []
    elif corpus == "sportsDB":
        check_elements_types = []
        regex_elements_in_col = []

    ####################################
    # get results of the different LFs
    # from unlabeled data (generate new
    # training data)
    ####################################
    if gen_train_data:
        # all LFs stored in a dictionary with name as key and path to the labels as value
        lfs = {
            "emb_clus":
            join(
                source_1, "without_knn", "out",
                f"{corpus}_clustering_n_classify_results_gen_train_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            ),
            "header_to_sem_type_sim":
            join(
                source_2, "header_to_sem_type_sim", "out", "results",
                f"{corpus}_header_to_sem_type_results_{cosine_sim_threshold}_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            ),
        }

        # all sem. types labeled with LF: Check elements in cols
        for element in check_elements_types:
            lfs[f"check_elements_in_col_{element}"] = join(
                source_2, "check_elements_in_col", "out", "results",
                f"{corpus}_check_elements_in_col_results_{element}_0.2_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            )

        # all sem. types labeled with LF: Check regex in cols
        for element in regex_elements_in_col:
            lfs[f"regex_elements_in_col_{element}"] = join(
                source_2, "regex_elements_in_col", "out", "results",
                f"{corpus}_regex_elements_in_col_results_{element}_0.2_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            )

        # generated training data with LF for numericy.
        # until yet, there are only numeric LF used for the public bi num
        # For Public BI Num, only use emb_clus with the numeric LF. Header to sem type is not used because the identifiers of the numeric types are derived from the headers anyway
        if corpus == "public_bi_num":
            lfs = {
                "emb_clus": join(source_1, "without_knn", "out", f"{corpus}_clustering_n_classify_results_gen_train_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}_strings_GoogleUSEv3.csv")
            }
            
            lfs[f"emd"] = join(os.environ["WORKING_DIR"], "labeling_functions",
                               "numerics", "normal_EMD", "out", "gen_train_data", f"{corpus}_gen_training_data_pruningNone_0.01_None_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv")
        elif corpus == "sportsDB":
            lfs = {
                "emb_clus": join(source_1, "without_knn", "out", f"{corpus}_clustering_n_classify_results_gen_train_data_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}_strings_GoogleUSEv3.csv")
            }
            
            lfs["emd"] = join(os.environ["WORKING_DIR"], "labeling_functions",
                               "numerics", "normal_EMD", "out", "gen_train_data", f"{corpus}_gen_training_data_0.1_None_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv")
    ####################################
    # get results of the different LFs
    # from test data (validation only)
    ####################################
    else:
        # all LFs stored in a dictionary with name as key and path to the labels as value
        lfs = {
            "emb_clus":
            join(
                source_1, "without_knn", "out",
                f"{corpus}_clustering_n_classify_results_test_{distance_threshold}_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            ),
            "header_to_sem_type_sim":
            join(
                source_2, "header_to_sem_type_sim", "out", "results",
                f"{corpus}_header_to_sem_type_results_test_{cosine_sim_threshold}_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            ),
        }

        for element in regex_elements_in_col:
            lfs[f"regex_elements_in_col_{element}"] = join(
                source_2, "regex_elements_in_col", "out", "results",
                f"{corpus}_regex_elements_in_col_results_test_{element}_0.2_{labeled_data_size}_{unlabeled_data_size}_20.0_{random_state}.csv"
            )

    # load the given labels of each LF
    # load a table from header to sem type to bring emb_cus results in the right sort
    #df_CH = pd.read_csv(lfs["header_to_sem_type_sim"])
    L_train = []

    ## first generate new index structur with dataset_id from each generated train data
    new_index_order = []
    for lf in lfs.keys():
        if lf == "emb_clus":
            if corpus == "public_bi":
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            elif corpus == "public_bi_num":
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            elif corpus == "sportsDB":
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "used_col_values","already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            else:
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "batch", "already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            predictions = predictions[predictions["already_labeled"] == False]
            new_index_order.extend(predictions["dataset_id"].tolist())
        else:
            predictions = pd.read_csv(
                lfs[lf])
            new_index_order.extend(predictions["dataset_id"].tolist())
    
    new_index_order = list(set(new_index_order))    

    for lf in lfs.keys():
        # special for emb_clus
        if lf == "emb_clus":
            if corpus == "public_bi":
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            elif corpus == "public_bi_num":
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            elif corpus == "sportsDB":
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "used_col_values","already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            else:
                predictions = pd.read_csv(lfs[lf],
                                          header=0,
                                          names=[
                                              "table", "column", "dataset_id",
                                              "batch", "already_labeled",
                                              "semanticType", "cluster_label",
                                              "predicted_semantic_type"
                ])
            predictions = predictions[predictions["already_labeled"] == False]
            # for ensuring that each label
            
            predictions = predictions.set_index("dataset_id")
            #predictions = predictions.reindex(index=df_CH["dataset_id"])
            
            predictions = predictions.reindex(index=new_index_order)
            predictions = predictions.reset_index()
            predictions["predicted_semantic_type"] = predictions["predicted_semantic_type"].fillna("None")
            predictions = predictions["predicted_semantic_type"].tolist()
        else:
            predictions = pd.read_csv(
                lfs[lf])
            predictions = predictions.set_index("dataset_id")
            #predictions = predictions.reindex(index=df_CH["dataset_id"])4
            predictions = predictions.reindex(index=new_index_order)
            predictions = predictions.reset_index()
            predictions["predicted_semantic_type"] = predictions["predicted_semantic_type"].fillna("None")
            predictions = predictions["predicted_semantic_type"].tolist()
        predictions = [
            label_enc.transform([x])[0] if x != "None" else -1
            for x in predictions
        ]
        L_train.append(predictions)
    L_train = np.array(L_train).transpose()

    if snorkel_label_model == "maj":
        # Majority-Vote Labeler
        from snorkel.labeling.model import MajorityLabelVoter

        label_model = MajorityLabelVoter(cardinality=len(valid_types))
    else:
        # Snorkels Label-Model
        from snorkel.labeling.model import LabelModel

        label_model = LabelModel(cardinality=len(valid_types), verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # predict the labels
    preds_train = label_model.predict(L=L_train)
    preds_train_str = [
        label_enc.inverse_transform([x])[0] if x != -1 else "None"
        for x in preds_train
    ]

    # store generated trainig data
    #df_unlabeled = pd.read_csv(lfs["header_to_sem_type_sim"])
    #df_unlabeled["predicted_semantic_type"] = preds_train_str
    df_unlabeled = pd.DataFrame({"table": [table.split("+")[0] for table in new_index_order], "column": [table.split("+")[1] for table in new_index_order], "dataset_id": new_index_order, "semanticType":None, "predicted_semantic_type": preds_train_str})

    ### read real semantic types to make validation possible
    with open(join(valid_headers_path, f"{corpus}_{os.environ['TYPENAME']}.json")) as f:
        valid_headers = json.load(f)
    for index, row in df_unlabeled.iterrows():
        df_unlabeled.at[index, "semanticType"] = valid_headers[row["table"]][row["column"]]["semanticType"]

    ###############################
    # for generate new train data
    # from unlabeled data
    ###############################
    if gen_train_data:
        # store results
        df_unlabeled.to_csv(join(
            "combined_LFs", "results",
            f"{corpus}_results_{snorkel_label_model}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
            index=False)

        df_unlabeled = df_unlabeled[
            df_unlabeled["predicted_semantic_type"] != "None"]
        df_unlabeled[[
            "table", "column", "dataset_id", "predicted_semantic_type"
        ]].to_csv(join(
            "combined_LFs", "gen_training_data",
            f"{corpus}_gen_training_data_all_combined_{snorkel_label_model}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
            index=False)

        # store validation
        from sklearn.metrics import classification_report
        class_report = class_report = classification_report(
            df_unlabeled["semanticType"].tolist(),
            df_unlabeled["predicted_semantic_type"].tolist(),
            output_dict=True,
            zero_division=0)
        # save classification_report
        with open(
                join(
                    "combined_LFs", "validation",
                    f"{corpus}_classification_report_unlabeled_{snorkel_label_model}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
                ), "w") as f:
            json.dump(class_report, f)

    ###############################
    # for validation on test data
    ###############################
    else:
        # store results
        df_unlabeled.to_csv(join(
            "combined_LFs", "results",
            f"{corpus}_results_test_{snorkel_label_model}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.csv"
        ),
            index=False)

        # store validation
        from sklearn.metrics import classification_report
        class_report = class_report = classification_report(
            df_unlabeled["semanticType"].tolist(),
            df_unlabeled["predicted_semantic_type"].tolist(),
            output_dict=True,
            zero_division=0)
        # save classification_report
        with open(
                join(
                    "combined_LFs", "validation",
                    f"{corpus}_classification_report_test_{snorkel_label_model}_{labeled_data_size}_{unlabeled_data_size}_{test_data_size}_{random_state}.json"
                ), "w") as f:
            json.dump(class_report, f)
