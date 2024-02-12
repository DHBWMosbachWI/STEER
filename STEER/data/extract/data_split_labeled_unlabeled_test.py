import os
from os.path import join
import configargparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv(override=True)

valid_header_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                         "valid_headers")

save_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                 "labeled_unlabeled_test_split")

if __name__ == "__main__":

    parser = configargparse.ArgParser()
    parser.add("--labeled_size", type=float, default=0.1)
    parser.add("--unlabeled_size", type=float, default=0.7)
    parser.add("--test_size", type=float, default=0.2)
    # .json format and also .csv format from sato of valid header are possible
    parser.add("--valid_headers", type=str, default="public_bi_type78.json")
    parser.add("--corpus", type=str, default="public_bi")
    parser.add("--random_state", type=int, default=2)

    args = parser.parse_args()
    labeled_size = args.labeled_size
    unlabeled_size = args.unlabeled_size
    test_size = args.test_size
    valid_headers_file = args.valid_headers
    corpus = args.corpus
    RANDOM_STATE = args.random_state

    # load the valid headers with read sem. types
    valid_headers = join(valid_header_path, valid_headers_file)
    
    if valid_headers_file.endswith(".csv"):
        # read df
        valid_header_df = pd.read_csv(valid_headers)

        # generate array of all dataset_ids
        dataset_ids = []
        for index, row in valid_header_df.iterrows():
            dataset_id = row["dataset_id"]
            field_list = eval(row["field_list"])
            for column in field_list:
                dataset_ids.append(f"{dataset_id}+column_{column}")
        pass
    else:    
        with open(valid_headers, "r") as file:
            valid_headers = json.load(file)

        # transform valid header into df to make splitable
        valid_header_df_data = []
        for table in valid_headers.keys():
            for column in valid_headers[table].keys():
                valid_header_df_data.append(
                    [table, column, valid_headers[table][column]["semanticType"]])
        valid_header_df = pd.DataFrame(valid_header_df_data,
                                    columns=["table", "column", "semanticType"])

        # generate array of all dataset_ids
        dataset_ids = []
        for (table, column) in zip(list(valid_header_df["table"]),
                                list(valid_header_df["column"])):
            dataset_ids.append(table + "+" + column)

    #print(dataset_ids)

    # split the data in three splits: labeled, unlabeled, test
    labeled_unlabeled, test = train_test_split(dataset_ids,
                                               test_size=test_size,
                                               random_state=RANDOM_STATE)
    #print(len(labeled_unlabeled), len(test))
    labeled, unlabeled = train_test_split(labeled_unlabeled,
                                          test_size=unlabeled_size /
                                          (1 - test_size),
                                          random_state=RANDOM_STATE)
    print(
        f"Labeled data: {len(labeled)}({(len(labeled)/len(dataset_ids)):.2f}), Unlabeled data: {len(unlabeled)}({(len(unlabeled)/len(dataset_ids)):.2f}), Test data: {len(test)}({(len(test)/len(dataset_ids)):.2f})"
    )

    # save splitted datasets
    split_dic = {}
    split_dic[f"labeled{(len(labeled)/len(dataset_ids)*100):.1f}"] = list(
        labeled)
    split_dic[f"unlabeled{(len(unlabeled)/len(dataset_ids)*100):.1f}"] = list(
        unlabeled)
    split_dic[f"test{(len(test)/len(dataset_ids)*100):.1f}"] = list(test)

    with open(
            join(
                save_path,
                f"{corpus}_{(len(labeled)/len(dataset_ids)*100):.1f}_{(len(unlabeled)/len(dataset_ids)*100):.1f}_{(len(test)/len(dataset_ids)*100):.1f}_{RANDOM_STATE}.json"
            ), "w") as f:
        json.dump(split_dic, f)