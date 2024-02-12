import os
from os.path import join
import configargparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
load_dotenv(override=True)

valid_header_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                         "valid_headers")

valid_types_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                        "valid_types", "types.json")

save_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                 "labeled_unlabeled_test_split")

#RANDOM_STATE = 2

if __name__ == "__main__":

    parser = configargparse.ArgParser()
    # absolut number of columns per semantic type which shouls be selected as labeled type
    parser.add("--labeled_size", type=int, default=1)
    #parser.add("--unlabeled_size", type=float, default=0.7)
    parser.add("--test_size", type=float, default=0.2)
    # .json format and also .csv format from sato of valid header are possible
    parser.add("--valid_headers", type=str, default="public_bi_type78.json")
    parser.add("--corpus", type=str, default="public_bi")
    parser.add("--random_state", type=int, default=2)

    args = parser.parse_args()
    labeled_size = args.labeled_size
    unlabeled_size = "absolute"
    test_size = args.test_size
    valid_headers_file = args.valid_headers
    corpus = args.corpus
    RANDOM_STATE = args.random_state

    # load array of valid types
    with open(valid_types_path, "r") as f:
        valid_types = json.load(f)[os.environ["TYPENAME"]]

    # build a Label Encoder
    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    # load the valid headers with real sem. types
    valid_headers = join(valid_header_path, valid_headers_file)

    if valid_headers_file.endswith(".csv"):
        # read df
        valid_header_df = pd.read_csv(valid_headers)
        valid_header_df["field_list"] = valid_header_df["field_list"].apply(eval)
        valid_header_df["field_names"] = valid_header_df["field_names"].apply(eval)
        # def label_enc_apply(x):
        #     return label_enc.inverse_transform(x)
        # valid_header_df["field_names"] = valid_header_df["field_names"].apply(label_enc_apply)


        # generate array of all dataset_ids
        pbar = tqdm(total=len(valid_header_df))
        dataset_ids = []
        for index, row in valid_header_df.iterrows():
            #print(index,len(valid_header_df))
            dataset_id = row["dataset_id"]
            field_list = row["field_list"]
            field_names = row["field_names"]
            field_names = label_enc.inverse_transform(row["field_names"])
            # logic check
            assert len(field_list) == len(
                field_names
            ), "field_list and field_name should be in the same lenght, because field_names are the semantic types of each column in field_list"
            
            for index_col, column in enumerate(field_list):
                dataset_ids.append([dataset_id, column, field_names[index_col]])

            pbar.update(1)
        pbar.close()
        df = pd.DataFrame(dataset_ids, columns=["dataset_id", "column", "semantic_type"])
        df["column"] = df["column"].apply(str)
    # part of valid hedaer in .json
    # else:
    #     with open(valid_headers, "r") as file:
    #         valid_headers = json.load(file)

    df_labeled_unlabeled, df_test = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
    print(f"Total: {len(df)}, Labeled_Unlabeled: {len(df_labeled_unlabeled)}, Test:{len(df_test)}")

    # select a specific number of columns per semantic type
    #print(df)
    df_labeled = pd.DataFrame()
    for ind, valid_type in enumerate(valid_types):
        df_acc = df_labeled_unlabeled[df_labeled_unlabeled["semantic_type"] == valid_type]
        if len(df_acc) >= labeled_size:
            df_acc = df_acc.sample(n=labeled_size, random_state=RANDOM_STATE)

            df_labeled = df_labeled.append(df_acc)
        elif len(df_acc) > 0:
            df_acc = df_acc.sample(n=len(df_acc), random_state=RANDOM_STATE)
            df_labeled = df_labeled.append(df_acc)
        else:
            print(f"Not enough columns for type {valid_type}. Columns with that type: {len(df_acc)}. Wanted columns as labeled data: {labeled_size}")
        
        #print(labeled_data)
    
    # build df_unlabeled by delete all labeled data from labeled_unlabeled
    df_unlabeled = df_labeled_unlabeled[~df_labeled_unlabeled.apply(tuple, 1).isin(df_labeled.apply(tuple,1))]
    print(f"Total: {len(df)}, Labeled: {len(df_labeled)}, Unlabeled: {len(df_unlabeled)}, Test: {len(df_test)}")

    # create the array for each split and the .json file
    labeled_data = (df_labeled["dataset_id"]+"+column_"+df_labeled["column"]).tolist()
    unlabeled_data = (df_unlabeled["dataset_id"]+"+column_"+df_unlabeled["column"]).tolist()
    test_data = (df_test["dataset_id"]+"+column_"+df_test["column"]).tolist()

    assert len(labeled_data) == len(df_labeled)
    assert len(unlabeled_data) == len(df_unlabeled)
    assert len(test_data) == len(df_test)    

    # save splitted datasets
    split_dic = {}
    split_dic[f"labeled{labeled_size}"] = list(labeled_data)
    split_dic[f"unlabeled"] = list(unlabeled_data)
    split_dic[f"test{(len(test_data)/len(df)*100):.1f}"] = list(test_data)

    with open(
            join(
                save_path,
                f"{corpus}_{labeled_size}_{unlabeled_size}_{(len(test_data)/len(df)*100):.1f}_{RANDOM_STATE}.json"
            ), "w") as f:
        json.dump(split_dic, f)