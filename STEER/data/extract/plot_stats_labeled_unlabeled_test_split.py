import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from os.path import join
import os
import configargparse

from multiprocesspandas import applyparallel

os.environ["WORKING_DIR"] = "D:\\semantic_data_lake\\semantic_data_lake"

valid_header_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                         "valid_headers")

if __name__ == "__main__":

    parser = configargparse.ArgParser()
    parser.add("--labeled", type=float, default=0.1)
    parser.add("--unlabeled", type=float, default=0.7)
    parser.add("--test", type=float, default=0.2)
    parser.add("--corpus", type=str, default="public_bi")
    parser.add("--integer", type=bool, default=False)
    #parser.add("--valid_headers", type=str, default="public_bi_type78.json")

    args = parser.parse_args()
    integer = args.integer
    labeled = args.labeled
    test = args.test
    unlabeled = args.unlabeled
    if integer:
        labeled = int(args.labeled)
        test = int(args.test)
        unlabeled = int(args.unlabeled)

    corpus = args.corpus
    valid_headers_file = f"{corpus}_type_turl_valid.json"

    # load the valid headers with read sem. types
    valid_headers = join(valid_header_path, valid_headers_file)


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

    valid_header_df["dataset_id"] = valid_header_df["table"]+"+"+valid_header_df["column"]
    print(valid_header_df.head())


    #valid_header_df = valid_header_df.iloc[:500]

    # load laebeled_unlabeled_test_split
    split_file_path = join(os.environ["WORKING_DIR"], "data", "extract", "out", "labeled_unlabeled_test_split")
    split_file = f"{corpus}_{labeled}_{unlabeled}_{test}.json"

    with open(join(split_file_path, split_file), "r") as file:
        split_file = json.load(file)

    df_split_labeled = pd.DataFrame({"dataset_id": split_file[f"labeled{labeled}"]})
    df_split_labeled["split_part"] = "labeled"
    df_split_unlabeled = pd.DataFrame({"dataset_id": split_file[f"unlabeled{unlabeled}"]})
    df_split_unlabeled["split_part"] = "unlabeled"
    df_split_test = pd.DataFrame({"dataset_id": split_file[f"test{test}"]})
    df_split_test["split_part"] = "test"

    df_split = pd.concat([df_split_labeled, df_split_unlabeled, df_split_test])

    #df_split.head(-20)

    df = valid_header_df.join(df_split.set_index("dataset_id"), on="dataset_id")

    save_path = join(os.environ["WORKING_DIR"], "data", "extract", "out", "labeled_unlabeled_test_split")

    plt.figure(figsize=(10,30))

    ax = df[df["split_part"] == "labeled"].groupby(["semanticType"]).size().sort_values().plot.barh()

    for container in ax.containers:
        ax.bar_label(container)

    plt.grid()
    plt.title(f"#semantic types of labaled data ({labeled}/{unlabeled}/{test})")
    plt.savefig(join(save_path, f"turl_{labeled}_{unlabeled}_{test}_stats_labeled.png"), bbox_inches="tight", facecolor="white", transparent=False)
    #plt.show()

    plt.figure(figsize=(10,30))

    ax = df[df["split_part"] == "unlabeled"].groupby(["semanticType"]).size().sort_values().plot.barh()

    for container in ax.containers:
        ax.bar_label(container)

    plt.grid()
    plt.title(f"#semantic types of unlabaled data ({labeled}/{unlabeled}/{test})")
    plt.savefig(join(save_path, f"turl_{labeled}_{unlabeled}_{test}_stats_unlabeled.png"), bbox_inches="tight", facecolor="white", transparent=False)
    #plt.show()

    plt.figure(figsize=(10,30))

    ax = df[df["split_part"] == "test"].groupby(["semanticType"]).size().sort_values().plot.barh()

    for container in ax.containers:
        ax.bar_label(container)

    plt.grid()
    plt.title(f"#semantic types of test data ({labeled}/{unlabeled}/{test})")
    plt.savefig(join(save_path, f"turl_{labeled}_{unlabeled}_{test}_stats_test.png"), bbox_inches="tight", facecolor="white", transparent=False)
    #plt.show()

