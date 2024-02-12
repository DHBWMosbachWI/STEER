##########
## To-Dos
##########
# - add posibility to only calculate the word embeddings for the columns specified in an valid header file.
#   In actual state, embeddings for all cols in the table were calculated
# - Problems with memory when calculating word emeddings for fraction of 1.0

import os
from dotenv import load_dotenv
load_dotenv(override=True)
import sys

sys.path.insert(0, os.environ["WORKING_DIR"])

import tensorflow_hub as hub
import tensorflow_text
from tqdm import tqdm
import pandas as pd
import configargparse

from os.path import join
import random
from data_loader.utils import get_all_publicbi_domains, get_all_publicbi_tables, get_all_turl_tables, get_all_gittables_tables_in_a_dir, get_all_sportsDB_tables
from collections import OrderedDict
import errno

import numpy as np
import random
from functools import partial


def get_gittables_df_iter(domain="abstraction_tables", table_list=[]):
    for index, tablename in enumerate(get_all_gittables_tables_in_a_dir(domain=domain)):
        print(f"Start reading {tablename}")
        df = pd.read_parquet(os.path.join(os.environ["GITTABLES_DIR"], domain, tablename + ".parquet"))
        print(f"End reading {tablename}")
        
        yield {"df": df, "domain": domain, "table": tablename+".parquet"}
        
def get_sportsDB_df_iter(table_list=[]):
    for index, tablename in enumerate(get_all_sportsDB_tables(only_table_names=True)):
        if len(table_list) > 0:
            if tablename not in table_list:
                continue
        print(f"Start reading {tablename}")
        df = pd.read_csv(os.path.join(os.environ["SportsDB_DIR"], "Soccer", "soccerPlayerScraping", tablename + ".csv"))
        print(f"End reading {tablename} / Length: {len(df)}")

        yield {"df": df, "domain": None, "table": tablename+".csv"}


def get_turl_df_iter(table_list=[]):
    for index, tablename in enumerate(
            get_all_turl_tables(only_table_names=True)):
        # if there is a list of tables specified, then check if the current tablename is in this list. if not continue to the next iter step
        if len(table_list) > 0:
            if tablename not in table_list:
                continue
        print(f"Start reading {tablename}")
        df = pd.read_csv(os.path.join(os.environ["TURL"], tablename + ".csv"))
        print(f"End reading {tablename} / Length: {len(df)}")

        yield {"df": df, "domain": None, "table": tablename+".csv"}


def get_public_bi_benchmark_df_iter(frac, table_list=[], with_header=False):

    for index, domain in enumerate(
            get_all_publicbi_domains(only_domain_names=True)):
        for tablename in get_all_publicbi_tables(domain=domain,
                                                 only_table_names=True):
            # if there is a list of tables specified, then check if the current tablename is in this list. if not continue to the next iter step
            if len(table_list) > 0:
                if tablename not in table_list:
                    continue
            print(f"Start reading {tablename}")
            df = pd.read_csv(
                os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"), domain,
                             tablename + ".csv"),
                sep="|",
                error_bad_lines=False,
                warn_bad_lines=False,
                # other method to read only y fraction of the available data rows
                #skiprows=lambda i: i > 0 and random.random() > frac,
                header=None).sample(frac=frac)
            print(f"End reading {tablename} / Length: {len(df)}")
            if with_header:
                df_header = pd.read_csv(os.path.join(
                    os.environ.get("PUBLIC_BI_BENCHMARK"), domain, "samples",
                    tablename + ".header.csv"),
                                        sep="|")
                df.columns = df_header.columns
                yield {"df": df, "domain": domain, "table": tablename}
            else:
                yield {"df": df, "domain": domain, "table": tablename}


# word embedding model from tensorflow hub "Google USE"
# embed = hub.load(
#     "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

embed = hub.load(
    join(os.environ["WORKING_DIR"], "emb_clus", "word_embedding", "models",
         "google_use_3"))


def embedding_for_one_col(column: list):
    if len(column) == 0:
        return None
    #calc the embedding for each value
    embeddings = embed(column)

    # calc the average vector
    averageEmbedding = np.mean(embeddings.numpy(), axis=0).tolist()

    return averageEmbedding


def embedding_for_one_table(df: pd.DataFrame):
    result = {}
    # itterate over the table-columns and get the embedding for each column
    for column in df.columns:
        column_embedding = embedding_for_one_col(df[column].tolist())
        result[column] = column_embedding
    return result


def extract_word_embedding(n_samples: int, df_dic: dict):
    df, domain, table = df_dic["df"], df_dic["domain"], df_dic["table"]

    all_results = []

    with tqdm(total=len(df.columns)) as pbar:
        for col_num, col_name in enumerate(df.columns):
            all_results.append(OrderedDict())

            all_results[col_num]["table"] = table
            all_results[col_num]["column"] = "column_" + str(col_num)
            all_results[col_num]["dataset_id"] = table + "+" + "column_" + str(
                col_num)

            #print(all_results)

            # word embedding
            n_values = len(df[col_name].dropna())
            # if number of values more than the given n_samples to use
            if n_samples is None:
                all_results[col_num]["word_embedding"] = embedding_for_one_col(
                    df[col_name].dropna().apply(str).tolist())
            else:
                if n_values > n_samples:
                    n_values = n_samples
                all_results[col_num]["word_embedding"] = embedding_for_one_col(
                    random.sample(df[col_name].dropna().apply(str).tolist(),
                                  n_values))
            all_results[col_num]["used_col_values"] = n_values

            pbar.update(1)

    return pd.DataFrame(all_results)


valid_headers_path = join(os.environ["WORKING_DIR"], "data", "extract", "out",
                          "valid_headers")

if __name__ == "__main__":
    __spec__ = "Modulespec (name = 'builtins', loader = < class '_frozen_importlib.builtinimporter' > )"

    cache_size = 1

    parser = configargparse.ArgParser()

    parser.add("--corpus",
               type=str,
               default="public_bi",
               help="name of the data corpus")
    parser.add(
        "--fraction",
        type=float,
        default=1.0,
        help=
        "fraction of the data used to build the word embedding for one column")
    parser.add("--num_processes", type=int, default=4)
    parser.add("--add_to_existing_output_file", type=bool, default=False)
    parser.add("--table_list", type=str, nargs="*", default=[])
    parser.add("--n_samples", type=int, default=None)

    args = parser.parse_args()
    corpus = args.corpus
    fraction = args.fraction
    num_processes = args.num_processes
    add_to_existing_output_file = args.add_to_existing_output_file
    table_list = args.table_list
    n_samples = args.n_samples

    #############
    ## Load Data
    #############

    # load specific raw data regarding to the given corpus name
    if corpus.startswith("public"):
        if len(table_list) > 0:
            raw_df_iter = get_public_bi_benchmark_df_iter(
                frac=fraction, table_list=table_list)
        else:
            raw_df_iter = get_public_bi_benchmark_df_iter(frac=fraction)
    elif corpus.startswith("turl"):
        if len(table_list) > 0:
            raw_df_iter = get_turl_df_iter(table_list=table_list)
        else:
            raw_df_iter = get_turl_df_iter()
    elif corpus.startswith("gittables-abstraction"):
        if len(table_list) > 0:
            raw_df_iter = get_gittables_df_iter(domain="abstraction_tables", table_list=table_list)
        else:
            raw_df_iter = get_gittables_df_iter(domain="abstraction_tables")
    elif corpus.startswith("sports"):
        if len(table_list) > 0:
            raw_df_iter = get_sportsDB_df_iter(table_list=table_list)
        else:
            raw_df_iter = get_sportsDB_df_iter()

    #####################
    ## Set-up outputfile
    #####################

    output_file_path = join(os.environ["WORKING_DIR"], "emb_clus",
                            "word_embedding", "out",
                            f"{corpus}_{fraction}_df.csv")

    if add_to_existing_output_file:
        # check if there is a existing file
        if os.path.isfile(output_file_path):
            # load exising outputs
            #df = pd.read_csv(output_file_path)
            header, mode = False, "a"
        else:
            print(
                f"You have specified a output file to override, but this file does not exist"
            )
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    output_file_path)
    else:
        header, mode = True, "w"

    ##################################
    ## distribute the task using Pool
    ##################################

    output_file_path = join(os.environ["WORKING_DIR"], "emb_clus",
                            "word_embedding", "out",
                            f"{corpus}_{fraction}_df.csv")

    counter = 0
    #header, mode = True, 'w'
    col_counter = 0
    cache = []
    for raw_df in raw_df_iter:
        df_features = extract_word_embedding(n_samples, raw_df)
        counter += 1
        cache.append(df_features)
        if counter % cache_size == 0:
            df = pd.concat(cache)
            df.to_csv(output_file_path, header=header, index=False, mode=mode)
            col_counter += len(df)
            header, mode = False, 'a'
            cache = []
        
        
    #save the last cache
    if len(cache) > 0:
        df = pd.concat(cache)
        df.to_csv(output_file_path, header=header, index=False, mode=mode)

    print("Number of columns: {}".format(col_counter))
