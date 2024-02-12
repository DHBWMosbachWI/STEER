from sklearn.preprocessing import LabelEncoder
import json
from sql_metadata import Parser
import pyarrow.parquet as pq
import os
from os.path import join
import sys
import glob
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)


def load_sportsDB_soccer_table(table_name: str, usecols, col_headers, frac=None, only_headers: bool = False):
    print(f"Loading Table: {table_name}....")
    if frac == None:
        df = pd.read_csv(
            join(os.environ["SportsDB_DIR"], "Soccer", "soccerPlayerScraping", table_name+".csv"), on_bad_lines="skip",
            header=None, names=col_headers, usecols=usecols, low_memory=False, skiprows=1)
    else:
        df = pd.read_csv(
            join(os.environ["SportsDB_DIR"], "Soccer", "soccerPlayerScraping", table_name+".csv"), on_bad_lines="skip",
            header=None, names=col_headers, usecols=usecols, low_memory=False, skiprows=1).sample(frac=frac)
    print(f"Loaded with {len(df)} rows")
    if only_headers:
        return df.columns
    else:
        return df


def get_all_gittables_tables_in_a_dir(domain="abstraction_tables", only_table_names=True):
    list_of_tables = glob.glob(
        os.path.join(os.environ["GITTABLES_DIR"], domain, "*.parquet"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-8], list_of_tables))
    return list_of_tables


def get_gittables_schema_iter(domain, table_list):
    for index, table in enumerate(table_list):
        try:
            pq_metadata = pq.read_schema(
                join(os.environ["GITTABLES_DIR"], domain, table+".parquet"))
        except:
            pass
        yield {"locator": domain, "table": table, "metadata": pq_metadata}


def get_all_publicbi_domains(only_domain_names: bool = False):
    list_of_domains = glob.glob(
        os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"), "*/"))
    if only_domain_names:
        list_of_domains = list(
            map(lambda x: os.path.split(os.path.split(x)[0])[-1],
                list_of_domains))
    return list_of_domains


def get_all_sportsDB_tables(only_table_names: bool = False):
    """
    """
    list_of_tables = glob.glob(os.path.join(os.environ.get(
        "SportsDB_DIR"), "Soccer", "soccerPlayerScraping", "*.csv"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-4], list_of_tables))
    return list_of_tables


def get_all_publicbi_tables(domain: str, only_table_names: bool = False):
    """
    Function that returns a list of all Table-Paths for a given domain in Public BI Benchmark Data Corpus

    Parameters
    ----------
    domain: str
        The domain of the tables you would like to have the list of all tables available in Public BI Benchmark
    only_table_names: bool
        if true the returned only consist the tablename and not the complete absolute table-path

    Returns
    -------
    The list of all absolute table-paths
    """
    list_of_tables = glob.glob(
        os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"), domain, "*.csv"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-4], list_of_tables))
    return list_of_tables


def getPublicBIColumnNames(domain, table):
    fd = open(join(os.environ["PUBLIC_BI_BENCHMARK"],
              domain, "tables", f"{table}.table.sql"), "r")
    sqlStmt = fd.read()
    fd.close()
    # print(sqlStmt)
    #res = sql_metadata.get_query_tokens(sqlStmt)
    columns = Parser(sqlStmt).columns
    return columns


def load_public_bi_table(domain, tablename, frac, with_header=True):
    df = pd.read_csv(os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"),
                                  domain, tablename + ".csv"),
                     sep="|", on_bad_lines="skip",
                     header=None).sample(frac=frac)
    if with_header:
        # df_header = pd.read_csv(os.path.join(
        #     os.environ.get("PUBLIC_BI_BENCHMARK"), domain, "samples",
        #     tablename + ".header.csv"),
        #                         sep="|")
        # df.columns = df_header.columns
        df.columns = getPublicBIColumnNames(domain, tablename)
        return df
    else:
        return df


def load_public_bi_table_by_cols(domain, tablename, usecols, col_headers, frac=None):
    print(f"Loading Table: {tablename}....")
    if frac == None:
        df = pd.read_csv(os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"),
                                      domain, tablename + ".csv"),
                         sep="|", on_bad_lines="skip",
                         header=None, names=col_headers, usecols=usecols, low_memory=False)
    else:
        df = pd.read_csv(os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"),
                                      domain, tablename + ".csv"),
                         sep="|", on_bad_lines="skip",
                         header=None, names=col_headers, usecols=usecols, low_memory=False).sample(frac=frac)
    print(f"Loaded with {len(df)} rows")
    # if with_header:
    #     # df_header = pd.read_csv(os.path.join(
    #     #     os.environ.get("PUBLIC_BI_BENCHMARK"), domain, "samples",
    #     #     tablename + ".header.csv"),
    #     #                         sep="|")
    #     # df.columns = df_header.columns
    #     columnNames = getPublicBIColumnNames(domain, tablename)
    #     df.columns = [columnNames[i] for i in usecols]
    #     return df
    # else:
    return df


def get_all_turl_tables(only_table_names: bool = False):
    list_of_tables = glob.glob(os.path.join(
        os.environ["TURL"], "tables", "*.csv"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-4], list_of_tables))
    return list_of_tables

# table colum loader of raw data


def load_turl_tablecolumn(dataset_id: str, sample=5) -> list:
    """ Function which loads a tablecolum of the turl data corpus and returns it as list. 
    It only returns random samples of the given number from the specified list 

    Parameters
    ----------
    dataset_id: str
        dataset_id with the construction table+column_number (eg. 23122.csv+column_0 to load column 0 from table 23122)
    sample: int
        number of samples to select randomly from the column

    Returns
    -------
    column_values: list
        all 
    """

    table_id = dataset_id.split("+")[0]
    column_id = dataset_id.split("+")[1].split("_")[1]
    df_column = pd.read_csv(join(os.environ["TURL"], "tables", table_id), usecols=[
                            int(column_id)]).sample(n=sample, replace=True, random_state=42)
    return df_column.iloc[:, 0].values.tolist()


def get_label_encoder() -> LabelEncoder:
    """
    Returns
    -------
        LabelEncoder: The LabelEncoder object for encoding the the current active semantic types

    """

    with open(join(os.environ["WORKING_DIR"], "data", "extract", "out", "valid_types", "types.json")) as f:
        valid_types = json.load(f)[os.environ["TYPENAME"]]

    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    return label_enc
