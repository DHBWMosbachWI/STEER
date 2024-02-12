import os
import sys
import glob
import pandas as pd
from helper_functions import translate_header_file_to_list, cast_datatypes, check_attribute_completeness
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame

# create Spark Config
conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("MLB-similarity-calc")
# create a SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()

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


def load_public_bi_tables_in_spark_temp_view(domain_name, quantity=None, sample=False):
    # dict of string attributes for each table
    string_attributes = {}
    numeric_attributes = {}
    list_of_all_MLB_tables = get_all_publicbi_tables(domain_name, True)
    file_path = os.path.join(os.environ.get(
        "PUBLIC_BI_BENCHMARK"), domain_name)
    if quantity != None:
        list_of_all_MLB_tables = list_of_all_MLB_tables[:quantity]
    for table_name in list_of_all_MLB_tables:
        if sample:
            data_file = os.path.join(
                file_path, "samples", table_name + ".sample" + ".csv")
        else:
            data_file = os.path.join(file_path, table_name + ".csv")
        header_file = os.path.join(
            file_path, "samples", table_name + ".header.csv")
        datatype_file = os.path.join(
            file_path, "samples", table_name + ".datatypes.csv")
        # create a DataFrame using an ifered Schema
        orig_df = spark.read.option("header", "false") \
            .option("inferSchema", "true") \
            .option("delimiter", "|") \
            .csv(data_file).toDF(*translate_header_file_to_list(header_file))
        df = cast_datatypes(datatype_file, orig_df)
        # compare_schemas(orig_df, df)
        df.createOrReplaceTempView(table_name)
        string_attributes[table_name] = list(filter(lambda x: not x.startswith("Calculation"),
                                                    map(lambda x: x[0], filter(lambda tupel: tupel[1] == 'string', df.dtypes))))
        numeric_attributes[table_name] = list(filter(lambda x: not x.startswith("Calculation"),
                                                     map(lambda x: x[0],
                                                         filter(lambda tupel: tupel[1] == 'double' or
                                                                tupel[1] == 'int' or tupel[1].startswith('decimal'), df.dtypes))))
        check_attribute_completeness(df.columns, string_attributes[table_name],
                                     numeric_attributes[table_name])
    return (string_attributes, numeric_attributes)
