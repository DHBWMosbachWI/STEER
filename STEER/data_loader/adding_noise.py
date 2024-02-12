import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import mean, stddev

# create Spark Config
conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("MLB-similarity-calc")
# create a SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()


def add_additive_gaussian_noise(table_name, data_perc_with_noise, random_seed, numeric_attributes):
    df_original = spark.sql(f"SELECT * from {table_name}")
    df_without_noise, df_to_add_noise = df_original.randomSplit(
        [1-data_perc_with_noise, data_perc_with_noise], random_seed)
    # print(df.count(),df_without_noise.count(), df_to_add_noise.count())
    df_to_add_noise_pd = df_to_add_noise.toPandas()

    # adding noise to every numeric column
    for numeric_attribute in numeric_attributes[table_name]:
        # print(numeric_attribute)
        # if numeric_attribute != "AB":
        #  break
        df_stats = df_original.select(mean(f"`{numeric_attribute}`").alias(
            "mean"), stddev(f"`{numeric_attribute}`").alias("stddev")).collect()[0]
        df_col_mean = df_stats["mean"]
        df_col_stddev = df_stats["stddev"]

        for index, row in df_to_add_noise_pd.iterrows():
            if row[f"{numeric_attribute}"] != None:  # filter None values
                df_to_add_noise_pd.at[index, f"{numeric_attribute}"] = float(
                    row[f"{numeric_attribute}"]) + float(np.random.normal(0, df_col_stddev, 1)[0])

    df_to_add = spark.createDataFrame(df_to_add_noise_pd)
    df_noise_full = df_without_noise.union(
        spark.createDataFrame(df_to_add_noise_pd))
    return df_noise_full
