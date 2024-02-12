import os
from data_loader.utils import get_all_publicbi_tables
from helper_functions import pair_permutations_ordered, cast_datatypes, translate_datatype_file_to_list, check_attribute_completeness, translate_header_file_to_list
from scipy.stats import wasserstein_distance

from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, StructType, StructField
from pyspark.sql.functions import udf, col, pandas_udf, PandasUDFType, collect_list, count, avg, lit, mean, stddev, monotonically_increasing_id, row_number
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame

from tqdm import tqdm

# create Spark Config
conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("MLB-similarity-calc")
# create a SparkSession
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# create and register UDF-Function to calc EMD-Distance


@udf(returnType=FloatType())
def emd_UDF(col1, col2):
    if len(col1) == 0 or len(col2) == 0:
        return None
    return float(wasserstein_distance(col1, col2))


spark.udf.register("emd_UDF", emd_UDF)


# approach 0 without join
def appr0_calc_similarities(outer, inner, numeric_attributes, sel_attr=None):
    """
    Function that calculates the earth movers distance between all column-combinations of two given tables

    Parameters
    ----------
    outer: str
        Tablename that exists in the current spark context as Temp-View
    inner: str
        Tablename that exists in the current spark context as Temp-View
    sel_attr: None/array
        if None than all columns combination are calculates, 
        else you can provide a list of column-names from the inner table to only select the combinations with that specific columns   

    Returns
    -------
    df: spark.DataFrame
        all results of emd betwenn all column-combinations
    """
    intersecting_attr = list(
        set(numeric_attributes[inner]) & set(numeric_attributes[outer]))
    sqlOuter = spark.sql("SELECT * FROM " + outer)
    # filter out null tupels with null values
    sqlOuter = sqlOuter.dropna(subset=list(
        map(lambda cur_col: "`{cur_col}`".format(cur_col=cur_col),
            sqlOuter.columns)))
    sqlInner = spark.sql("SELECT * FROM "+inner)
    # filter out null tupels with null values
    sqlInner = sqlInner.dropna(subset=list(
        map(lambda cur_col: "`{cur_col}`".format(cur_col=cur_col),
            sqlInner.columns)))
    attr_variations = pair_permutations_ordered(intersecting_attr)

    # selsect specific attr_variation with a specific attribute included
    if sel_attr == None:
        sel_attr_variations = attr_variations
    else:
        sel_attr = sel_attr
        sel_attr_variations = list(
            filter(lambda x: x[1] in sel_attr, attr_variations))

    result = []
    pbar = tqdm(total=len(sel_attr_variations))
    for index_attr, curr_item in enumerate(sel_attr_variations):
        first_attr = curr_item[0]
        second_attr = curr_item[1]
        emd = wasserstein_distance(sqlOuter.select(collect_list(f"`{first_attr}`")).collect()[
                                   0][0], sqlInner.select(collect_list(f"`{second_attr}`")).collect()[0][0])
        result.append([outer, first_attr, inner, second_attr, float(emd)])
        pbar.update(1)
    if len(result) == 0:
        return None
    resultDF = spark.createDataFrame(result).toDF(
        "OUTER", "OUTER_ATTR", "INNER", "INNER_ATTR", "EMD")
    return resultDF

# approach 1


def appr1_calc_similarities(outer, inner, string_attributes, numeric_attributes, sel_attr=None):
    """
    Function that calculates the earth movers distance between all column-combinations of two given tables

    Parameters
    ----------
    outer: str
        Tablename that exists in the current spark context as Temp-View
    inner: str
        Tablename that exists in the current spark context as Temp-View
    sel_attr: None/array
        if None than all columns combination are calculates, 
        else you can provide a list of column-names from the inner table to only select the combinations with that specific columns   

    Returns
    -------
    df: spark.DataFrame
        all results of emd betwenn all column-combinations
    """
    join_attributes = list(
        set(string_attributes[inner]) & set(string_attributes[outer]))
    join_condition = "ON (" + " AND ".join(map(lambda join_att: "o.`{join_att}` = i.`{join_att}`".format(join_att=join_att),
                                           join_attributes))
    intersecting_attr = list(
        set(numeric_attributes[inner]) & set(numeric_attributes[outer]))

    # create projection list
    projection_list = " , ".join(
        map(lambda attr: "o.`{attr}` as `{attr}`".format(attr=attr),
            join_attributes)
    ) + " , " + " , ".join(
        map(
            lambda attr: "o.`{attr}` as `o.{attr}` , i.`{attr}` as `i.{attr}`".
            format(attr=attr), intersecting_attr))
    sqlDF = spark.sql("SELECT "+projection_list+" FROM " + outer + " o JOIN " +
                      inner + " i " + join_condition+")")
    # filter out null tupels with null values
    sqlDF = sqlDF.dropna(subset=list(
        map(lambda cur_col: "`{cur_col}`".format(cur_col=cur_col),
            sqlDF.columns)))

    attr_variations = pair_permutations_ordered(intersecting_attr)

    # selsect specific attr_variation with a specific attribute included
    if sel_attr == None:
        sel_attr_variations = attr_variations
    else:
        sel_attr = sel_attr
        sel_attr_variations = list(
            filter(lambda x: x[1] in sel_attr, attr_variations))

    result = []
    pbar = tqdm(total=len(sel_attr_variations))
    for index_attr, curr_item in enumerate(sel_attr_variations):
        print(str(index_attr) + "/" + str(len(sel_attr_variations)))
        first_attr = curr_item[0]
        second_attr = curr_item[1]
        emd = wasserstein_distance(sqlDF.select(collect_list(f"`o.{first_attr}`")).collect()[
                                   0][0], sqlDF.select(collect_list(f"`i.{second_attr}`")).collect()[0][0])
        result.append([outer, first_attr, inner, second_attr, float(emd)])
        pbar.update(1)
    if len(result) == 0:
        return None
    resultDF = spark.createDataFrame(result).toDF(
        "OUTER", "OUTER_ATTR", "INNER", "INNER_ATTR", "EMD")
    return resultDF

# join over string attributes


def join_over_string_attr(outer, inner, string_attributes, numeric_attributes):
    """
    Function that join two tables over their string attributes

    Parameters
    ----------
    outer: str
        Tablename that exists in the current spark context as Temp-View
    inner: str
        Tablename that exists in the current spark context as Temp-View
    Returns
    -------
    df: spark.DataFrame
        the result Dataframe after join over string attributes of the two given tables
    """
    outer = outer
    inner = inner
    join_attributes = list(
        set(string_attributes[inner]) & set(string_attributes[outer]))
    join_condition = "ON (" + " AND ".join(map(lambda join_att: "o.`{join_att}` = i.`{join_att}`".format(join_att=join_att),
                                           join_attributes))
    intersecting_attr = list(
        set(numeric_attributes[inner]) & set(numeric_attributes[outer]))

    # create projection list
    projection_list = " , ".join(
        map(lambda attr: "o.`{attr}` as `{attr}`".format(attr=attr),
            join_attributes)
    ) + " , " + " , ".join(
        map(
            lambda attr: "o.`{attr}` as `o.{attr}` , i.`{attr}` as `i.{attr}`".
            format(attr=attr), intersecting_attr))
    sqlDF = spark.sql("SELECT "+projection_list+" FROM " + outer + " o JOIN " +
                      inner + " i " + join_condition+")")
    # filter out null tupels with null values
    sqlDF = sqlDF.dropna(subset=list(
        map(lambda cur_col: "`{cur_col}`".format(cur_col=cur_col),
            sqlDF.columns)))

    return sqlDF


# appraoch 3
def appr3_calc_similarities(outer, inner, string_attributes, numeric_attributes, max_group_count, sel_attr=None):
    """
    Function that calculates the earth movers distance between all column-combinations of two given tables

    Parameters
    ----------
    outer: str
        Tablename that exists in the current spark context as Temp-View
    inner: str
        Tablename that exists in the current spark context as Temp-View
    max_group_count: int
        After join and group over the string attriutes, only consider builded groups with the maximum amount of max_group_count to calculate the earth movers distance between columns
    sel_attr: None/array
        if None than all columns combination are calculates, 
        else you can provide a list of column-names from the inner table to only select the combinations with that specific columns   

    Returns
    -------
    df: spark.DataFrame
        all results of emd betwenn all column-combinations
    """
    outer = outer
    inner = inner
    join_attributes = list(
        set(string_attributes[inner]) & set(string_attributes[outer]))
    join_condition = "ON (" + " AND ".join(map(lambda join_att: "o.`{join_att}` = i.`{join_att}`".format(join_att=join_att),
                                           join_attributes))
    intersecting_attr = list(
        set(numeric_attributes[inner]) & set(numeric_attributes[outer]))

    # create projection list
    projection_list = " , ".join(
        map(lambda attr: "o.`{attr}` as `{attr}`".format(attr=attr),
            join_attributes)
    ) + " , " + " , ".join(
        map(
            lambda attr: "o.`{attr}` as `o.{attr}` , i.`{attr}` as `i.{attr}`".
            format(attr=attr), intersecting_attr))
    sqlDF = spark.sql("SELECT "+projection_list+" FROM " + outer + " o JOIN " +
                      inner + " i " + join_condition+")")
    # filter out null tupels with null values
    sqlDF = sqlDF.dropna(subset=list(
        map(lambda cur_col: "`{cur_col}`".format(cur_col=cur_col),
            sqlDF.columns)))

    attr_variations = pair_permutations_ordered(intersecting_attr)

    # selsect specific attr_variation with a specific attribute included
    if sel_attr == None:
        sel_attr_variations = attr_variations
    else:
        sel_attr = sel_attr
        sel_attr_variations = list(
            filter(lambda x: x[1] in sel_attr, attr_variations))

    pbar = tqdm(total=len(sel_attr_variations))
    for index_attr, curr_item in enumerate(sel_attr_variations):
        #print(str(index_attr) + "/" + str(len(sel_attr_variations)))
        first_attr = curr_item[0]
        second_attr = curr_item[1]
        # print(first_attr)
        # print(second_attr)
        if index_attr == 0:
            curDF = sqlDF.groupby(join_attributes).agg(
                emd_UDF(collect_list("`o.{first_attr}`".format(first_attr=first_attr)),
                        collect_list("`i.{second_attr}`".format(second_attr=second_attr))).alias("EMD"),
                count("`i.{second_attr}`".format(second_attr=second_attr)).alias("count")).where(col("count") <= max_group_count).select(col("EMD"), col("count"))
            curDF = curDF.withColumn("OUTER", lit(outer)).withColumn(
                "OUTER_ATTR",
                lit(first_attr)).withColumn("INNER", lit(inner)).withColumn(
                    "INNER_ATTR", lit(second_attr))
        else:
            newDF = sqlDF.groupby(join_attributes).agg(
                emd_UDF(collect_list("`o.{first_attr}`".format(first_attr=first_attr)),
                        collect_list("`i.{second_attr}`".format(second_attr=second_attr))).alias("EMD"),
                count("`i.{second_attr}`".format(second_attr=second_attr)).alias("count")).where(col("count") <= max_group_count).select(col("EMD"), col("count"))
            newDF = newDF.withColumn("OUTER", lit(outer)).withColumn(
                "OUTER_ATTR",
                lit(first_attr)).withColumn("INNER", lit(inner)).withColumn(
                    "INNER_ATTR", lit(second_attr))
            curDF = curDF.union(newDF)
        pbar.update(1)
    curDF = curDF.select("*").groupBy("OUTER", "OUTER_ATTR",
                                      "INNER", "INNER_ATTR").avg("EMD").alias("EMD")
    return curDF

# approach 4


def appr4_calc_similarities(outer, inner, string_attributes, numeric_attributes, max_group_count, sel_attr=None):
    """
    Function that calculates the earth movers distance between all column-combinations of two given tables

    Parameters
    ----------
    outer: str
        Tablename that exists in the current spark context as Temp-View
    inner: str
        Tablename that exists in the current spark context as Temp-View
    max_group_count: int
        After join and group over the string attriutes, only consider builded groups with the maximum amount of max_group_count to calculate the earth movers distance between columns
    sel_attr: None/array
        if None than all columns combination are calculates, 
        else you can provide a list of column-names from the inner table to only select the combinations with that specific columns   

    Returns
    -------
    df: spark.DataFrame
        all results of emd betwenn all column-combinations
    """
    outer = outer
    inner = inner
    # find matching attributes to compare
    join_attributes = list(
        set(string_attributes[inner]) & set(string_attributes[outer]))
    join_condition = "ON (" + " AND ".join(map(lambda join_att: "o.`{join_att}` = i.`{join_att}`".format(join_att=join_att),
                                           join_attributes))
    intersecting_attr = list(
        set(numeric_attributes[inner]) & set(numeric_attributes[outer]))

    # create projection list
    projection_list = " , ".join(
        map(lambda attr: "o.`{attr}` as `{attr}`".format(attr=attr),
            join_attributes)
    ) + " , " + " , ".join(
        map(
            lambda attr: "o.`{attr}` as `o.{attr}` , i.`{attr}` as `i.{attr}`".
            format(attr=attr), intersecting_attr))
    sqlDF = spark.sql("SELECT "+projection_list+" FROM " + outer + " o JOIN " +
                      inner + " i " + join_condition+")")
    # filter out null tupels with null values
    sqlDF = sqlDF.dropna(subset=list(
        map(lambda cur_col: "`{cur_col}`".format(cur_col=cur_col),
            sqlDF.columns)))

    # groupby string attribute and filter out instances which only consider specific times
    sqlDF_instances_to_consider = sqlDF.groupby(join_attributes).agg(
        count(join_attributes[0]).alias("count")).where(col("count") <= max_group_count)
    resSqlDF = sqlDF.join(sqlDF_instances_to_consider,
                          on=join_attributes, how='inner')

    attr_variations = pair_permutations_ordered(intersecting_attr)

    # selsect specific attr_variation with a specific attribute included
    if sel_attr == None:
        sel_attr_variations = attr_variations
    else:
        sel_attr = sel_attr
        sel_attr_variations = list(
            filter(lambda x: x[1] in sel_attr, attr_variations))

    result_list = []
    pbar = tqdm(total=len(sel_attr_variations))
    for index_attr, curr_item in enumerate(sel_attr_variations):
        # print(str(index_attr)+"/"+str(len(sel_attr_variations)))
        first_attr = curr_item[0]
        second_attr = curr_item[1]
        emd = resSqlDF.select(emd_UDF(collect_list("`o.{first_attr}`".format(first_attr=first_attr)), collect_list(
            "`i.{second_attr}`".format(second_attr=second_attr))).alias("EMD")).collect()[0]["EMD"]
        if emd == None:
            print(f"EMD is none {first_attr},{second_attr}")
            continue
        result_list.append(
            [outer, first_attr, inner, second_attr, max_group_count, float(emd)])
        pbar.update(1)
    if len(result_list) == 0:
        return None
    resultDF = spark.createDataFrame(result_list).toDF(
        "OUTER", "OUTER_ATTR", "INNER", "INNER_ATTR", "COUNT", "EMD")
    return resultDF
