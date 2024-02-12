"""Helper functions for the jupyter notebooks"""

REMOTE_HOST="192.168.11.3"

from os import listdir
from os.path import abspath, isfile, join
from socket import gethostname, gethostbyname
from typing import List
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DecimalType, DoubleType, IntegerType, StringType
from pyspark.sql.functions import col, isnan, when, count


def print_df_to_html(sparkDF: DataFrame): 
    from IPython.display import HTML
    newdf = sparkDF.toPandas()
    return HTML(newdf.to_html())

def translate_to_local_file_path(filename,directory=''):  
    if (gethostbyname(gethostname())) == REMOTE_HOST :
        if directory:
            filepath= "../{directory}/{filename}".format(directory=directory, filename=filename)
        else:
            filepath= "../{filename}".format(filename=filename)
    else:
        if directory:
            filepath= "../{directory}/{filename}".format(directory=directory, filename=filename)
        else:
            filepath= "../{filename}".format(filename=filename)
    print(abspath(filepath))   
    return "file:///path".format(path=abspath(filepath))

def translate_to_file_string(filepath):
    return "file:///path".format(path=abspath(filepath))


def translate_header_file_to_list(filepath):
   """Reads the column names from the given file and converts it to a list."""
   with open(filepath, "r") as f:
       return f.readline().rstrip().split("|")  

def translate_datatype_file_to_list(filepath) -> list:
    """reads the data types from the file, consolidates them to (string, double, int, date). 
    Retruns a list of the consolidated datatypes."""
    result = []
    with open(filepath, "r") as f:
        raw_datatype_list=  f.readline().rstrip().split("|")
        for curr_rt in raw_datatype_list:
            if (curr_rt.startswith("smallint")):
                result.append(IntegerType())
            elif (curr_rt.startswith("decimal")):
                decimal_params = curr_rt[curr_rt.find("("):curr_rt.find(")")+1]
                num_digits = int(decimal_params[decimal_params.find("(")+1:decimal_params.find(",")])
                num_scale = int(decimal_params[decimal_params.find(",")+1:decimal_params.find(")")])
                result.append(DecimalType(num_digits, num_scale))
            elif (curr_rt.startswith("varchar")):
                result.append(StringType())
            else:
                result.append(StringType())
    return result        

def concat_files(output_file,result_dir):
    """Concats the files in the output dir do one file"""
    file_list = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
    file_list = list (map(lambda file : result_dir+"/"+file , filter(lambda x : (not x.startswith(".")) and (not x.startswith("_")), file_list)))
    print(file_list)
    with open(output_file, "w") as outfile:
        for fname in file_list:
            with open(fname) as infile:
                outfile.write(infile.read())


def variations(items :list, k :int) -> list:
    if k==0 or k>len(items):
        return [set()]
    else:
        new_result = []
        for item in items :
            for curr_set in variations(items, k-1):
                if item not in curr_set:
                    curr_set.add(item)
                    if curr_set not in new_result:
                        new_result.append(curr_set)
        return new_result

def pair_permutations_ordered(items: list) -> list:
    pair_permutations = []
    for curr_attr_set in variations(items,2):
        first_attr = curr_attr_set.pop()
        second_attr = curr_attr_set.pop()  
        pair_permutations.append([first_attr,second_attr])
        pair_permutations.append([second_attr,first_attr])
    for curr_attr in items:
        pair_permutations.append([curr_attr, curr_attr])
    return pair_permutations

def cast_datatypes(datatype_file, input_df:DataFrame) -> DataFrame:
    datatype_list  = translate_datatype_file_to_list(datatype_file)
    df = input_df.alias('tmp_df')
    if len(df.columns) == len(datatype_list):
        for i in range(len(datatype_list)):
            if datatype_list[i] != type(df.schema[i].dataType) :
                #print (f"{df.columns[i]} {datatype_list[i]} {df.schema[i].dataType}") 
                df = df.withColumn(df.columns[i],col("`{column}`".format(column=df.columns[i])).cast(datatype_list[i]))                
        return df
    else:
        return df

def check_attribute_completeness(all_columns:list, string_attributes:list, numeric_attributes:list):
    for curr_col in all_columns:
        if curr_col not in string_attributes and curr_col not in numeric_attributes :
            print ("{curr_col} is not numeric or string".format(curr_col=curr_col))

def compare_schemas(df1:DataFrame, df2:DataFrame):
    if len(df1.dtypes) != len(df2.dtypes):
        print ("Schemas have differen sizes!!!")
    else :
        for i in range(len(df1.dtypes)):
            if df1.dtypes[i][0] != df2.dtypes[i][0] or df1.dtypes[i][1] != df2.dtypes[i][1] :
                print ("Columns differ {dtype1} {dtype2}".format(dtype1=df1.dtypes[i], dtype2=df2.dtypes[i]))