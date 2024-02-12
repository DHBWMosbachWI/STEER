import unittest
from os import getcwd
from os.path import split
from helper_functions import print_df_to_html, translate_to_local_file_path, translate_header_file_to_list, variations, translate_datatype_file_to_list
from pyspark.sql.types import DecimalType, IntegerType, StringType

class HelperFunctionsTestCase(unittest.TestCase):
    """Test for HelperFunction.py"""
    
    def test_translate_to_local_file_path(self):
        pwd=getcwd()
        head, tail = split(pwd)
        filename= "test.txt"
        dirname="dir"
        file_path = translate_to_local_file_path(filename)
        self.assertEqual(file_path, f"file:///{head}/{filename}")
        file_path = translate_to_local_file_path(filename,dirname)
        self.assertEqual(file_path, f"file:///{head}/{dirname}/{filename}")

    def test_translate_header_file_to_list(self):
        result = translate_header_file_to_list("data/samples/MLB_1.header.csv")
        self.assertEqual(result,["AB", 'AVG', 'BABIP', 'BB.', 'BB', 'BIP', 'FB.', 'FB', 'GB.', 'GB', 'GIDP', 'HBP', 'HR', 'H', 'ISO', 'K.', 'LD.', 'LD', 'Number of Records', 'OBP', 'PA', 'PU.', 'PU', 'SF', 'SH', 'SLG', 'SOL', 'SOS', 'SO', 'TB', 'X1B', 'X2B', 'X3B', 'batter_name', 'field', 'iBB', 'league', 'parentteam', 'pwRC.', 'stand', 'teamname', 'wOBA', 'wRAA', 'wRC.', 'wRC', 'year', 'Calculation_40532458112880653', 'Calculation_40532458117070874'])

    def test_translate_datatype_file_to_list(self):
        result = translate_datatype_file_to_list("data/samples/MLB_1.datatypes.csv")    
        self.assertListEqual(result,[IntegerType(), DecimalType(4, 3), DecimalType(4, 3), 
                                    IntegerType(), IntegerType(), IntegerType(), DecimalType(4, 1), 
                                    IntegerType(), DecimalType(4, 1), IntegerType(), IntegerType(), 
                                    IntegerType(), IntegerType(), IntegerType(), DecimalType(4, 3), 
                                    IntegerType(), DecimalType(4, 1), IntegerType(), IntegerType(), 
                                    DecimalType(4, 3), IntegerType(), DecimalType(4, 1), IntegerType(),
                                    IntegerType(), IntegerType(), DecimalType(4, 3), IntegerType(), 
                                    IntegerType(), IntegerType(), IntegerType(), IntegerType(), IntegerType(),
                                    IntegerType(), StringType(), StringType(), IntegerType(), StringType(), 
                                    StringType(), IntegerType(), StringType(), StringType(), 
                                    DecimalType(4, 3), DecimalType(3, 1), IntegerType(), 
                                    DecimalType(3, 1), IntegerType(), DecimalType(17, 14), 
                                    DecimalType(17, 14)])


    def test_print_df_to_html(self):
        from pyspark.sql import SparkSession
        from pyspark import SparkConf
        from IPython.core.display import display
        # create a SparkSession
        conf = SparkConf()
        conf.setMaster("local[1]")
        conf.setAppName("test_print_df_to_html")
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        df = spark.createDataFrame( [ (1, 'foo'), (2, 'bar')], ['id', 'txt'] ) 
        result = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>id</th>\n      <th>txt</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>foo</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>bar</td>\n    </tr>\n  </tbody>\n</table>'       
        self.assertEqual(print_df_to_html(df).data,result)

    def test_variations(self):
        input_list = [1,2,3]
        self.assertListEqual(variations(input_list,0),[set()])
        self.assertListEqual(variations(input_list,1),[{1},{2},{3}])
        self.assertListEqual(variations(input_list, 2), [{1,2}, {1,3},{2,3}])
        self.assertListEqual(variations(input_list, 3), [{1,2,3}])
        self.assertListEqual(variations(input_list, 4), [set()])

if __name__ == "__main__":
    unittest.main()