'''
For sample data:
time spark-submit project_code.py 2> /dev/null

For big data:
time spark-submit --master yarn \
                    --deploy-mode cluster \
                    --executor-cores 2 \
                    --driver-memory 7G \
                    --executor-memory 7G \
                    --conf spark.dynamicAllocation.maxExecutors=50 \
                    preprocessing.py

'''
import re
import ast
import json

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, col, udf, size
from pyspark.sql.types import StructType, StructField, IntegerType

sc = SparkContext(appName="Notebook_Project")
sc.setLogLevel("ERROR")

spark = SparkSession.builder.getOrCreate()

# Big Data (~45 GB)
# df_python = spark.read.json("/user/s2435462/notebook_data/*/*.json")

# Subset Data (~20 GB)
df_python = spark.read.json("/user/s2435462/notebook_data/[0-4]/*.json")

# Drop unwanted columns
df = df.drop("nbformat", "nbformat_minor", "worksheets")

# Python only notebooks
df_python = df_python.filter(df_python.metadata.kernelspec.language == "python")

# Remove notebooks with null cells column
df_python = df_python.filter(col("cells").isNotNull())

# Add Python version used in the notebook
df_python = df_python.withColumn("py_version", df.metadata.kernelspec.display_name)

# Drop the metadata column since it's no longer needed
df_python = df_python.drop(col("metadata"))

# Add row numbers to the notebooks which would act as IDs
w = Window.orderBy(lit(1))
df_python = df_python.withColumn("id", row_number().over(w))

# ========================================================================================================

df_python.drop("cells")\
            .coalesce(1) \
            .write.csv("notebook_results_id/py_version_id", header = True)

df_python.drop("py_version")

# ========================================================================================================

# Number of cells in a notebook
df_python = df_python.withColumn("cell_len", size(col("cells"))) 
df_python = df_python.filter(df_python.cell_len > 0) # Filter notebooks with atleast 1 cell
df_python.select("id", "cell_len") \
            .coalesce(1) \
            .write.csv("notebook_results_id/length", header = True)

df_python = df_python.drop("cell_len")
# ========================================================================================================

# Filter Code cell count
def code_count(cells):
	cd = len([1 for x in cells if x["cell_type"] == "code"])
	return cd

code_count_udf = udf(lambda x: code_count(x))

df_python = df_python.withColumn("cd_len", code_count_udf(col("cells")))
df_python = df_python.filter(df_python.cd_len > 0) # Filter notebooks with atleast 1 code cell
df_python = df_python.drop("cd_len")

# ========================================================================================================

# Markdown count vs Code count

def mark_code_count(cells):
	md = len([1 for x in cells if x["cell_type"] == "markdown"])
	cd = len([1 for x in cells if x["cell_type"] == "code"])
	return (md, cd)

schema = StructType([
						StructField("md", IntegerType(), False),
						StructField("cd", IntegerType(), False)
					])

mark_code_count_udf = udf(lambda x: mark_code_count(x), schema)

df_python.withColumn("md_cd", mark_code_count_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)
            .write.json("notebook_results_id/md_cd")

# ========================================================================================================

# Linear execution or not

def linear(cells):
	a = 0
	for cell in cells:
		if cell["cell_type"] == "code":
			if cell["execution_count"]:
				if cell["execution_count"] > a:
					a = cell["execution_count"]
				else:
					return False
	return True

linear_udf = udf(lambda x: linear(x))

df_python.withColumn("linear_ec", linear_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/linear_ec", header = True)

# ========================================================================================================

# Check if JSON decodable string or not

def check_JSON(cells):
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				try:
					lines = json.loads(cell["source"])
					return "JSON"
				except:
					return "null"
			else:
				continue
	return "string"


check_JSON_udf = udf(lambda x: check_JSON(x))

df_python = df_python.withColumn("check_JSON", check_JSON_udf(col("cells")))

df_python = df_python.filter(df_python.check_JSON != "null")

df_python = df_python.drop("check_JSON")

# ========================================================================================================

# Select the ID and Cells column and save it to a folder as preprocessed data
# for further analysis
df_python = df_python.select("id", "cells")

df_python.coalesce(50) \
          .write.json("notebook_cells_id")

# ========================================================================================================