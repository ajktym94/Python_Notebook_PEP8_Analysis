'''
For sample data:
time spark-submit main.py 2> /dev/null

For big data:
time spark-submit --master yarn \
                    --deploy-mode cluster \
                    --executor-cores 2 \
                    --driver-memory 7G \
                    --executor-memory 7G \
                    --conf spark.dynamicAllocation.maxExecutors=50 \
                    main.py

Execution time (the UDFs were executed one by one):

real	5m24.976s
user	0m11.342s
sys		0m2.463s

'''
import re
import ast
import json

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, size
from pyspark.sql.types import StructType, StructField, IntegerType


sc = SparkContext(appName="Notebook_Project")
sc.setLogLevel("ERROR")

spark = SparkSession.builder.getOrCreate()

# Sample Data (~40 MB)
# df_python = spark.read.json("/user/s2435462/notebook_data_sample/*/*.json")

# Subset Big Data (~20 GB) (Preprocessed)
df_python = spark.read.json("/user/s2435462/notebook_cells_id/*.json")

# ========================================================================================================

# Check Max Length

def max_length(cells):
	max_len = 0
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				if len(line) > max_len:
					max_len = len(line)
	return str(max_len)

max_length_udf = udf(lambda x: max_length(x))

df_python = df_python.withColumn("max_length", max_length_udf(col("cells")))

df_python.drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/max_length", header = True)

# ========================================================================================================

# Assign labels to max length

def label(length):
    if int(length) > 98:
        return "Wrong"
    elif int(length) > 76 and int(length) <= 99:
        return "Allowed"
    else:
        return "Good"  

label_udf = udf(lambda x: label(x))

df_python = df_python.withColumn("label", label_udf(col("max_length")))

df_python.drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/max_length_label", header = True)

df_python = df_python.withColumn("count", lit(1))

df_python.drop("cells")\
         .groupBy("label")\
         .sum("count")\
         .coalesce(1)\
         .write.csv("notebook_results_id/max_length_grouped", header = True)

df_python = df_python.drop("max_len", "label", "count")
# ========================================================================================================

# Check mixed quotes

def check_quotes(cells):
	q = [False, False]

	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				if not q[0] and "'" in line:
					q[0] = True
				if not q[1] and '"' in line:
					q[1] = True
		if q[0] and q[1]:
			return True
	return False

check_quotes_udf = udf(lambda x: check_quotes(x))

df_python.withColumn("quote_mismatch", check_quotes_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/quote_mismatch", header = True)

# ========================================================================================================

# Wildcard imports

WCIMPORT = re.compile(r"^from\s[a-z]*\simport\s\*")

def wc_import(cells):
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				if WCIMPORT.match(line):
					return True
	return False

wc_import_udf = udf(lambda x: wc_import(x))

df_python.withColumn("wc_import", wc_import_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/wc_import", header = True)

# ========================================================================================================

# Trailing whitespaces

def trailing(cells):
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				line = line.rstrip("\n")
				line = line.rstrip("\r")
				line = line.rstrip("\x0c")
				stripped = line.rstrip(" \t\v")
				if line != stripped:
					return True
	return False

trailing_udf = udf(lambda x: trailing(x))

df_python.withColumn("trailing_ws", trailing_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/trailing", header = True)

# ========================================================================================================

# Import on separate lines

def imports_same(cells):
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				if line.startswith("import "):
					comma = line.find(",")
					if comma > -1:
						return True
	return False

imports_same_udf = udf(lambda x: imports_same(x))

df_python.withColumn("imports_same", imports_same_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/imports_same", header = True)

# ========================================================================================================

# Non English characters
NON_EN = re.compile(r"[^\x00-\x7F]+")

def non_en(cells):
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
					lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				if NON_EN.match(line):
					return True
	return False

non_en_udf = udf(lambda x: non_en(x))

df_python.withColumn("non_en", non_en_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/non_en", header = True)

# ========================================================================================================

# Extra Whitespace around operator

OPERATOR_REGEX = re.compile(r"(?:[^,\s])(\s*)(?:[-+*/|!<=>%&^]+|:=)(\s*)")

def whitespace_around_operator(cells):
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
			    for match in OPERATOR_REGEX.finditer(line):
			        before, after = match.groups()
			        if "\t" in before:
			            return True
			        elif len(before) > 1:
			            return True
			        if "\t" in after:
			            return True
			        elif len(after) > 1:
			            return True
	return False

ex_ws_op_udf = udf(lambda x: whitespace_around_operator(x))

df_python.withColumn("ex_ws_op", ex_ws_op_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/ex_ws_op", header = True)

# ========================================================================================================

# Mixed Indentation
INDENT_REGEX = re.compile(r"([ \t]*)")

def mixed_indent(cells):
	indent_char = None
	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				if not indent_char and line.startswith(" ") or line.startswith("\t"):
					indent_char = line[0]
				if indent_char:
					indent = INDENT_REGEX.match(line).group(1)
					for offset, char in enumerate(indent):
						if char != indent_char:
							return True
	return False

mixed_indent_udf = udf(lambda x: mixed_indent(x))

df_python.withColumn("mixed_indent", mixed_indent_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/mixed_indent", header = True)

# ========================================================================================================

# Find Error count

def error_count(cells):
	ec = 0
	for cell in cells:
		if cell["cell_type"] == "code":
			if len(cell["outputs"]) > 0:
				for op in cell["outputs"]:
					if op["output_type"] == "error":
						ec = ec + 1
	return ec

error_count_udf = udf(lambda x: error_count(x))

df_python.withColumn("error_count", error_count_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/error_count", header = True)


# ========================================================================================================

# Check import in the beginning

def import_top_error(cells):
	n_code_cells = len([1 for cell in cells if cell["cell_type"] == "code"])
	latest_import = 0
	curr_line = 0
	docstring_started = False
	first_code_cell_found = False
	first_code_cell = 0

	for cell in cells:
		if cell["cell_type"] == "code" and cell["source"]:
			if cell["source"][0] == "[":
				lines = json.loads(cell["source"])
			else:
				lines = cell["source"].split("\n")
			for line in lines:
				curr_line = curr_line + 1
				if (line.startswith('"""') or line.startswith("'''")) and (line.endswith('"""') or line.endswith("'''")):
					continue
				if (line.startswith('"""') or line.startswith("'''")) and not docstring_started:
					docstring_started = True
					continue
				if (line.startswith('"""') or line.startswith("'''")) and docstring_started:
					docstring_started = False
					continue
				if not docstring_started:
					if "import" in line and (line.find("import") < line.find("#") or line.find("#") == -1):
						latest_import = curr_line
					elif not first_code_cell_found and not line.startswith("#"):
						first_code_cell_found = True
						first_code_cell = curr_line


	if latest_import > first_code_cell:
		return True
	else:
		return False

import_top_error_udf = udf(lambda x: import_top_error(x))
df_python.withColumn("import_top_error", import_top_error_udf(col("cells")))\
            .drop("cells")\
            .coalesce(1)\
            .write.csv("notebook_results_id/import_top_error", header = True)

# ========================================================================================================