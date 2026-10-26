# Databricks notebook source
# MAGIC %md
# MAGIC # Q1_raw_tables
# MAGIC Q1 - Raw tables for each source dataset.
# MAGIC
# MAGIC Raw
# MAGIC - case the column names in a way so Delta accepts that
# MAGIC - keep every value as STRING so nothing is silently lost or can come out as null.

# COMMAND ----------

# DBTITLE 1,run utils
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Install openpyxl for Excel reading
# MAGIC %pip install openpyxl

# COMMAND ----------

# DBTITLE 1,import lib, funcs
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType

# COMMAND ----------

# DBTITLE 1,Read Products Table
# Config paths
VOLUME_PATH = "/Volumes/ecommerce_sales/default/sales_extract/raw/"
PRODUCTS_PATH = VOLUME_PATH + "Products.csv"
ORDERS_PATH = VOLUME_PATH + "Orders.json"
CUSTOMERS_PATH = VOLUME_PATH + "Customer.xlsx"


products_sample = (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(PRODUCTS_PATH))
print("=== Products.csv ===")
print(f"Columns ({len(products_sample.columns)}): {products_sample.columns}")
display(products_sample)

## Found Product Names over riding into state columns 

# COMMAND ----------

# DBTITLE 1,Sample Read all 3 files to view the structure
#Products.csv (CSV, inferSchema) using MultiLine
products_sample = (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(PRODUCTS_PATH))
print("=== Products.csv ===")
print(f"Columns ({len(products_sample.columns)}): {products_sample.columns}")
display(products_sample)

#Customer.xlsx (Excel via pandas, types inferred)
customers_pdf = pd.read_excel(CUSTOMERS_PATH, sheet_name=0, engine="openpyxl")
print("=== Customer.xlsx ===")
print(f"Columns ({len(customers_pdf.columns)}): {list(customers_pdf.columns)}")
print(f"Inferred dtypes:\n{customers_pdf.dtypes}")


    # Cast all columns to string so mixed types (e.g. phone) convert cleanly, then show in Spark
customers_pdf = customers_pdf.astype(str)
customers_sample = spark.createDataFrame(customers_pdf)
display(customers_sample)

# COMMAND ----------

# DBTITLE 1,Check for unique keys from above
products_sample.createOrReplaceTempView("products_v")
customers_sample.createOrReplaceTempView("customers_v")

# 1) Row count vs distinct count (equal => unique)
print("=== Uniqueness summary (total_rows == distinct_ids means unique) ===")
display(spark.sql("""
        SELECT 'Product ID' AS key_column,
               COUNT(*) AS total_rows,
               COUNT(DISTINCT `Product ID`) AS distinct_ids
        FROM products_v
        UNION ALL
        SELECT 'Customer ID',
               COUNT(*),
               COUNT(DISTINCT `Customer ID`)
        FROM customers_v
"""))

    # 2) List any duplicated keys (empty result means the key is unique)
print("=== Duplicate Product IDs (empty = unique) ===")
display(spark.sql("""
        SELECT `Product ID`, COUNT(*) AS cnt
        FROM products_v
        GROUP BY `Product ID`
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC, `Product ID`
"""))

print("=== Duplicate Customer IDs (empty = unique) ===")
display(spark.sql("""
        SELECT `Customer ID`, COUNT(*) AS cnt
        FROM customers_v
        GROUP BY `Customer ID`
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC, `Customer ID`
"""))

### Product ID are duplicated :: subsequent check on the State column
## If (Product ID, State) is unique, the key is really composite (Product ID + State)
print("=== Uniqueness of (Product ID, State) composite key ===")
display(spark.sql("""
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT (`Product ID` || '|' || `State`)) AS distinct_product_state_pairs,
               COUNT(DISTINCT `Product ID`) AS distinct_product_ids
        FROM products_v
"""))

print("=== Duplicate (Product ID, State) pairs (empty = composite key is unique) ===")
display(spark.sql("""
        SELECT `Product ID`, `State`, COUNT(*) AS cnt
        FROM products_v
        GROUP BY `Product ID`, `State`
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC, `Product ID`, `State`
"""))

# COMMAND ----------

# DBTITLE 1,Check Raw Orders File

orders_sample = (spark.read
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .option("multiLine", True)
        .json(ORDERS_PATH))
print("=== Orders.json ===")
print(f"Columns ({len(orders_sample.columns)}): {orders_sample.columns}")
display(orders_sample.limit(5))

# COMMAND ----------

# DBTITLE 1,read colums in order and retain in string type
# Explicit schemas (all STRING): skips the extra schema-inference pass over the data,

RAW_PRODUCT_COLUMNS = ["Product ID", "Category", "Sub-Category", "Product Name", "State", "Price per product"]
RAW_ORDER_COLUMNS = ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
                 "Product ID", "Quantity", "Price", "Discount", "Profit"]
PRODUCTS_SCHEMA = StructType([StructField(c, StringType(), True) for c in RAW_PRODUCT_COLUMNS])
ORDERS_SCHEMA = StructType([StructField(c, StringType(), True) for c in RAW_ORDER_COLUMNS]
                           + [StructField("_corrupt_record", StringType(), True)])

# COMMAND ----------

# DBTITLE 1,Raw Layer : default to string
def _to_raw(df: DataFrame, source: str) -> DataFrame:
    df = standardise_columns(df)
    df = df.select([F.col(c).cast("string").alias(c) for c in df.columns])
    return df.withColumn("_source", F.lit(source)).withColumn("_ingested_at", F.current_timestamp())

# COMMAND ----------

# DBTITLE 1,Read Products
def read_products(spark: SparkSession, path: str) -> DataFrame:

    df = (spark.read.schema(PRODUCTS_SCHEMA).option("header", True)
          .option("enforceSchema", False)        # header must match the schema names
          .option("multiLine", True).option("escape", '"').csv(path))
    ##display(df.limit(5))
    return _to_raw(df, path)

# COMMAND ----------

# DBTITLE 1,safe pass the json with dual formats
def read_orders(spark: SparkSession, path: str) -> DataFrame:
    """Orders.json -> raw DataFrame."""
    first = spark.read.text(path).limit(1).collect()
    is_array = bool(first) and first[0][0].lstrip().startswith("[")
    df = (spark.read.schema(ORDERS_SCHEMA).option("multiLine", is_array)
          .option("mode", "PERMISSIVE").json(path))
    ##display(df.limit(5))
    return _to_raw(df, path)

# COMMAND ----------

def read_customers(spark: SparkSession, path: str, sheet_name=0) -> DataFrame:
    """Customer.xlsx -> raw DataFrame.

    Excel is read on the driver with pandas/openpyxl (fine for customer-sized data;
    on Databricks use a /Volumes/... or /dbfs/... path).
    """
    pdf = pd.read_excel(path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    pdf = pdf.astype(object).where(pd.notnull(pdf), None)
    schema = StructType([StructField(str(c), StringType(), True) for c in pdf.columns])
    df = spark.createDataFrame(pdf.values.tolist(), schema)
    ##display(df.limit(5))
    return _to_raw(df, path)

# COMMAND ----------

# DBTITLE 1,Write the respective dataframes into tables

CATALOG = "ecommerce_sales"
SCHEMA = "default"
TABLE_FORMAT = "delta"



# write_table(df, table_name, fmt="delta", mode="overwrite")
# qualified_name(catalog, schema, name) -> "catalog.schema.name"

write_table(read_orders(spark, ORDERS_PATH), qualified_name(CATALOG, SCHEMA, "raw_orders"), fmt=TABLE_FORMAT)
write_table(read_customers(spark, CUSTOMERS_PATH), qualified_name(CATALOG, SCHEMA, "raw_customers"), fmt=TABLE_FORMAT)
write_table(read_products(spark, PRODUCTS_PATH), qualified_name(CATALOG, SCHEMA, "raw_products"), fmt=TABLE_FORMAT)

# COMMAND ----------

# DBTITLE 1,check the tables
# MAGIC %sql
# MAGIC select * from ecommerce_sales.default.raw_customers   

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ecommerce_sales.default.raw_orders   

# COMMAND ----------

# DBTITLE 1,Count Checks - Orders
# MAGIC %sql
# MAGIC select count(*) from ecommerce_sales.default.raw_orders 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ecommerce_sales.default.raw_products   