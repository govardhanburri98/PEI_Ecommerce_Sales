# Databricks notebook source
# MAGIC %md
# MAGIC # Q2 enriched customers & products
# MAGIC - Enriched customer and product tables (typed, cleaned, de-duplicated).

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

from pyspark.sql import DataFrame, Window, functions as F

# COMMAND ----------

CUSTOMER_COLUMNS = ["customer_id", "customer_name", "email", "phone", "segment",
                    "country", "city", "state", "postal_code", "region"]
PRODUCT_COLUMNS = ["product_id", "category", "sub_category", "product_name"]

# COMMAND ----------

def clean_name(col):
    """Clean up names: remove special characters, collapse extra spaces, blanks become Nulls."""
    # Drop anything that is not a letter, digit, or space, then collapse runs of spaces.
    cleaned = F.regexp_replace(col, r"[^a-zA-Z0-9 ]", " ")
    cleaned = F.regexp_replace(cleaned, r"\s+", " ")
    cleaned = F.trim(cleaned)
    return blank_to_null(cleaned)

# COMMAND ----------

# DBTITLE 1,Enrich Customer Columns - Clean & De Duplicated
def enrich_customers(raw: DataFrame) -> DataFrame:
    require_columns(raw, ["customer_id", "customer_name", "country"], "raw_customers")
    df = (raw
          .withColumn("customer_id", blank_to_null(F.col("customer_id")))
          .withColumn("customer_name", clean_name(F.col("customer_name")))
          .withColumn("country", blank_to_null(F.col("country"))))
    for c in ("email", "phone", "segment", "city", "state", "postal_code", "region"):
        if c in df.columns:
            df = df.withColumn(c, blank_to_null(F.col(c)))
    df = df.filter(F.col("customer_id").isNotNull()) ## Filtered for not null Cust Id's


    # One row per customer_id; deterministic pick so reruns give identical output.
    w = Window.partitionBy("customer_id").orderBy(
        F.col("customer_name").asc_nulls_last(), F.col("country").asc_nulls_last())
    df = df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")
    return select_existing(df, CUSTOMER_COLUMNS)

# COMMAND ----------

# DBTITLE 1,Enrich Products - Cleaned and De Duplicated
def enrich_products(raw: DataFrame) -> DataFrame:
    require_columns(raw, ["product_id", "category", "sub_category"], "raw_products")
    df = (raw
          .withColumn("product_id", blank_to_null(F.col("product_id")))
          .withColumn("category", blank_to_null(F.col("category")))
          .withColumn("sub_category", blank_to_null(F.col("sub_category"))))
    if "product_name" in df.columns:
        df = df.withColumn("product_name", blank_to_null(F.col("product_name")))
    df = df.filter(F.col("product_id").isNotNull())
    # The source repeats a product_id per state (with different prices). For category
    # enrichment we need exactly one row per product_id so the join cannot fan out.
    order_cols = [F.col(c).asc_nulls_last() for c in
                  ("product_name", "state", "price_per_product") if c in df.columns]
    w = Window.partitionBy("product_id").orderBy(*(order_cols or [F.col("product_id")]))
    df = df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
    return select_existing(df, PRODUCT_COLUMNS)

# COMMAND ----------

# DBTITLE 1,Write into delta tables
# Config values (from ../config/1_config) - not loaded by %run ./utils
CATALOG = "ecommerce_sales"
SCHEMA = "default"
TABLE_FORMAT = "delta"

# write_table(df, table_name, fmt="delta", mode="overwrite")
# qualified_name(catalog, schema, name) -> "catalog.schema.name"
write_table(enrich_customers(spark.table(qualified_name(CATALOG, SCHEMA, "raw_customers"))),
            qualified_name(CATALOG, SCHEMA, "enriched_customers"), fmt=TABLE_FORMAT)
write_table(enrich_products(spark.table(qualified_name(CATALOG, SCHEMA, "raw_products"))),
            qualified_name(CATALOG, SCHEMA, "enriched_products"), fmt=TABLE_FORMAT)

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from ecommerce_sales.default.enriched_customers  

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from ecommerce_sales.default.enriched_products  