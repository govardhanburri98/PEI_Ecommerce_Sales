# Databricks notebook source
### Utils added 
### Helper funcs column renaming, column hygeine like required columns, cast to date, write to a table functions listed 

# COMMAND ----------

import re
from typing import Iterable
from pyspark.sql import Column, DataFrame, functions as F

# COMMAND ----------

def to_snake_case(name: str) -> str:
    """'Sub-Category' -> 'sub_category', 'Price per product' -> 'price_per_product'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()

# COMMAND ----------

def standardise_columns(df: DataFrame) -> DataFrame:
    """Rename every column to snake_case  as Delta rejects spaces/special characters."""
    # columns starting with "_" are technical (e.g. _corrupt_record) and keep their name
    new_names = [c if c.startswith("_") else to_snake_case(c) for c in df.columns]
    if len(set(new_names)) != len(new_names):
        raise ValueError(f"Column names collide after standardisation: {new_names}")
    return df.toDF(*new_names)

# COMMAND ----------

def require_columns(df: DataFrame, required: Iterable[str], df_name: str) -> None:
    """Fail fast, with a clear message, if the schema is not what we expect."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")

# COMMAND ----------

def safe_cast(col_name: str, spark_type: str) -> Column:
    """Cast that yields NULL (instead of raising, even under ANSI mode) on bad input."""
    return F.expr(f"try_cast(`{col_name}` AS {spark_type})")

# COMMAND ----------

def parse_date(col_name: str, fmt: str) -> Column:
    """Parse a string to DATE using `fmt`; unparseable values become NULL."""
    return F.try_to_timestamp(F.col(col_name), F.lit(fmt)).cast("date")

# COMMAND ----------

def blank_to_null(col: Column) -> Column:
    """Trim a string column and turn empty strings into NULL."""
    trimmed = F.trim(col)
    return F.when(trimmed == "", F.lit(None)).otherwise(trimmed)

# COMMAND ----------

def select_existing(df: DataFrame, wanted: Iterable[str]) -> DataFrame:
    """Select the wanted columns that exist, preserving the requested order."""
    return df.select(*[c for c in wanted if c in df.columns])

# COMMAND ----------

def write_table(df: DataFrame, table_name: str, fmt: str = "delta", mode: str = "overwrite") -> None:
    """Persist as a managed table. `fmt='parquet'` is used in local tests."""
    df.write.format(fmt).mode(mode).option("overwriteSchema", "true" if fmt == "delta" else "false") \
        .saveAsTable(table_name)


# COMMAND ----------

def qualified_name(catalog, schema, name):
    """'ecommerce_sales.default.raw_orders' (catalog optional for local runs)."""
    if catalog:
        # row[0] because the column is named 'catalog'/'namespace' locally but
        # 'catalog'/'databaseName' on Databricks; catalogExists/schemaExists are
        # not available on Spark Connect.
        catalogs = [row[0] for row in spark.sql("SHOW CATALOGS").collect()]
        if catalog not in catalogs:
            raise ValueError(f"Catalog '{catalog}' does not exist. Available: {catalogs}")
        schemas = [row[0] for row in spark.sql(f"SHOW SCHEMAS IN {catalog}").collect()]
        if schema not in schemas:
            raise ValueError(f"Schema '{schema}' does not exist in catalog '{catalog}'. Available: {schemas}")
    return f"{catalog}.{schema}.{name}" if catalog else f"{schema}.{name}"