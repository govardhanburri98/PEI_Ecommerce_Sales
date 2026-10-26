# Databricks notebook source
# MAGIC %md
# MAGIC # Q4 profit & aggregate
# MAGIC Q4 - Aggregate table: profit by Year, Product Category, Sub Category and Customer.

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F

# COMMAND ----------

GROUP_COLUMNS = ["order_year", "category", "sub_category", "customer_id", "customer_name"] ## include only required columns
UNKNOWN = "Unknown"

# COMMAND ----------

def build_profit_aggregate(enriched_orders: DataFrame) -> DataFrame:
    """Orders whose product/customer could not be matched are reported under 'Unknown'
    (instead of NULL groups) so stakeholders can see - and chase - the unmatched profit."""
    require_columns(enriched_orders, GROUP_COLUMNS + ["profit"], "enriched_orders")
    labelled = enriched_orders
    for c in ("category", "sub_category", "customer_name"):
        labelled = labelled.withColumn(c, F.coalesce(F.col(c), F.lit(UNKNOWN)))
    return (labelled
            .groupBy(*GROUP_COLUMNS)
            .agg(F.round(F.sum("profit"), 2).alias("profit"))
            .orderBy(*GROUP_COLUMNS))

# COMMAND ----------

# DBTITLE 1,Write to a table
CATALOG = "ecommerce_sales"
SCHEMA = "default"
TABLE_FORMAT = "delta"


write_table(build_profit_aggregate(spark.table(qualified_name(CATALOG, SCHEMA, "enriched_orders"))), qualified_name(CATALOG, SCHEMA, "profit_aggregate"), fmt = TABLE_FORMAT)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ecommerce_sales.default.profit_aggregate  