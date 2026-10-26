# Databricks notebook source
# MAGIC %md
# MAGIC # Q3 enriched_orders
# MAGIC Q3 - Enriched orders: order info + customer name/country + product category/sub-category.

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F

# COMMAND ----------

# DBTITLE 1,Consolidate the list of columns
DEFAULT_DATE_FORMAT = "d/M/yyyy"
OUTPUT_COLUMNS = ["order_id", "row_id", "order_date", "ship_date", "ship_mode", "order_year",
                  "customer_id", "customer_name", "country",
                  "product_id", "category", "sub_category",
                  "quantity", "price", "discount", "profit"]

# COMMAND ----------

def clean_orders(raw_orders: DataFrame, date_format: str = DEFAULT_DATE_FORMAT) -> DataFrame:
    """Type the orders, round profit to 2dp, derive order_year, drop unusable rows."""
    require_columns(raw_orders, ["order_id", "order_date", "customer_id", "product_id", "profit"],
                    "raw_orders")
    df = raw_orders
    if "_corrupt_record" in df.columns:          # malformed JSON rows from PERMISSIVE mode are dropped
        df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")
    source_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.select(source_cols).dropDuplicates()  # exact duplicate rows only

    for c in ("order_id", "customer_id", "product_id"):  ## required set to join with subsequent tables 
        df = df.withColumn(c, blank_to_null(F.col(c)))
    df = df.withColumn("order_date", parse_date("order_date", date_format))
    if "ship_date" in df.columns:
        df = df.withColumn("ship_date", parse_date("ship_date", date_format))
    if "quantity" in df.columns:
        df = df.withColumn("quantity", safe_cast("quantity", "int"))
    for c in ("price", "discount"):
        if c in df.columns:
            df = df.withColumn(c, safe_cast(c, "double"))
    # Requirement 3.1.1: profit rounded to 2 decimal places. DECIMAL(18,2) (not DOUBLE) so
   
    df = df.withColumn("profit", F.round(safe_cast("profit", "double"), 2).cast("decimal(36,2)"))
    df = df.withColumn("order_year", F.year("order_date"))
    return df.filter(F.col("order_id").isNotNull())

# COMMAND ----------

def enrich_orders(raw_orders: DataFrame, customers: DataFrame, products: DataFrame,
                  date_format: str = DEFAULT_DATE_FORMAT) -> DataFrame:
    """Left-join so no order (and therefore no profit) is lost when a lookup is missing.

    `customers` / `products` must be the Q2 outputs (one row per key) so the joins
    can never multiply order rows.
    """
    require_columns(customers, ["customer_id", "customer_name", "country"], "enriched_customers")
    require_columns(products, ["product_id", "category", "sub_category"], "enriched_products")

    
    orders = clean_orders(raw_orders, date_format)
    # Dimension tables are tiny vs. orders: broadcast them so the big side is never shuffled.
    cust = F.broadcast(customers.select("customer_id", "customer_name", "country"))
    prod = F.broadcast(products.select("product_id", "category", "sub_category"))
    joined = orders.join(cust, "customer_id", "left").join(prod, "product_id", "left")
    return select_existing(joined, OUTPUT_COLUMNS)

# COMMAND ----------

# DBTITLE 1,Write the ouput into a table
CATALOG = "ecommerce_sales"
SCHEMA = "default"
TABLE_FORMAT = "delta"


orders = enrich_orders(
    raw_orders=spark.table(qualified_name(CATALOG, SCHEMA, "raw_orders")),
    customers=spark.table(qualified_name(CATALOG, SCHEMA, "enriched_customers")),
    products=spark.table(qualified_name(CATALOG, SCHEMA, "enriched_products")),
    date_format=DEFAULT_DATE_FORMAT,
)

##. Write into a enriched orders table 
write_table(orders, qualified_name(CATALOG, SCHEMA, "enriched_orders"), fmt=TABLE_FORMAT)

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC select * from ecommerce_sales.default.enriched_orders  

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select count(*) from ecommerce_sales.default.enriched_orders  
# MAGIC
# MAGIC --- matches with the source count