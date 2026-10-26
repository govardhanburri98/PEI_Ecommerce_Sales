# Databricks notebook source
# MAGIC %md
# MAGIC # Q5 SQL_Aggregates
# MAGIC

# COMMAND ----------

# DBTITLE 1,Profit by Year
# MAGIC %sql
# MAGIC SELECT order_year,
# MAGIC        ROUND(SUM(profit), 2) AS profit
# MAGIC FROM ecommerce_sales.default.profit_aggregate
# MAGIC GROUP BY order_year
# MAGIC ORDER BY order_year;

# COMMAND ----------

# DBTITLE 1,Profit by Year + Product Category
# MAGIC %sql
# MAGIC SELECT order_year,
# MAGIC        category,
# MAGIC        ROUND(SUM(profit), 2) AS profit
# MAGIC FROM ecommerce_sales.default.profit_aggregate
# MAGIC GROUP BY order_year, category
# MAGIC ORDER BY order_year, category;

# COMMAND ----------

# DBTITLE 1,Profit by Customer
# MAGIC %sql
# MAGIC SELECT customer_id,
# MAGIC        customer_name,
# MAGIC        ROUND(SUM(profit), 2) AS profit
# MAGIC FROM ecommerce_sales.default.profit_aggregate
# MAGIC GROUP BY customer_id, customer_name
# MAGIC ORDER BY customer_id, customer_name;

# COMMAND ----------

# DBTITLE 1,Profit by Customer + Year
# MAGIC %sql
# MAGIC SELECT customer_id,
# MAGIC        customer_name,
# MAGIC        order_year,
# MAGIC        ROUND(SUM(profit), 2) AS profit
# MAGIC FROM ecommerce_sales.default.profit_aggregate
# MAGIC GROUP BY customer_id, customer_name, order_year
# MAGIC ORDER BY customer_id, customer_name, order_year;