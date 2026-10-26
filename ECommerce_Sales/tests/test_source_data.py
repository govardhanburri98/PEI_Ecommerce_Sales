import json
import os
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pyspark.sql import functions as F

from helpers import load_notebook, rows, sql_statements

# these tests use the real files from the assessment; put them in ./data or set SOURCE_DATA_DIR
DATA = Path(os.environ.get("SOURCE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
pytestmark = pytest.mark.skipif(
    not all((DATA / f).exists() for f in ("Orders.json", "Customer.xlsx", "Products.csv")),
    reason="source files not found (set SOURCE_DATA_DIR)")


@pytest.fixture(scope="module")
def source():
    # independent pandas view of the files, used as the expected values
    orders = pd.DataFrame(json.load(open(DATA / "Orders.json")))
    orders["year"] = pd.to_datetime(orders["Order Date"], format="%d/%m/%Y").dt.year
    orders["profit"] = orders["Profit"].round(2)
    return SimpleNamespace(orders=orders,
                           customers=pd.read_excel(DATA / "Customer.xlsx", dtype=str),
                           products=pd.read_csv(DATA / "Products.csv", dtype=str))


@pytest.fixture(scope="module")
def built(spark):
    # run the real notebook functions end to end on the real files (in memory, no tables written)
    q1, q2, q3, q4 = (load_notebook(n, spark=spark) for n in
                      ("Q1_raw_tables", "Q2_enriched_customers_products", "Q3_enriched_orders",
                       "Q4_profit_aggregate"))
    raw_o = q1.read_orders(spark, str(DATA / "Orders.json"))
    raw_c = q1.read_customers(spark, str(DATA / "Customer.xlsx"))
    raw_p = q1.read_products(spark, str(DATA / "Products.csv"))
    cust, prod = q2.enrich_customers(raw_c), q2.enrich_products(raw_p)
    orders = q3.enrich_orders(raw_o, cust, prod).cache()
    agg = q4.build_profit_aggregate(orders).cache()
    return SimpleNamespace(raw_o=raw_o, raw_c=raw_c, raw_p=raw_p, cust=cust, prod=prod, orders=orders, agg=agg)


# the raw tables keep every row of the three source files
def test_raw_row_counts(built, source):
    assert built.raw_o.count() == len(source.orders)
    assert built.raw_c.count() == len(source.customers)
    assert built.raw_p.count() == len(source.products)


# one enriched row per customer id in the source
def test_enriched_customers_unique(built, source):
    assert built.cust.count() == source.customers["Customer ID"].nunique()
    assert built.cust.select("customer_id").distinct().count() == built.cust.count()


# one enriched row per product id, even though the file repeats ids by state
def test_enriched_products_unique(built, source):
    assert built.prod.count() == source.products["Product ID"].nunique()


# no order is lost or duplicated by the joins
def test_enriched_orders_keep_every_order(built, source):
    assert built.orders.count() == len(source.orders)


# every order date is readable and the years match the source
def test_order_dates_all_parsed(built, source):
    assert built.orders.filter("order_year is null").count() == 0
    years = {r["order_year"] for r in rows(built.orders.select("order_year").distinct())}
    assert years == set(source.orders["year"])


# profit is rounded to 2 decimals on every row
def test_profit_has_two_decimals(built):
    assert built.orders.filter(F.col("profit") != F.round("profit", 2)).count() == 0


# total profit in the aggregate equals the total in the source file
def test_total_profit_matches_source(built, source):
    total = built.agg.agg(F.sum("profit")).first()[0]
    assert abs(total - Decimal(str(round(source.orders["profit"].sum(), 2)))) < Decimal("0.01")


# profit for each year matches the source
def test_profit_by_year_matches_source(built, source):
    got = {r["order_year"]: r["p"] for r in rows(built.agg.groupBy("order_year").agg(F.sum("profit").alias("p")))}
    expected = source.orders.groupby("year")["profit"].sum()
    assert all(abs(got[y] - Decimal(str(round(v, 2)))) < Decimal("0.01") for y, v in expected.items())


# profit for each customer matches the source
def test_profit_by_customer_matches_source(built, source):
    got = {r["customer_id"]: r["p"] for r in rows(built.agg.groupBy("customer_id").agg(F.sum("profit").alias("p")))}
    expected = source.orders.groupby("Customer ID")["profit"].sum()
    assert all(abs(got[c] - Decimal(str(round(v, 2)))) < Decimal("0.01") for c, v in expected.items())


# orders whose product is not in Products.csv are reported as Unknown and their profit is kept
def test_unmatched_products_reported_as_unknown(built, source):
    missing = source.orders[~source.orders["Product ID"].isin(source.products["Product ID"])]
    unknown = built.agg.filter("category = 'Unknown'").agg(F.sum("profit")).first()[0]
    assert built.orders.filter("category is null").count() == len(missing)
    assert abs(unknown - Decimal(str(round(missing["profit"].sum(), 2)))) < Decimal("0.01")


# the four Q5 queries give the same grand total as the aggregate table
def test_sql_queries_reconcile_on_real_data(built, spark):
    built.agg.createOrReplaceTempView("profit_aggregate_real")
    total = built.agg.agg(F.sum("profit")).first()[0]
    for sql in sql_statements("Q5_sql_aggregates"):
        result = spark.sql(sql.replace("ecommerce_sales.default.profit_aggregate", "profit_aggregate_real"))
        assert result.agg(F.sum("profit")).first()[0] == total
