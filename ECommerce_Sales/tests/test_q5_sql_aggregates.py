from decimal import Decimal as D
import json
import pytest

from helpers import rows, sql_statements

NOTEBOOK = "Q5_sql_aggregates"
REAL_TABLE = "ecommerce_sales.default.profit_aggregate"
TEST_VIEW = "profit_aggregate_test"


@pytest.fixture(scope="module")
def queries(spark):
    # same rows the Q4 table would hold; the real table name is swapped for a temp view
    data = [(2016, "Furniture", "Chairs", "C1", "Ann", D("10.00")),
            (2016, "Tech", "Phones", "C1", "Ann", D("5.50")),
            (2016, "Tech", "Phones", "C2", "Bob", D("4.50")),
            (2017, "Furniture", "Chairs", "C1", "Ann", D("-3.00")),
            (2017, "Unknown", "Unknown", "C3", "Unknown", D("2.00"))]
    spark.createDataFrame(data, "order_year int, category string, sub_category string, customer_id string, "
                                "customer_name string, profit decimal(36,2)").createOrReplaceTempView(TEST_VIEW)
    return [s.replace(REAL_TABLE, TEST_VIEW) for s in sql_statements(NOTEBOOK)]


def run(spark, queries, i):
    return rows(spark.sql(queries[i]))


# the notebook has exactly the four requested queries
def test_four_queries_found():
    assert len(sql_statements(NOTEBOOK)) == 4


# 5.1 profit by year
def test_profit_by_year(spark, queries):
    out = [(r["order_year"], r["profit"]) for r in run(spark, queries, 0)]
    assert out == [(2016, D("20.00")), (2017, D("-1.00"))]


# 5.2 profit by year and category
def test_profit_by_year_and_category(spark, queries):
    out = [(r["order_year"], r["category"], r["profit"]) for r in run(spark, queries, 1)]
    assert out == [(2016, "Furniture", D("10.00")), (2016, "Tech", D("10.00")),
                   (2017, "Furniture", D("-3.00")), (2017, "Unknown", D("2.00"))]


# 5.3 profit by customer
def test_profit_by_customer(spark, queries):
    out = {r["customer_id"]: r["profit"] for r in run(spark, queries, 2)}
    assert out == {"C1": D("12.50"), "C2": D("4.50"), "C3": D("2.00")}


# 5.4 profit by customer and year
def test_profit_by_customer_and_year(spark, queries):
    out = {(r["customer_id"], r["order_year"]): r["profit"] for r in run(spark, queries, 3)}
    assert out == {("C1", 2016): D("15.50"), ("C1", 2017): D("-3.00"),
                   ("C2", 2016): D("4.50"), ("C3", 2017): D("2.00")}


# all four queries add up to the same grand total, so nothing is dropped or doubled
def test_all_queries_reconcile(spark, queries):
    totals = {sum(r["profit"] for r in run(spark, queries, i)) for i in range(4)}
    assert totals == {D("19.00")}
