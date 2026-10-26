import pytest
import os

from helpers import load_notebook, rows

RAW_SCHEMA = ("row_id string, order_id string, order_date string, ship_date string, ship_mode string, "
              "customer_id string, product_id string, quantity string, price string, discount string, "
              "profit string")


@pytest.fixture(scope="module")
def q3(spark):
    return load_notebook("Q3_enriched_orders", spark=spark)


def order(**kw):
    """One raw order row as a tuple, any column can be overridden."""
    base = dict(row_id="1", order_id="O1", order_date="21/8/2016", ship_date="25/8/2016",
                ship_mode="Standard", customer_id="C1", product_id="P1", quantity="2",
                price="10.5", discount="0.1", profit="3.456")
    base.update(kw)
    return tuple(base.values())


def raw(spark, *orders):
    return spark.createDataFrame(list(orders), RAW_SCHEMA)


@pytest.fixture
def lookups(spark):
    cust = spark.createDataFrame([("C1", "Ann", "US"), ("C2", "Bob", "UK")],
                                 "customer_id string, customer_name string, country string")
    prod = spark.createDataFrame([("P1", "Furniture", "Chairs"), ("P2", "Tech", "Phones")],
                                 "product_id string, category string, sub_category string")
    return cust, prod


# profit is rounded to 2 decimals and stored as a decimal, not a float
def test_clean_orders_profit_rounded(q3, spark):
    out = q3.clean_orders(raw(spark, order(profit="14.5678")))
    assert dict(out.dtypes)["profit"] == "decimal(36,2)"
    assert str(rows(out)[0]["profit"]) == "14.57"


# text that is not a number becomes null profit but the order stays
def test_clean_orders_bad_profit(q3, spark):
    out = rows(q3.clean_orders(raw(spark, order(profit="abc"))))
    assert len(out) == 1 and out[0]["profit"] is None


# order date is parsed and the year is taken from it
def test_clean_orders_order_year(q3, spark):
    out = rows(q3.clean_orders(raw(spark, order(order_date="1/2/2017"))))[0]
    assert str(out["order_date"]) == "2017-02-01" and out["order_year"] == 2017


# an invalid date gives null date and null year, not an error
def test_clean_orders_bad_date(q3, spark):
    out = rows(q3.clean_orders(raw(spark, order(order_date="not a date"))))[0]
    assert out["order_date"] is None and out["order_year"] is None


# rows flagged in _corrupt_record are removed and the column disappears
def test_clean_orders_drops_corrupt_rows(q3, spark):
    good = order() + (None,)
    broken = (None,) * 11 + ("{broken",)
    df = spark.createDataFrame([good, broken], RAW_SCHEMA + ", _corrupt_record string")
    out = q3.clean_orders(df)
    assert "_corrupt_record" not in out.columns and out.count() == 1


# an exact duplicate row is kept only once
def test_clean_orders_drops_exact_duplicates(q3, spark):
    assert q3.clean_orders(raw(spark, order(), order())).count() == 1


# same order id on different rows (different products) is not a duplicate
def test_clean_orders_keeps_multi_line_orders(q3, spark):
    df = raw(spark, order(row_id="1", product_id="P1"), order(row_id="2", product_id="P2"))
    assert q3.clean_orders(df).count() == 2


# rows with a null or blank order id are dropped
def test_clean_orders_drops_missing_order_id(q3, spark):
    df = raw(spark, order(), order(row_id="2", order_id=None), order(row_id="3", order_id="  "))
    assert q3.clean_orders(df).count() == 1


# missing required columns raise an error that names the table
def test_clean_orders_missing_columns(q3, spark):
    with pytest.raises(ValueError, match="raw_orders"):
        q3.clean_orders(spark.createDataFrame([("O1",)], "order_id string"))


# customer name/country and product category/sub category are added
def test_enrich_orders_adds_attributes(q3, spark, lookups):
    out = rows(q3.enrich_orders(raw(spark, order(customer_id="C2", product_id="P2")), *lookups))[0]
    assert (out["customer_name"], out["country"]) == ("Bob", "UK")
    assert (out["category"], out["sub_category"]) == ("Tech", "Phones")


# unknown customer or product keeps the order, with nulls for the lookups
def test_enrich_orders_keeps_unmatched(q3, spark, lookups):
    out = rows(q3.enrich_orders(raw(spark, order(customer_id="C9", product_id="P9", profit="7")), *lookups))[0]
    assert float(out["profit"]) == 7.0
    assert out["customer_name"] is None and out["category"] is None


# the joins must not multiply orders
def test_enrich_orders_no_fan_out(q3, spark, lookups):
    df = raw(spark, order(row_id="1"), order(row_id="2"), order(row_id="3", order_id="O2"))
    assert q3.enrich_orders(df, *lookups).count() == 3


# customers without the needed columns are rejected with a clear message
def test_enrich_orders_bad_customers(q3, spark, lookups):
    bad = spark.createDataFrame([("C1",)], "customer_id string")
    with pytest.raises(ValueError, match="enriched_customers"):
        q3.enrich_orders(raw(spark, order()), bad, lookups[1])
