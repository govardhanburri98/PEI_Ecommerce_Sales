from decimal import Decimal as D
import os
import pytest

from helpers import load_notebook, rows

SCHEMA = ("order_year int, category string, sub_category string, customer_id string, "
          "customer_name string, profit decimal(36,2)")


@pytest.fixture(scope="module")
def q4(spark):
    return load_notebook("Q4_profit_aggregate", spark=spark)


def frame(spark, data):
    return spark.createDataFrame(data, SCHEMA)


# profit is added up for every year/category/sub-category/customer combination
def test_profit_summed_per_group(q4, spark):
    df = frame(spark, [(2016, "Furniture", "Chairs", "C1", "Ann", D("1.10")),
                       (2016, "Furniture", "Chairs", "C1", "Ann", D("2.20")),
                       (2016, "Furniture", "Tables", "C1", "Ann", D("5.00")),
                       (2017, "Furniture", "Chairs", "C1", "Ann", D("4.00"))])
    out = {(r["order_year"], r["sub_category"]): r["profit"] for r in rows(q4.build_profit_aggregate(df))}
    assert out == {(2016, "Chairs"): D("3.30"), (2016, "Tables"): D("5.00"), (2017, "Chairs"): D("4.00")}


# decimals add up exactly with no float noise (0.10 + 0.20 is 0.30)
def test_sum_is_exact(q4, spark):
    df = frame(spark, [(2016, "A", "B", "C1", "Ann", D("0.10")), (2016, "A", "B", "C1", "Ann", D("0.20"))])
    assert str(rows(q4.build_profit_aggregate(df))[0]["profit"]) == "0.30"


# missing category, sub category and customer name show up as Unknown
def test_nulls_labelled_unknown(q4, spark):
    df = frame(spark, [(2016, None, None, "C9", None, D("5.00")), (2016, None, None, "C9", None, D("1.00"))])
    out = rows(q4.build_profit_aggregate(df))
    assert len(out) == 1 and out[0]["profit"] == D("6.00")
    assert (out[0]["category"], out[0]["sub_category"], out[0]["customer_name"]) == ("Unknown",) * 3


# nothing is lost, the grand total equals the input total
def test_total_is_preserved(q4, spark):
    df = frame(spark, [(2016, "A", "B", "C1", "Ann", D("1.11")), (2017, "A", "C", "C2", "Bob", D("2.22")),
                       (2017, None, None, "C3", None, D("3.33"))])
    assert sum(r["profit"] for r in rows(q4.build_profit_aggregate(df))) == D("6.66")


# output columns are the group columns followed by profit
def test_output_columns(q4, spark):
    out = q4.build_profit_aggregate(frame(spark, [(2016, "A", "B", "C1", "Ann", D("1"))]))
    assert out.columns == q4.GROUP_COLUMNS + ["profit"]


# missing columns raise an error that names the table
def test_missing_columns(q4, spark):
    with pytest.raises(ValueError, match="enriched_orders"):
        q4.build_profit_aggregate(spark.createDataFrame([(1,)], "order_year int"))
