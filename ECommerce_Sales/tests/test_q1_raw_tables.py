import json
import os
import pandas as pd
import pytest

from helpers import load_notebook, rows


@pytest.fixture(scope="module")
def q1(spark):
    return load_notebook("Q1_raw_tables", spark=spark)


# _to_raw renames columns to snake_case and adds the two lineage columns
def test_to_raw_adds_lineage_columns(q1, spark):
    df = spark.createDataFrame([("a", "b")], ["Row ID", "Sub-Category"])
    out = q1._to_raw(df, "some/path.csv")
    assert out.columns == ["row_id", "sub_category", "_source", "_ingested_at"]
    assert rows(out)[0]["_source"] == "some/path.csv"


# numbers come out as strings so nothing gets silently converted
def test_to_raw_casts_everything_to_string(q1, spark):
    df = spark.createDataFrame([(1, 2.5)], ["Qty", "Price"])
    out = q1._to_raw(df, "x")
    assert dict(out.dtypes)["qty"] == "string" and rows(out)[0]["price"] == "2.5"


# products csv is read with snake_case headers and untouched string values
def test_read_products(q1, spark, tmp_path):
    f = tmp_path / "Products.csv"
    f.write_text("Product ID,Category,Sub-Category,Product Name,State,Price per product\n"
                 "P1,Furniture,Chairs,Red Chair,CA,10.50\n")
    out = rows(q1.read_products(spark, str(f)))[0]
    assert out["product_id"] == "P1" and out["sub_category"] == "Chairs"
    assert out["price_per_product"] == "10.50"          # not turned into 10.5


# a comma inside quotes must not split the product name into two columns
def test_read_products_quoted_comma(q1, spark, tmp_path):
    f = tmp_path / "Products.csv"
    f.write_text('Product ID,Category,Sub-Category,Product Name,State,Price per product\n'
                 'P1,Tech,Phones,"Phone, Black",CA,99\n')
    assert rows(q1.read_products(spark, str(f)))[0]["product_name"] == "Phone, Black"


# a renamed header must fail instead of silently shifting columns
def test_read_products_wrong_header_fails(q1, spark, tmp_path):
    f = tmp_path / "Products.csv"
    f.write_text("Product ID,Cat,Sub-Category,Product Name,State,Price per product\nP1,x,y,z,CA,1\n")
    with pytest.raises(Exception):
        q1.read_products(spark, str(f)).collect()


ORDER = {"Row ID": 1, "Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1",
         "Product ID": "P1", "Quantity": 2, "Price": 10.5, "Discount": 0.2, "Profit": 14.5}


# a normal json array is read and numbers are kept as text
def test_read_orders_json_array(q1, spark, tmp_path):
    f = tmp_path / "Orders.json"
    f.write_text(json.dumps([ORDER]))
    out = rows(q1.read_orders(spark, str(f)))[0]
    assert out["order_id"] == "O1" and out["profit"] == "14.5" and out["quantity"] == "2"


# a pretty printed array (like the real file) is detected and read fully
def test_read_orders_pretty_printed_array(q1, spark, tmp_path):
    f = tmp_path / "Orders.json"
    f.write_text(json.dumps([ORDER, dict(ORDER, **{"Order ID": "O2"})], indent=4))
    assert q1.read_orders(spark, str(f)).count() == 2


# a broken json line ends up in _corrupt_record, the good line is still read
def test_read_orders_flags_corrupt_line(q1, spark, tmp_path):
    f = tmp_path / "Orders.json"
    f.write_text(json.dumps(ORDER) + "\n{this is not json}\n")
    out = rows(q1.read_orders(spark, str(f)))
    assert sum(1 for r in out if r["_corrupt_record"]) == 1
    assert sum(1 for r in out if r["order_id"] == "O1") == 1


# excel customers are read with snake_case columns and empty cells as null
def test_read_customers_blank_cells(q1, spark, tmp_path):
    f = tmp_path / "Customer.xlsx"
    pd.DataFrame({"Customer ID": ["C1", "C2"], "Customer Name": ["Ann", None],
                  "Country": ["US", "UK"]}).to_excel(f, index=False)
    out = {r["customer_id"]: r for r in rows(q1.read_customers(spark, str(f)))}
    assert out["C1"]["customer_name"] == "Ann" and out["C2"]["customer_name"] is None


# ids like 00123 must keep their leading zeros (everything is read as text)
def test_read_customers_keeps_leading_zeros(q1, spark, tmp_path):
    f = tmp_path / "Customer.xlsx"
    pd.DataFrame({"Customer ID": ["00123"], "Customer Name": ["Ann"], "Country": ["US"]}).to_excel(f, index=False)
    assert rows(q1.read_customers(spark, str(f)))[0]["customer_id"] == "00123"
