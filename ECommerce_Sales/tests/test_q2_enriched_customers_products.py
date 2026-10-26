import pytest
import os

from helpers import load_notebook, rows


@pytest.fixture(scope="module")
def q2(spark):
    return load_notebook("Q2_enriched_customers_products", spark=spark)


def customers(spark, data):
    return spark.createDataFrame(data, "customer_id string, customer_name string, country string")


def products(spark, data):
    return spark.createDataFrame(
        data, "product_id string, category string, sub_category string, product_name string, "
              "state string, price_per_product string")


def clean(q2, spark, name):
    df = spark.createDataFrame([(name,)], "n string")
    return rows(df.select(q2.clean_name(df["n"]).alias("c")))[0]["c"]


# special characters are replaced and the leftover spaces collapse to one
@pytest.mark.parametrize("dirty, expected", [
    ("Tam&^*ara Willing___)ham", "Tam ara Willing ham"),
    ("Jim Mi....tchum", "Jim Mi tchum"),
    ("  Ann    Lee  ", "Ann Lee"),
    ("Gary567 Hansen", "Gary567 Hansen"),           # digits are allowed by the regex
])
def test_clean_name(q2, spark, dirty, expected):
    assert clean(q2, spark, dirty) == expected


# a name made only of symbols ends up as null
def test_clean_name_only_symbols(q2, spark):
    assert clean(q2, spark, "%^&*") is None


# accented letters should survive but the current regex strips them
@pytest.mark.xfail(reason="regex [^a-zA-Z0-9 ] turns 'Bühler' into 'B hler'")
def test_clean_name_keeps_accented_letters(q2, spark):
    assert clean(q2, spark, "Peter Bühler") == "Peter Bühler"


# ids and countries are trimmed and the name is cleaned
def test_enrich_customers_cleans_columns(q2, spark):
    raw = customers(spark, [(" C1 ", "Ann!", " US ")])
    out = rows(q2.enrich_customers(raw))[0]
    assert (out["customer_id"], out["customer_name"], out["country"]) == ("C1", "Ann", "US")


# rows without a customer id (null or blank) are dropped
def test_enrich_customers_drops_missing_ids(q2, spark):
    raw = customers(spark, [("C1", "Ann", "US"), (None, "Ghost", "US"), ("  ", "Blank", "US")])
    assert [r["customer_id"] for r in rows(q2.enrich_customers(raw))] == ["C1"]


# duplicate ids collapse to one row
def test_enrich_customers_dedupes(q2, spark):
    raw = customers(spark, [("C1", "Ann", "US"), ("C1", "Ann", "US"), ("C2", "Bob", "UK")])
    assert q2.enrich_customers(raw).count() == 2


# when ids repeat the row with the alphabetically first name wins, every time
def test_enrich_customers_pick_is_deterministic(q2, spark):
    raw = customers(spark, [("C1", "Zed", "US"), ("C1", "Abe", "US")])
    assert rows(q2.enrich_customers(raw))[0]["customer_name"] == "Abe"


# a missing required column gives a clear error
def test_enrich_customers_missing_column(q2, spark):
    raw = spark.createDataFrame([("C1", "Ann")], "customer_id string, customer_name string")
    with pytest.raises(ValueError, match="country"):
        q2.enrich_customers(raw)


# one row per product id even if the id repeats for several states
def test_enrich_products_one_row_per_id(q2, spark):
    raw = products(spark, [("P1", "Furniture", "Chairs", "Chair", "CA", "10"),
                           ("P1", "Furniture", "Chairs", "Chair", "NY", "12"),
                           ("P2", "Tech", "Phones", "Phone", "CA", "99")])
    assert sorted(r["product_id"] for r in rows(q2.enrich_products(raw))) == ["P1", "P2"]


# products without an id are removed
def test_enrich_products_drops_missing_ids(q2, spark):
    raw = products(spark, [(None, "X", "Y", "Z", "CA", "1"), (" ", "X", "Y", "Z", "CA", "1"),
                           ("P1", "A", "B", "C", "CA", "1")])
    assert [r["product_id"] for r in rows(q2.enrich_products(raw))] == ["P1"]


# blank category text is turned into null and the rest is trimmed
def test_enrich_products_blank_to_null(q2, spark):
    raw = products(spark, [("P1", "  ", " Chairs ", "Chair", "CA", "1")])
    out = rows(q2.enrich_products(raw))[0]
    assert out["category"] is None and out["sub_category"] == "Chairs"


# a missing required column gives a clear error
def test_enrich_products_missing_column(q2, spark):
    raw = spark.createDataFrame([("P1", "Tech")], "product_id string, category string")
    with pytest.raises(ValueError, match="sub_category"):
        q2.enrich_products(raw)
