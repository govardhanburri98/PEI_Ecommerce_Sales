import pytest
from pyspark.sql import functions as F

from helpers import load_notebook, rows


@pytest.fixture(scope="module")
def u(spark):
    return load_notebook("utils", spark=spark)


# spaces, dashes and mixed case all turn into lower snake_case
@pytest.mark.parametrize("raw, expected", [
    ("Sub-Category", "sub_category"),
    ("Price per product", "price_per_product"),
    ("  Row ID  ", "row_id"),
    ("Order  Date!", "order_date"),
])
def test_to_snake_case(u, raw, expected):
    assert u.to_snake_case(raw) == expected


# every normal column gets renamed
def test_standardise_columns_renames(u, spark):
    df = spark.createDataFrame([("a", "b")], ["Row ID", "Sub-Category"])
    assert u.standardise_columns(df).columns == ["row_id", "sub_category"]


# two columns that end up with the same name should fail loudly
def test_standardise_columns_collision(u, spark):
    df = spark.createDataFrame([("a", "b")], ["Row ID", "row-id"])
    with pytest.raises(ValueError, match="collide"):
        u.standardise_columns(df)


# the error names the dataframe and lists the missing columns sorted
def test_require_columns_missing(u, spark):
    df = spark.createDataFrame([(1,)], ["a"])
    with pytest.raises(ValueError, match=r"demo is missing required columns: \['y', 'z'\]"):
        u.require_columns(df, ["a", "z", "y"], "demo")


# good strings are cast, bad strings and nulls come back as null (no exception)
def test_safe_cast_double(u, spark):
    df = spark.createDataFrame([("1.5",), ("abc",), (None,)], "v string")
    out = [r["c"] for r in df.select(u.safe_cast("v", "double").alias("c")).collect()]
    assert out == [1.5, None, None]


# a normal d/M/yyyy date is parsed, single digit day and month are fine
def test_parse_date_default_format(u, spark):
    df = spark.createDataFrame([("21/8/2016",), ("1/1/2017",)], "d string")
    out = [str(r["p"]) for r in df.select(u.parse_date("d", "d/M/yyyy").alias("p")).collect()]
    assert out == ["2016-08-21", "2017-01-01"]


# unparseable or missing dates become null instead of crashing
def test_parse_date_bad_values(u, spark):
    df = spark.createDataFrame([("garbage",), (None,), ("32/13/2016",)], "d string")
    out = [r["p"] for r in df.select(u.parse_date("d", "d/M/yyyy").alias("p")).collect()]
    assert out == [None, None, None]


# text is trimmed, empty and whitespace-only values become null
def test_blank_to_null(u, spark):
    df = spark.createDataFrame([("  hi  ",), ("",), ("   ",), (None,)], "v string")
    out = [r["c"] for r in df.select(u.blank_to_null(F.col("v")).alias("c")).collect()]
    assert out == ["hi", None, None, None]


# only columns that exist are selected and the requested order is kept
def test_select_existing(u, spark):
    df = spark.createDataFrame([(1, 2, 3)], ["a", "b", "c"])
    assert u.select_existing(df, ["c", "zzz", "a"]).columns == ["c", "a"]


# default mode is overwrite so a second write replaces the first
def test_write_table_overwrites(u, spark):
    first = spark.createDataFrame([(1,), (2,)], ["id"])
    second = spark.createDataFrame([(9,)], ["id"])
    u.write_table(first, "default.t_utils_overwrite", fmt="parquet")
    u.write_table(second, "default.t_utils_overwrite", fmt="parquet")
    try:
        assert [r["id"] for r in spark.table("default.t_utils_overwrite").collect()] == [9]
    finally:
        spark.sql("DROP TABLE IF EXISTS default.t_utils_overwrite")


# an unknown catalog is rejected and the error lists what exists
def test_qualified_name_unknown_catalog(u):
    with pytest.raises(ValueError, match="Catalog 'nope' does not exist"):
        u.qualified_name("nope", "default", "t")
