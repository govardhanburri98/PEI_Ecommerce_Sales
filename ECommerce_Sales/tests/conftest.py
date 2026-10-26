import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    # one small local spark session shared by all tests, no databricks connection needed
    session = (SparkSession.builder.master("local[2]").appName("unit-tests")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("warehouse")))
               .getOrCreate())
    yield session
    session.stop()
