import os

import pytest

os.environ["SAMPLE_DB_PATH"] = "./test_sample.db"

from app.config import settings  # noqa: E402
from app.seed import seed  # noqa: E402
from app.tools import ToolError, calculator, run_tool, validate_sql  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def db():
    settings.sample_db_path = "./test_sample.db"
    seed(settings.sample_db_path)
    yield
    os.remove("./test_sample.db")


def test_calculator_ok():
    assert calculator("(1250 * 0.18) + 40") == "265.0"
    assert calculator("2 ** 10") == "1024"
    assert calculator("-7 // 2") == "-4"


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "1 +", "2 ** 9999", "abs(-1)"])
def test_calculator_rejects_unsafe(bad):
    with pytest.raises(ToolError):
        calculator(bad)


def test_division_by_zero_is_observation_not_crash():
    assert run_tool("calculator", {"expression": "1/0"}).startswith("ERROR")


def test_sql_validation():
    validate_sql("SELECT * FROM products;")
    for bad in ["DROP TABLE products", "SELECT 1; DROP TABLE orders", "DELETE FROM orders", "PRAGMA table_info(orders)", ""]:
        with pytest.raises(ToolError):
            validate_sql(bad)


def test_query_database():
    out = run_tool("query_database", {"sql": "SELECT name, price FROM products WHERE category='Furniture' ORDER BY price"})
    assert "name | price" in out and "Ergonomic Chair | 310.0" in out
    assert run_tool("query_database", {"sql": "SELECT * FROM nope"}).startswith("ERROR: SQL error")
    assert run_tool("query_database", {"sql": "UPDATE products SET price=0"}).startswith("ERROR")


def test_unknown_tool_and_bad_args():
    assert "unknown tool" in run_tool("hack", {})
    assert run_tool("calculator", {"wrong": 1}).startswith("ERROR: bad arguments")
