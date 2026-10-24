"""The three tools the agent can call. Each takes keyword args and returns a string."""
import ast
import html
import operator
import os
import re
import sqlite3
from typing import Any, Callable, Dict, List

import httpx

from .config import settings


class ToolError(Exception):
    """Raised for bad tool input; the message is shown to the model so it can self-correct."""


# ---------- calculator ----------
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("Exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ToolError("Only numbers and + - * / // % ** ( ) are allowed")


def calculator(expression: str) -> str:
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        raise ToolError("Division by zero")
    except SyntaxError:
        raise ToolError("Invalid expression syntax")
    return str(round(result, 10) if isinstance(result, float) else result)


# ---------- web search (Wikipedia) ----------
def web_search(query: str) -> str:
    """Search Wikipedia and return the intro paragraph of the top results."""
    try:
        r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 3,
                "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": 700, "format": "json",
            },
            headers={"User-Agent": "ToolsmithAgent/1.0 (https://github.com/Nandha6124/toolsmith-agent; learning project) httpx"},
            timeout=15,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except (httpx.HTTPError, ValueError) as exc:
        raise ToolError(f"Search failed: {exc}")
    if not pages:
        return "No results found."
    ranked = sorted(pages.values(), key=lambda p: p.get("index", 99))
    return "\n\n".join(
        f"- {p['title']}: {html.unescape(' '.join(p.get('extract', '').split()))}" for p in ranked
    )


# ---------- read-only database query ----------
MAX_ROWS = 50
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I)


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ToolError("Empty query")
    if ";" in sql:
        raise ToolError("Only a single statement is allowed")
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise ToolError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(sql):
        raise ToolError("Query contains a forbidden keyword; the database is read-only")
    return sql


def query_database(sql: str) -> str:
    sql = validate_sql(sql)
    path = os.path.abspath(settings.sample_db_path)
    if not os.path.exists(path):
        raise ToolError("Database is not available")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)  # read-only connection
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as exc:
        raise ToolError(f"SQL error: {exc}")
    finally:
        conn.close()
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    out = [" | ".join(cols)] + [" | ".join(str(v) for v in row) for row in rows]
    if truncated:
        out.append(f"... (truncated to {MAX_ROWS} rows)")
    return "\n".join(out) if rows else "Query returned no rows."


# ---------- registry (also the function declarations sent to the model) ----------
TOOLS: Dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "web_search": web_search,
    "query_database": query_database,
}

DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Use for any math instead of guessing.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"expression": {"type": "STRING", "description": "e.g. (1250 * 0.18) + 40"}},
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": "Search Wikipedia for general knowledge and facts. Returns titles and snippets.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Search terms"}},
            "required": ["query"],
        },
    },
    {
        "name": "query_database",
        "description": (
            "Run a read-only SQL SELECT on the company sample database. Tables: "
            "products(id, name, category, price, stock); "
            "customers(id, name, city); "
            "orders(id, customer_id, product_id, quantity, order_date)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"sql": {"type": "STRING", "description": "A single SELECT statement"}},
            "required": ["sql"],
        },
    },
]


def run_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool; any failure becomes an 'ERROR: ...' observation instead of crashing the agent."""
    fn = TOOLS.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    try:
        return fn(**args)
    except ToolError as exc:
        return f"ERROR: {exc}"
    except TypeError:
        return f"ERROR: bad arguments for {name}: {args}"
