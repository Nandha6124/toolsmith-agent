import os
import sqlite3

SCHEMA = """
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER, order_date TEXT);
"""

PRODUCTS = [
    (1, "Laptop Pro 14", "Electronics", 1250.0, 15), (2, "Wireless Mouse", "Electronics", 25.5, 120),
    (3, "Mechanical Keyboard", "Electronics", 89.99, 60), (4, "Standing Desk", "Furniture", 420.0, 8),
    (5, "Ergonomic Chair", "Furniture", 310.0, 12), (6, "Notebook 5-pack", "Stationery", 9.5, 300),
    (7, "Gel Pens 10-pack", "Stationery", 6.25, 450), (8, "USB-C Hub", "Electronics", 45.0, 75),
]
CUSTOMERS = [
    (1, "Asha Rao", "Hyderabad"), (2, "Ravi Kumar", "Bengaluru"), (3, "Meera Iyer", "Chennai"),
    (4, "Kiran Patel", "Mumbai"), (5, "Sneha Reddy", "Hyderabad"),
]
ORDERS = [
    (1, 1, 1, 1, "2025-01-10"), (2, 1, 2, 2, "2025-01-10"), (3, 2, 4, 1, "2025-02-03"),
    (4, 3, 3, 1, "2025-02-14"), (5, 4, 6, 10, "2025-03-01"), (6, 5, 1, 2, "2025-03-09"),
    (7, 2, 8, 3, "2025-03-15"), (8, 5, 5, 1, "2025-04-02"), (9, 3, 7, 20, "2025-04-18"),
    (10, 4, 1, 1, "2025-05-05"),
]


def seed(path: str) -> None:
    if os.path.exists(path):
        return
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", ORDERS)
    conn.commit()
    conn.close()
