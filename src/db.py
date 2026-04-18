"""Toy e-commerce SQLite DB the NL->SQL assistant queries against."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "shop.db"

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    signup_date TEXT
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price REAL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    order_date TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);
"""

CUSTOMERS = [
    (1, "Asha Rao", "Bangalore", "2024-01-10"),
    (2, "Ben Silva", "Mumbai", "2024-02-15"),
    (3, "Chen Wu", "Bangalore", "2024-03-02"),
    (4, "Divya Menon", "Delhi", "2024-03-20"),
    (5, "Evan Kim", "Mumbai", "2024-04-05"),
]

PRODUCTS = [
    (1, "Wireless Mouse", "Electronics", 799.0),
    (2, "Office Chair", "Furniture", 5499.0),
    (3, "Notebook Set", "Stationery", 199.0),
    (4, "USB-C Hub", "Electronics", 1299.0),
    (5, "Desk Lamp", "Furniture", 899.0),
]

ORDERS = [
    (1, 1, 1, 2, "2024-05-01"),
    (2, 1, 3, 5, "2024-05-03"),
    (3, 2, 2, 1, "2024-05-04"),
    (4, 3, 4, 1, "2024-05-10"),
    (5, 3, 1, 1, "2024-05-11"),
    (6, 4, 5, 2, "2024-05-12"),
    (7, 5, 2, 1, "2024-05-15"),
    (8, 1, 4, 1, "2024-05-20"),
    (9, 2, 1, 3, "2024-05-21"),
    (10, 3, 3, 2, "2024-05-22"),
]


def build_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", ORDERS)
    conn.commit()
    conn.close()


def get_schema_text() -> str:
    return SCHEMA


if __name__ == "__main__":
    build_db()
    print(f"built {DB_PATH}")
