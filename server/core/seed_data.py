"""中文电商演示库种子数据生成器（DuckDB）。

生成带中文表/列注释的业务库 —— 这些注释就是 schema linking 的"业务黑话词典"，
是 ChatBI 场景下非常规 RAG 的载体。确定性随机（固定 seed），保证实验可复现。

用法：python -m server.core.seed_data --rows 100000 --path data/ecommerce.duckdb
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

CATEGORIES = [
    "手机数码", "电脑办公", "家用电器", "服饰鞋包", "美妆个护",
    "食品生鲜", "母婴玩具", "图书文娱", "运动户外", "家居家装",
]
BRANDS = ["星辰", "云杉", "极光", "海豚", "青竹", "流星", "白鹭", "远山"]
PRODUCT_NOUNS = {
    "手机数码": ["手机", "耳机", "充电宝", "智能手表"],
    "电脑办公": ["笔记本电脑", "机械键盘", "显示器", "打印机"],
    "家用电器": ["空调", "冰箱", "扫地机器人", "电饭煲"],
    "服饰鞋包": ["卫衣", "运动鞋", "双肩包", "羽绒服"],
    "美妆个护": ["洗面奶", "防晒霜", "洗发水", "口红"],
    "食品生鲜": ["坚果礼盒", "牛奶", "咖啡豆", "大米"],
    "母婴玩具": ["积木", "婴儿车", "绘本", "安抚玩偶"],
    "图书文娱": ["小说", "编程书", "漫画", "台历"],
    "运动户外": ["跑步鞋", "瑜伽垫", "帐篷", "跳绳"],
    "家居家装": ["台灯", "收纳箱", "四件套", "香薰"],
}
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆",
          "长沙", "郑州", "青岛", "苏州", "东莞", "佛山", "宁波", "合肥", "福州", "厦门"]
STATUSES = ["已完成", "已发货", "已支付", "待支付", "已退款"]
STATUS_WEIGHTS = [0.55, 0.15, 0.12, 0.08, 0.10]

TABLE_COMMENTS = {
    "categories": "商品类目维表",
    "products": "商品维表，含所属类目与定价",
    "users": "用户维表，含城市与注册时间",
    "orders": "订单事实表，amount 为订单应付总额（元）",
    "order_items": "订单明细事实表，一行代表订单内一种商品的购买记录",
}
COLUMN_COMMENTS = {
    "orders.created_at": "下单时间（精确到秒）",
    "orders.amount": "订单实付总额（元），已扣除优惠",
    "orders.status": "订单状态：待支付/已支付/已发货/已完成/已退款",
    "order_items.quantity": "购买数量（件）",
    "order_items.unit_price": "成交单价（元），可能低于商品定价（促销）",
    "products.price": "商品标准定价（元）",
    "users.city": "用户收货城市",
    "users.registered_at": "注册时间",
}


def _gen_products(rng: random.Random, categories: list[tuple[int, str]]) -> list[tuple]:
    products = []
    pid = 1
    for cat_id, cat_name in categories:
        for _ in range(rng.randint(150, 260)):
            noun = rng.choice(PRODUCT_NOUNS[cat_name])
            name = f"{rng.choice(BRANDS)}{noun}"
            price = round(rng.uniform(19, 8999), 2)
            products.append((pid, name, cat_id, price, rng.randint(0, 5000)))
            pid += 1
    return products


def generate(path: str, n_orders: int = 100_000, seed: int = 42) -> dict[str, int]:
    import duckdb

    rng = random.Random(seed)
    conn = duckdb.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS order_items")
        conn.execute("DROP TABLE IF EXISTS orders")
        conn.execute("DROP TABLE IF EXISTS products")
        conn.execute("DROP TABLE IF EXISTS users")
        conn.execute("DROP TABLE IF EXISTS categories")

        conn.execute("CREATE TABLE categories (category_id INTEGER, name VARCHAR)")
        conn.execute(
            "CREATE TABLE products (product_id INTEGER, name VARCHAR, category_id INTEGER, "
            "price DOUBLE, stock INTEGER)"
        )
        conn.execute(
            "CREATE TABLE users (user_id INTEGER, name VARCHAR, gender VARCHAR, "
            "city VARCHAR, registered_at TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE orders (order_id INTEGER, user_id INTEGER, status VARCHAR, "
            "amount DOUBLE, created_at TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE order_items (item_id INTEGER, order_id INTEGER, product_id INTEGER, "
            "quantity INTEGER, unit_price DOUBLE)"
        )

        categories = [(i + 1, name) for i, name in enumerate(CATEGORIES)]
        products = _gen_products(rng, categories)
        product_ids = [p[0] for p in products]
        product_price = {p[0]: p[3] for p in products}

        users = []
        for uid in range(1, max(1000, n_orders // 20) + 1):
            reg = datetime(2023, 1, 1) + timedelta(seconds=rng.randint(0, 3600 * 24 * 730))
            users.append((uid, f"用户{uid:06d}", rng.choice(["男", "女"]), rng.choice(CITIES), reg))

        orders, items = [], []
        item_id = 1
        base = datetime(2024, 1, 1)
        span_seconds = int((datetime(2025, 8, 31, 23, 59, 59) - base).total_seconds())
        # 时间加权：近期订单更多（构造"增长趋势"，让演示图表更有故事）
        for oid in range(1, n_orders + 1):
            u = rng.choice(users)
            created = base + timedelta(seconds=int(rng.triangular(0, span_seconds, span_seconds * 0.8)))
            n_items = rng.choices([1, 2, 3, 4, 5], weights=[0.45, 0.28, 0.15, 0.08, 0.04])[0]
            amount = 0.0
            for _ in range(n_items):
                pid = rng.choice(product_ids)
                qty = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
                unit_price = round(product_price[pid] * rng.uniform(0.75, 1.0), 2)
                amount += qty * unit_price
                items.append((item_id, oid, pid, qty, unit_price))
                item_id += 1
            status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
            orders.append((oid, u[0], status, round(amount, 2), created))

        def _insert(table: str, rows: list[tuple]) -> None:
            for i in range(0, len(rows), 10_000):
                conn.executemany(
                    f"INSERT INTO {table} VALUES ({', '.join('?' * len(rows[0]))})",
                    rows[i : i + 10_000],
                )

        _insert("categories", categories)
        _insert("products", products)
        _insert("users", users)
        _insert("orders", orders)
        _insert("order_items", items)

        for t, comment in TABLE_COMMENTS.items():
            conn.execute(f"COMMENT ON TABLE {t} IS '{comment}'")
        for col, comment in COLUMN_COMMENTS.items():
            conn.execute(f"COMMENT ON COLUMN {col} IS '{comment}'")

        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    counts = {
        "categories": len(CATEGORIES),
        "products": len(products),
        "users": len(users),
        "orders": len(orders),
        "order_items": len(items),
    }
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="生成中文电商演示库")
    parser.add_argument("--path", default="data/ecommerce.duckdb")
    parser.add_argument("--rows", type=int, default=100_000, help="订单行数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.path).parent.mkdir(parents=True, exist_ok=True)
    counts = generate(args.path, n_orders=args.rows, seed=args.seed)
    print(f"[seed_data] 已生成 {args.path}")
    for table, n in counts.items():
        print(f"  {table:<12} {n:>8} rows")


if __name__ == "__main__":
    main()
