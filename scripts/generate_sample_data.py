import os
import random
from datetime import date, timedelta
import pandas as pd

# Set fixed seed for reproducibility
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def random_date(start_year=2021, end_year=2024):
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 6, 30)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    d = start_date + timedelta(days=random_days)
    
    fmt_choice = random.choice(["iso", "us", "custom"])
    if fmt_choice == "iso":
        return d.strftime("%Y-%m-%d")
    elif fmt_choice == "us":
        return d.strftime("%m/%d/%Y")
    else:
        return f"{d.day:02d}-{MONTHS[d.month - 1]}-{d.year}"


def generate_products():
    """
    60 rows:
    - columns: product_id, product_name, category, brand, sku, price
    - At least 8 variant rows across ~4 base products with casing/spelling differences
    - Category variants (e.g. "Electronics", "electronics ", "Electronic")
    - ~5 rows with missing/blank price
    - ~3 rows with leading/trailing whitespace in product_name or sku
    - 2 sku values exact duplicates of another row
    """
    brands = ["Apex", "Vertex", "Nova", "Aura", "Peak", "Zenith", "Pulse", "Core"]

    base_variants = [
        # (base_category, [(name, category)])
        ("Apparel", [
            ("T-Shirt", "Apparel"),
            ("T Shirt", "apparel"),
            ("tee", "Apparel "),
            ("TShirt", "Apparel"),
        ]),
        ("Footwear", [
            ("Running Shoes", "Footwear"),
            ("running shoes", "footwear"),
            ("Run Shoes", "Footwear "),
        ]),
        ("Electronics", [
            ("Wireless Headphones", "Electronics"),
            ("wireless headphones", "electronics "),
            ("Electronic Headphones", "Electronic"),
        ]),
        ("Home & Kitchen", [
            ("Coffee Maker", "Home & Kitchen"),
            ("coffee maker", "Home & Kitchen "),
            ("Coffee-Maker", "HOME & KITCHEN"),
        ]),
    ]

    other_products = [
        ("Slim Jeans", "Apparel"),
        ("Leather Belt", "Apparel"),
        ("Winter Jacket", "Apparel"),
        ("Cotton Hoodie", "Apparel"),
        ("Wool Scarf", "Apparel"),
        ("Baseball Cap", "Apparel"),
        ("Hiking Boots", "Footwear"),
        ("Trail Sneakers", "Footwear"),
        ("Casual Loafers", "Footwear"),
        ("Slip-On Sandals", "Footwear"),
        ("Bluetooth Speaker", "Electronics"),
        ("Smart Watch", "Electronics"),
        ("USB-C Cable", "Electronics"),
        ("Power Bank", "Electronics"),
        ("Mechanical Keyboard", "Electronics"),
        ("Gaming Mouse", "Electronics"),
        ("27-inch Monitor", "Electronics"),
        ("Webcam HD", "Electronics"),
        ("Electric Kettle", "Home & Kitchen"),
        ("Blender 500W", "Home & Kitchen"),
        ("Toaster 2-Slice", "Home & Kitchen"),
        ("Air Fryer", "Home & Kitchen"),
        ("Ceramic Pan", "Home & Kitchen"),
        ("Chef Knife 8in", "Home & Kitchen"),
        ("Stainless Water Bottle", "Sports & Outdoors"),
        ("Yoga Mat", "Sports & Outdoors"),
        ("Resistance Bands", "Sports & Outdoors"),
        ("Foam Roller", "Sports & Outdoors"),
        ("Camping Tent", "Sports & Outdoors"),
        ("Sleeping Bag", "Sports & Outdoors"),
        ("Hiking Backpack", "Sports & Outdoors"),
        ("Trekking Poles", "Sports & Outdoors"),
        ("LED Desk Lamp", "Home & Office"),
        ("Ergonomic Chair", "Home & Office"),
        ("Standing Desk Mat", "Home & Office"),
        ("Monitor Arm", "Home & Office"),
        ("Cable Organizer", "Home & Office"),
        ("Hardcover Notebook", "Home & Office"),
        ("Gel Pens 10pk", "Home & Office"),
        ("Desk Pad Felt", "Home & Office"),
        ("Ceramic Mug", "Home & Kitchen"),
        ("Insulated Tumbler", "Home & Kitchen"),
        ("Canvas Tote Bag", "Apparel"),
        ("Sunglasses Polarized", "Apparel"),
        ("Ankle Socks 3pk", "Apparel"),
        ("Fitness Tracker", "Electronics"),
        ("Noise Cancelling Earbuds", "Electronics"),
    ]

    records = []
    # Add variant items (13 items total across 4 base products)
    for _, items in base_variants:
        for name, cat in items:
            records.append((name, cat))

    # Add remaining to reach 60 products
    remaining_needed = 60 - len(records)
    for i in range(remaining_needed):
        records.append(other_products[i % len(other_products)])

    # Shuffle slightly with fixed seed to distribute
    random.shuffle(records)

    rows = []
    for idx, (name, cat) in enumerate(records, start=1):
        pid = f"P{idx:03d}"
        brand = random.choice(brands)
        sku = f"SKU-{1000 + idx}"
        price = round(random.uniform(9.99, 299.99), 2)
        rows.append({
            "product_id": pid,
            "product_name": name,
            "category": cat,
            "brand": brand,
            "sku": sku,
            "price": price,
        })

    # Apply specific messy requirements:
    # 1. ~5 rows have a missing/blank price
    blank_price_indices = [3, 14, 27, 41, 55]
    for i in blank_price_indices:
        rows[i]["price"] = None

    # 2. ~3 rows have leading/trailing whitespace in product_name or sku
    rows[6]["product_name"] = "  " + rows[6]["product_name"]
    rows[18]["sku"] = rows[18]["sku"] + "  "
    rows[32]["product_name"] = rows[32]["product_name"] + " "

    # 3. 2 sku values exact duplicates of another row
    rows[20]["sku"] = rows[5]["sku"]
    rows[45]["sku"] = rows[12]["sku"]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    return df


def generate_customers():
    """
    80 rows:
    - columns: customer_id, name, email, region, signup_date
    - region values: North, South, East, West with messy variants ("west", "WEST", " West", etc.)
    - signup_date: inconsistent formats ("YYYY-MM-DD", "MM/DD/YYYY", "DD-Mon-YYYY")
    - ~4 rows exact duplicate customer records (same customer_id repeated)
    - ~3 emails malformed (missing @, random casing)
    """
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
        "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    ]
    regions_clean = ["North", "South", "East", "West"]
    region_variants = ["North", "South", "East", "West", "west", "WEST", " West", " North", "south", "EAST"]

    # We need 80 total rows, with ~4 exact duplicates. So 76 unique records, then duplicate 4.
    rows = []
    for i in range(1, 77):
        cid = f"C{i:03d}"
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        name = f"{fn} {ln}"
        domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "example.com"])
        email = f"{fn.lower()}.{ln.lower()}@{domain}"
        region = random.choice(region_variants)
        s_date = random_date()
        rows.append({
            "customer_id": cid,
            "name": name,
            "email": email,
            "region": region,
            "signup_date": s_date,
        })

    # ~3 emails malformed (missing @, random casing)
    rows[7]["email"] = "jennifer.brown.example.com"  # missing @
    rows[22]["email"] = "MICHAEL_DAVIS!GMAIL.COM"    # missing @, caps
    rows[39]["email"] = "robert.miller@@yahoo..com"   # malformed casing/syntax

    # ~4 rows exact duplicate customer records (duplicate 4 existing rows)
    dup_indices = [5, 15, 30, 50]
    for idx in dup_indices:
        rows.append(dict(rows[idx]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    return df


def generate_stores():
    """
    10 rows:
    - columns: store_id, store_name, region, city
    - region same 4 values as customers, with similar messy casing variants
    - 1 duplicate store_id row (total 10 rows: 9 unique stores + 1 duplicate)
    """
    store_info = [
        ("Downtown Flagship", "East", "New York"),
        ("Bay Area Tech Hub", "West", "San Francisco"),
        ("Midwest Central", "North", "Chicago"),
        ("Lone Star Plaza", "South", "Dallas"),
        ("Pacific Galleria", "west", "Seattle"),
        ("Northwest Square", " North", "Portland"),
        ("Sunshine Mall", "South ", "Miami"),
        ("Liberty Center", "EAST", "Philadelphia"),
        ("Rocky Mountain Outlet", "West", "Denver"),
    ]

    rows = []
    for i, (name, reg, city) in enumerate(store_info, start=1):
        rows.append({
            "store_id": f"S{i:02d}",
            "store_name": name,
            "region": reg,
            "city": city,
        })

    # 1 duplicate store_id row
    dup_row = dict(rows[2])  # Duplicate S03
    rows.append(dup_row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "stores.csv"), index=False)
    return df


def generate_orders(products_df, customers_df, stores_df):
    """
    300 rows:
    - columns: order_id, customer_id, product_id, store_id, quantity, revenue, order_date, region
    - order_date same inconsistent formats as signup_date
    - ~15 rows reference a customer_id or product_id that does NOT exist (broken join keys)
    - ~10 rows have negative or null revenue
    - ~5 rows have quantity stored as a string like "5 units"
    - 3 exact duplicate order_id rows
    - region should mostly (not always) match the customer's region
    """
    valid_customer_ids = list(customers_df["customer_id"].unique())
    valid_product_ids = list(products_df["product_id"].unique())
    valid_store_ids = list(stores_df["store_id"].unique())
    
    # Map customer to their primary cleaned region
    customer_region_map = dict(zip(customers_df["customer_id"], customers_df["region"]))

    rows = []
    total_unique_orders = 297  # + 3 duplicates = 300 rows

    for i in range(1, total_unique_orders + 1):
        oid = f"ORD-{i:04d}"
        cid = random.choice(valid_customer_ids)
        pid = random.choice(valid_product_ids)
        sid = random.choice(valid_store_ids)
        qty = random.randint(1, 6)
        rev = round(qty * random.uniform(15.0, 85.0), 2)
        odate = random_date()
        
        # Region mostly matches customer's region, 15% random variation
        cust_reg = str(customer_region_map.get(cid, "East")).strip().title()
        if random.random() < 0.15:
            reg = random.choice(["North", "South", "East", "West"])
        else:
            reg = cust_reg

        rows.append({
            "order_id": oid,
            "customer_id": cid,
            "product_id": pid,
            "store_id": sid,
            "quantity": qty,
            "revenue": rev,
            "order_date": odate,
            "region": reg,
        })

    # ~15 rows reference broken customer_id or product_id
    broken_cust_indices = [10, 35, 62, 95, 120, 165, 210, 245]
    for idx in broken_cust_indices:
        rows[idx]["customer_id"] = f"C9{idx:02d}"

    broken_prod_indices = [25, 50, 85, 140, 185, 230, 275]
    for idx in broken_prod_indices:
        rows[idx]["product_id"] = f"P9{idx:02d}"

    # ~10 rows have negative or null revenue
    null_rev_indices = [15, 72, 133, 198, 255]
    for idx in null_rev_indices:
        rows[idx]["revenue"] = None

    neg_rev_indices = [40, 88, 152, 215, 280]
    for idx in neg_rev_indices:
        rows[idx]["revenue"] = -round(abs(rows[idx]["revenue"] or 49.99), 2)

    # ~5 rows have quantity stored as string like "5 units"
    str_qty_indices = [12, 67, 111, 178, 242]
    qty_phrases = ["5 units", "2 items", "3 pk", "1 unit", "4 pcs"]
    for idx, phrase in zip(str_qty_indices, qty_phrases):
        rows[idx]["quantity"] = phrase

    # 3 exact duplicate order_id rows (duplicate existing records)
    dup_order_indices = [20, 100, 200]
    for idx in dup_order_indices:
        rows.append(dict(rows[idx]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "orders.csv"), index=False)
    return df


def generate_inventory(products_df, stores_df):
    """
    50 rows:
    - columns: product_id, store_id, stock_quantity, last_restock_date
    - last_restock_date entirely null for ~10 rows
    - ~5 rows reference a product_id or store_id that doesn't exist elsewhere (broken keys)
    """
    valid_product_ids = list(products_df["product_id"].unique())
    valid_store_ids = list(stores_df["store_id"].unique())

    rows = []
    for i in range(1, 51):
        pid = random.choice(valid_product_ids)
        sid = random.choice(valid_store_ids)
        qty = random.randint(0, 150)
        r_date = random_date(2023, 2024)
        rows.append({
            "product_id": pid,
            "store_id": sid,
            "stock_quantity": qty,
            "last_restock_date": r_date,
        })

    # ~10 rows with last_restock_date null
    null_date_indices = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
    for idx in null_date_indices:
        rows[idx]["last_restock_date"] = None

    # ~5 rows reference non-existent product_id or store_id
    broken_indices = [7, 16, 23, 35, 42]
    for i, idx in enumerate(broken_indices):
        if i % 2 == 0:
            rows[idx]["product_id"] = f"P99{i}"
        else:
            rows[idx]["store_id"] = f"S9{i}"

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "inventory.csv"), index=False)
    return df


def print_summary(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    df = pd.read_csv(filepath, keep_default_na=True)
    print(f"\n{'='*50}")
    print(f"File: {filepath}")
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Nulls per column:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        print(f"  {col}: {null_count}")
    print(f"{'='*50}")


def main():
    print("Generating sample datasets...")
    prods = generate_products()
    custs = generate_customers()
    stores = generate_stores()
    generate_orders(prods, custs, stores)
    generate_inventory(prods, stores)
    print("Generation complete! Dataset summaries:")

    for fname in ["products.csv", "customers.csv", "stores.csv", "orders.csv", "inventory.csv"]:
        print_summary(fname)


if __name__ == "__main__":
    main()
