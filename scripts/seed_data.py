import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_db, init_db
from api.models import User, Product, Order, KnowledgeBase
from core import SecurityManager


def seed_database():
    print("🌱 Seeding database with sample data...")

    init_db()

    db = next(get_db())
    security = SecurityManager()

    try:
        print("\n👤 Creating users...")
        users = seed_users(db, security)

        print("\n📦 Creating products...")
        products = seed_products(db)

        print("\n🛒 Creating orders...")
        orders = seed_orders(db, users, products)

        print("\n📚 Creating knowledge base entries...")
        knowledge = seed_knowledge_base(db, users)

        db.commit()

        print("\n✅ Database seeded successfully!")
        print(f"  - Users: {len(users)}")
        print(f"  - Products: {len(products)}")
        print(f"  - Orders: {len(orders)}")
        print(f"  - Knowledge entries: {len(knowledge)}")

    except Exception as e:
        db.rollback()
        print(f"\n❌ Error seeding database: {str(e)}")
        raise
    finally:
        db.close()


def seed_users(db, security):
    users = []

    existing_users = db.query(User).count()
    if existing_users > 0:
        print(f"  ⚠️  Users already exist ({existing_users}), skipping...")
        return db.query(User).all()

    user_data = [
        {"username": "admin", "email": "admin@chatbi.com", "password": "admin123"},
        {"username": "demo", "email": "demo@chatbi.com", "password": "demo123"},
        {"username": "alice", "email": "alice@example.com", "password": "alice123"},
        {"username": "bob", "email": "bob@example.com", "password": "bob123"},
        {"username": "charlie", "email": "charlie@example.com", "password": "charlie123"}
    ]

    for data in user_data:
        user = User(
            username=data["username"],
            email=data["email"],
            password_hash=security.hash_password(data["password"]),
            api_key=security.generate_api_key(),
            is_active=True
        )
        db.add(user)
        users.append(user)
        print(f"  ✅ Created user: {data['username']}")

    return users


def seed_products(db):
    products = []

    existing_products = db.query(Product).count()
    if existing_products > 0:
        print(f"  ⚠️  Products already exist ({existing_products}), skipping...")
        return db.query(Product).all()

    product_data = [
        {"name": "Laptop Pro", "category": "Electronics", "price": 1299.99, "stock": 50},
        {"name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock": 200},
        {"name": "Mechanical Keyboard", "category": "Electronics", "price": 99.99, "stock": 150},
        {"name": "4K Monitor", "category": "Electronics", "price": 399.99, "stock": 75},
        {"name": "USB-C Hub", "category": "Electronics", "price": 49.99, "stock": 300},
        {"name": "Webcam HD", "category": "Electronics", "price": 79.99, "stock": 120},

        {"name": "Ergonomic Chair", "category": "Furniture", "price": 299.99, "stock": 40},
        {"name": "Standing Desk", "category": "Furniture", "price": 599.99, "stock": 25},
        {"name": "Bookshelf", "category": "Furniture", "price": 149.99, "stock": 60},
        {"name": "Filing Cabinet", "category": "Furniture", "price": 199.99, "stock": 35},

        {"name": "Notebook Set", "category": "Stationery", "price": 14.99, "stock": 500},
        {"name": "Premium Pens", "category": "Stationery", "price": 24.99, "stock": 300},
        {"name": "Sticky Notes", "category": "Stationery", "price": 7.99, "stock": 800},
        {"name": "Planner 2024", "category": "Stationery", "price": 19.99, "stock": 250},
        {"name": "Highlighter Set", "category": "Stationery", "price": 9.99, "stock": 400},

        {"name": "Office Suite", "category": "Software", "price": 149.99, "stock": 999},
        {"name": "Antivirus Pro", "category": "Software", "price": 59.99, "stock": 999},
        {"name": "Cloud Storage", "category": "Software", "price": 9.99, "stock": 999},
        {"name": "Video Editor", "category": "Software", "price": 199.99, "stock": 999},
        {"name": "Password Manager", "category": "Software", "price": 29.99, "stock": 999}
    ]

    for data in product_data:
        product = Product(**data)
        db.add(product)
        products.append(product)
        print(f"  ✅ Created product: {data['name']}")

    return products


def seed_orders(db, users, products):
    orders = []

    existing_orders = db.query(Order).count()
    if existing_orders > 0:
        print(f"  ⚠️  Orders already exist ({existing_orders}), skipping...")
        return db.query(Order).all()

    start_date = datetime.now() - timedelta(days=30)
    statuses = ['completed', 'completed', 'completed', 'pending', 'cancelled']

    for i in range(100):
        user = random.choice(users[1:])
        product = random.choice(products)

        quantity = random.randint(1, 5)
        amount = product.price * quantity

        days_ago = random.randint(0, 30)
        order_date = start_date + timedelta(days=days_ago)

        order = Order(
            user_id=user.id,
            product_id=product.id,
            quantity=quantity,
            amount=amount,
            status=random.choice(statuses),
            created_at=order_date
        )

        db.add(order)
        orders.append(order)

        if (i + 1) % 20 == 0:
            print(f"  ✅ Created {i + 1} orders...")

    print(f"  ✅ Created total {len(orders)} orders")
    return orders


def seed_knowledge_base(db, users):
    knowledge_entries = []

    existing_kb = db.query(KnowledgeBase).count()
    if existing_kb > 0:
        print(f"  ⚠️  Knowledge base already has entries ({existing_kb}), skipping...")
        return db.query(KnowledgeBase).all()

    kb_data = [
        {
            "title": "SQL Query Best Practices",
            "content": """
Best practices for writing SQL queries in ChatBI:
1. Always use SELECT statements for data retrieval
2. Include WHERE clauses to filter data
3. Use JOIN operations for combining tables
4. Apply GROUP BY for aggregations
5. Add ORDER BY for sorted results
6. Use LIMIT to restrict result size
7. Avoid SELECT * in production queries
            """,
            "category": "SQL",
            "tags": ["sql", "best-practices", "query"]
        },
        {
            "title": "Common Data Analysis Patterns",
            "content": """
Common patterns for data analysis:
1. Time series analysis: Track metrics over time
2. Cohort analysis: Group users by characteristics
3. Funnel analysis: Track conversion steps
4. Segmentation: Divide data into meaningful groups
5. Correlation analysis: Find relationships between variables
6. Anomaly detection: Identify outliers
7. Trend analysis: Identify patterns and directions
            """,
            "category": "Analysis",
            "tags": ["analysis", "patterns", "methodology"]
        },
        {
            "title": "Chart Selection Guide",
            "content": """
Choosing the right chart for your data:
- Line Chart: Time series data, trends over time
- Bar Chart: Comparing categories, rankings
- Pie Chart: Parts of a whole, percentages
- Scatter Plot: Correlations, relationships
- Heatmap: Matrix data, intensity visualization
- Box Plot: Statistical distributions, quartiles
- Histogram: Frequency distributions
            """,
            "category": "Visualization",
            "tags": ["charts", "visualization", "guide"]
        },
        {
            "title": "Sales Dashboard Queries",
            "content": """
Useful queries for sales analysis:

1. Total Revenue:
SELECT SUM(amount) as total_revenue FROM orders WHERE status = 'completed'

2. Top Products:
SELECT p.name, COUNT(o.id) as sales_count, SUM(o.amount) as revenue
FROM products p
JOIN orders o ON p.id = o.product_id
GROUP BY p.id ORDER BY revenue DESC LIMIT 10

3. Daily Sales:
SELECT DATE(created_at) as date, COUNT(*) as orders, SUM(amount) as revenue
FROM orders GROUP BY DATE(created_at) ORDER BY date DESC
            """,
            "category": "SQL",
            "tags": ["sales", "queries", "dashboard"]
        },
        {
            "title": "Data Quality Checks",
            "content": """
Essential data quality checks:
1. Null value detection
2. Duplicate record identification
3. Data type validation
4. Range and boundary checks
5. Referential integrity validation
6. Format consistency checks
7. Business rule validation
            """,
            "category": "Analysis",
            "tags": ["data-quality", "validation", "checks"]
        }
    ]

    for data in kb_data:
        entry = KnowledgeBase(
            title=data["title"],
            content=data["content"],
            category=data["category"],
            tags=data["tags"],
            created_by=users[0].id
        )

        db.add(entry)
        knowledge_entries.append(entry)
        print(f"  ✅ Created knowledge: {data['title']}")

    return knowledge_entries


def generate_sample_csv():
    print("\n📄 Generating sample CSV file...")

    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    data = {
        'date': dates,
        'product': [random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Chair']) for _ in range(100)],
        'category': [random.choice(['Electronics', 'Furniture', 'Stationery']) for _ in range(100)],
        'quantity': [random.randint(1, 20) for _ in range(100)],
        'revenue': [round(random.uniform(50, 2000), 2) for _ in range(100)],
        'region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(100)],
        'customer_satisfaction': [round(random.uniform(3.0, 5.0), 1) for _ in range(100)]
    }

    df = pd.DataFrame(data)

    csv_path = Path("data/sample_generated.csv")
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"  ✅ Generated sample CSV with {len(df)} rows")
    print(f"  📍 Saved to: {csv_path}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Seed ChatBI database")
    parser.add_argument("--users-only", action="store_true", help="Only seed users")
    parser.add_argument("--products-only", action="store_true", help="Only seed products")
    parser.add_argument("--csv", action="store_true", help="Generate sample CSV")
    parser.add_argument("--force", action="store_true", help="Force reseed (clear existing data)")

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════╗
║       ChatBI Database Seeder          ║
╚═══════════════════════════════════════╝
    """)

    if args.csv:
        generate_sample_csv()
    else:
        if args.force:
            response = input("⚠️  This will clear existing data. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return

        seed_database()

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
