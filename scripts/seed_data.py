#!/usr/bin/env python3
"""
Data seeding script for ChatBI platform
Populates database with sample data for testing and demonstration
"""

import os
import sys
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import pymysql
from faker import Faker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)
fake = Faker()


class DataSeeder:
    """Handle data seeding operations"""

    def __init__(self, config: Config):
        self.config = config
        self.connection = None

    def connect(self):
        """Connect to database"""
        try:
            self.connection = pymysql.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                charset='utf8mb4'
            )
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def seed_users(self):
        """Create default users with different roles"""
        users = [
            {
                'username': 'admin',
                'email': 'admin@chatbi.com',
                'password': 'admin123',  # In production, use strong passwords
                'role': 'admin'
            },
            {
                'username': 'analyst',
                'email': 'analyst@chatbi.com',
                'password': 'analyst123',
                'role': 'analyst'
            },
            {
                'username': 'viewer',
                'email': 'viewer@chatbi.com',
                'password': 'viewer123',
                'role': 'viewer'
            },
            {
                'username': 'demo',
                'email': 'demo@chatbi.com',
                'password': 'demo123',
                'role': 'guest'
            }
        ]

        try:
            with self.connection.cursor() as cursor:
                for user in users:
                    # Check if user exists
                    cursor.execute(
                        "SELECT id FROM users WHERE username = %s",
                        (user['username'],)
                    )

                    if cursor.fetchone():
                        logger.info(f"User {user['username']} already exists, skipping")
                        continue

                    # Insert user
                    cursor.execute(
                        """INSERT INTO users (username, email, password_hash, role) 
                           VALUES (%s, %s, %s, %s)""",
                        (user['username'], user['email'],
                         self.hash_password(user['password']), user['role'])
                    )

                    logger.info(f"Created user: {user['username']} ({user['role']})")

                self.connection.commit()
                logger.info("Users seeding completed")

        except Exception as e:
            logger.error(f"Failed to seed users: {e}")
            raise

    def seed_customers(self, count: int = 100):
        """Generate sample customer data"""
        try:
            with self.connection.cursor() as cursor:
                # Check if customers already exist
                cursor.execute("SELECT COUNT(*) FROM customers")
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    logger.info(f"Found {existing_count} existing customers, skipping customer seeding")
                    return

                customers = []
                for _ in range(count):
                    customer = (
                        fake.name(),
                        fake.email(),
                        fake.phone_number()[:20],  # Limit phone length
                        fake.address(),
                        fake.city(),
                        fake.country(),
                        fake.date_between(start_date='-2y', end_date='today'),
                        random.choice(['active', 'inactive', 'suspended'])
                    )
                    customers.append(customer)

                cursor.executemany(
                    """INSERT INTO customers (name, email, phone, address, city, country, 
                                           registration_date, status) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    customers
                )

                self.connection.commit()
                logger.info(f"Created {count} sample customers")

        except Exception as e:
            logger.error(f"Failed to seed customers: {e}")
            raise

    def seed_products(self, count: int = 50):
        """Generate sample product data"""
        try:
            with self.connection.cursor() as cursor:
                # Check if products already exist
                cursor.execute("SELECT COUNT(*) FROM products")
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    logger.info(f"Found {existing_count} existing products, skipping product seeding")
                    return

                categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books',
                              'Sports', 'Toys', 'Beauty', 'Automotive']

                products = []
                for _ in range(count):
                    cost = round(random.uniform(10, 500), 2)
                    price = round(cost * random.uniform(1.2, 3.0), 2)  # 20-200% markup

                    product = (
                        fake.catch_phrase(),  # Product name
                        random.choice(categories),
                        price,
                        cost,
                        random.randint(0, 1000)  # Stock quantity
                    )
                    products.append(product)

                cursor.executemany(
                    """INSERT INTO products (name, category, price, cost, stock_quantity) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    products
                )

                self.connection.commit()
                logger.info(f"Created {count} sample products")

        except Exception as e:
            logger.error(f"Failed to seed products: {e}")
            raise

    def seed_orders(self, count: int = 200):
        """Generate sample order data"""
        try:
            with self.connection.cursor() as cursor:
                # Check if orders already exist
                cursor.execute("SELECT COUNT(*) FROM orders")
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    logger.info(f"Found {existing_count} existing orders, skipping order seeding")
                    return

                # Get customer and product IDs
                cursor.execute("SELECT id FROM customers")
                customer_ids = [row[0] for row in cursor.fetchall()]

                cursor.execute("SELECT id, price FROM products")
                products = cursor.fetchall()

                if not customer_ids or not products:
                    logger.warning("No customers or products found, cannot create orders")
                    return

                # Create orders
                orders = []
                order_items = []

                for _ in range(count):
                    customer_id = random.choice(customer_ids)
                    order_date = fake.date_between(start_date='-1y', end_date='today')
                    status = random.choice(['pending', 'processing', 'shipped', 'delivered', 'cancelled'])

                    # Calculate total amount based on order items
                    num_items = random.randint(1, 5)
                    selected_products = random.sample(products, min(num_items, len(products)))
                    total_amount = 0

                    order_id = len(orders) + 1  # Temporary ID for linking

                    for product_id, price in selected_products:
                        quantity = random.randint(1, 3)
                        unit_price = price
                        total_amount += quantity * unit_price

                        order_items.append((
                            order_id,  # Will be updated with actual order ID
                            product_id,
                            quantity,
                            unit_price
                        ))

                    orders.append((
                        customer_id,
                        order_date,
                        round(total_amount, 2),
                        status,
                        fake.address()  # Shipping address
                    ))

                # Insert orders
                cursor.executemany(
                    """INSERT INTO orders (customer_id, order_date, total_amount, status, shipping_address) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    orders
                )

                # Get actual order IDs
                cursor.execute("SELECT id FROM orders ORDER BY id DESC LIMIT %s", (count,))
                actual_order_ids = [row[0] for row in cursor.fetchall()]
                actual_order_ids.reverse()  # Match insertion order

                # Update order items with actual order IDs
                updated_order_items = []
                item_index = 0

                for i, order_id in enumerate(actual_order_ids):
                    while item_index < len(order_items) and order_items[item_index][0] == i + 1:
                        updated_order_items.append((
                            order_id,  # Actual order ID
                            order_items[item_index][1],  # product_id
                            order_items[item_index][2],  # quantity
                            order_items[item_index][3]  # unit_price
                        ))
                        item_index += 1

                # Insert order items
                cursor.executemany(
                    """INSERT INTO order_items (order_id, product_id, quantity, unit_price) 
                       VALUES (%s, %s, %s, %s)""",
                    updated_order_items
                )

                self.connection.commit()
                logger.info(f"Created {count} sample orders with {len(updated_order_items)} order items")

        except Exception as e:
            logger.error(f"Failed to seed orders: {e}")
            raise

    def seed_chat_sessions(self):
        """Create sample chat sessions"""
        try:
            with self.connection.cursor() as cursor:
                # Check if chat sessions already exist
                cursor.execute("SELECT COUNT(*) FROM chat_sessions")
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    logger.info(f"Found {existing_count} existing chat sessions, skipping")
                    return

                # Get user IDs
                cursor.execute("SELECT id, username FROM users")
                users = cursor.fetchall()

                sample_sessions = [
                    {
                        'title': 'Sales Analysis Q4 2024',
                        'messages': [
                            ('user', 'Show me total sales for Q4 2024'),
                            ('assistant', 'I\'ll analyze the Q4 2024 sales data for you.'),
                            ('user', 'What were the top 5 products by revenue?'),
                            ('assistant', 'Based on the data, here are the top 5 products by revenue in Q4 2024...')
                        ]
                    },
                    {
                        'title': 'Customer Demographics',
                        'messages': [
                            ('user', 'Can you analyze our customer demographics?'),
                            ('assistant', 'I\'ll provide a comprehensive analysis of your customer demographics.'),
                            ('user', 'Which cities have the most customers?'),
                            ('assistant', 'Here\'s the breakdown of customers by city...')
                        ]
                    },
                    {
                        'title': 'Monthly Revenue Trends',
                        'messages': [
                            ('user', 'Show me monthly revenue trends for this year'),
                            ('assistant', 'I\'ll create a visualization of monthly revenue trends.'),
                        ]
                    }
                ]

                for user_id, username in users:
                    if username in ['admin', 'analyst']:  # Only create sessions for active users
                        for session_data in sample_sessions:
                            session_id = str(uuid.uuid4())

                            # Insert chat session
                            cursor.execute(
                                """INSERT INTO chat_sessions (id, user_id, title) 
                                   VALUES (%s, %s, %s)""",
                                (session_id, user_id, session_data['title'])
                            )

                            # Insert messages
                            for message_type, content in session_data['messages']:
                                cursor.execute(
                                    """INSERT INTO chat_messages (session_id, user_id, message_type, content) 
                                       VALUES (%s, %s, %s, %s)""",
                                    (session_id, user_id, message_type, content)
                                )

                self.connection.commit()
                logger.info("Created sample chat sessions")

        except Exception as e:
            logger.error(f"Failed to seed chat sessions: {e}")
            raise

    def seed_query_logs(self):
        """Create sample query logs for analytics"""
        try:
            with self.connection.cursor() as cursor:
                # Check if query logs already exist
                cursor.execute("SELECT COUNT(*) FROM query_logs")
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    logger.info(f"Found {existing_count} existing query logs, skipping")
                    return

                # Get user IDs
                cursor.execute("SELECT id FROM users WHERE role IN ('admin', 'analyst')")
                user_ids = [row[0] for row in cursor.fetchall()]

                if not user_ids:
                    logger.warning("No admin/analyst users found for query logs")
                    return

                sample_queries = [
                    {
                        'query_text': 'Show me sales by month',
                        'executed_sql': 'SELECT MONTH(order_date) as month, SUM(total_amount) as sales FROM orders GROUP BY MONTH(order_date)',
                        'query_type': 'natural',
                        'execution_time': 0.25,
                        'row_count': 12,
                        'status': 'success'
                    },
                    {
                        'query_text': 'Top customers by revenue',
                        'executed_sql': 'SELECT c.name, SUM(o.total_amount) as revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY revenue DESC LIMIT 10',
                        'query_type': 'natural',
                        'execution_time': 0.45,
                        'row_count': 10,
                        'status': 'success'
                    },
                    {
                        'query_text': 'SELECT * FROM users',
                        'query_type': 'sql',
                        'execution_time': None,
                        'row_count': None,
                        'status': 'blocked',
                        'error_message': 'Access to users table is not permitted'
                    }
                ]

                logs = []
                for _ in range(50):  # Create 50 sample logs
                    query = random.choice(sample_queries)
                    user_id = random.choice(user_ids)
                    created_at = fake.date_time_between(start_date='-30d', end_date='now')

                    logs.append((
                        user_id,
                        None,  # session_id
                        query['query_text'],
                        query['query_type'],
                        query.get('executed_sql'),
                        query.get('execution_time'),
                        query.get('row_count'),
                        query['status'],
                        query.get('error_message'),
                        created_at
                    ))

                cursor.executemany(
                    """INSERT INTO query_logs (user_id, session_id, query_text, query_type, 
                                             executed_sql, execution_time, row_count, status, 
                                             error_message, created_at) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    logs
                )

                self.connection.commit()
                logger.info(f"Created {len(logs)} sample query logs")

        except Exception as e:
            logger.error(f"Failed to seed query logs: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


def create_sample_csv():
    """Create sample CSV file for data upload testing"""
    try:
        data_dir = project_root / 'data'
        data_dir.mkdir(exist_ok=True)

        csv_file = data_dir / 'sample.csv'

        if csv_file.exists():
            logger.info("Sample CSV already exists")
            return

        # Generate sample data
        with open(csv_file, 'w') as f:
            f.write("date,product,category,sales,units,region\n")

            products = ['ProductA', 'ProductB', 'ProductC', 'ProductD', 'ProductE']
            categories = ['Electronics', 'Clothing', 'Home']
            regions = ['North', 'South', 'East', 'West']

            for i in range(100):
                date = fake.date_between(start_date='-1y', end_date='today')
                product = random.choice(products)
                category = random.choice(categories)
                sales = round(random.uniform(100, 5000), 2)
                units = random.randint(1, 50)
                region = random.choice(regions)

                f.write(f"{date},{product},{category},{sales},{units},{region}\n")

        logger.info(f"Created sample CSV file: {csv_file}")

    except Exception as e:
        logger.error(f"Failed to create sample CSV: {e}")


def main():
    """Main seeding function"""
    logger.info("Starting data seeding...")

    try:
        config = Config()
        seeder = DataSeeder(config)
        seeder.connect()

        # Seed all data
        seeder.seed_users()
        seeder.seed_customers(100)
        seeder.seed_products(50)
        seeder.seed_orders(200)
        seeder.seed_chat_sessions()
        seeder.seed_query_logs()

        seeder.close()

        # Create sample files
        create_sample_csv()

        logger.info("✅ Data seeding completed successfully!")
        logger.info("\nDefault user accounts created:")
        logger.info("- admin/admin123 (Administrator)")
        logger.info("- analyst/analyst123 (Data Analyst)")
        logger.info("- viewer/viewer123 (Viewer)")
        logger.info("- demo/demo123 (Guest)")
        logger.info("\n100 customers, 50 products, and 200 orders generated")
        logger.info("Sample chat sessions and query logs created")

    except Exception as e:
        logger.error(f"Data seeding failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()