import argparse
import sys
import traceback
from pathlib import Path


def setup_database():
    print("\n🗄️  Setting up database...")

    try:
        sys.path.insert(0, str(Path.cwd()))
        from api.database import init_db

        init_db()
        print("  ✅ Database initialized")

        response = input("\n  Seed database with sample data? (y/n): ")
        if response.lower() == 'y':
            from scripts.seed_data import seed_database
            seed_database()
            print("  ✅ Database seeded")

    except Exception as e:
        print(f"  ❌ Database setup failed: {str(e)}")
        print("  Please ensure database is running and configured correctly")
        raise Exception(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Setup ChatBI environment")
    parser.add_argument("--skip-db", action="store_true", help="Skip database setup")
    parser.add_argument("--env-only", action="store_true", help="Only setup Python environment")

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════╗
║       ChatBI Setup Script             ║
║       Data Intelligence Platform      ║
╚═══════════════════════════════════════╝
    """)

    setup_database()

    print("""
╔═══════════════════════════════════════╗
║       Setup Complete! 🎉              ║
╚═══════════════════════════════════════╝

Next steps:
1. Update .env with your configuration
2. Run: python scripts/run.py
3. Open: http://localhost:8501 (Streamlit)
        http://localhost:8000/docs (API)
    """)


if __name__ == "__main__":
    main()
