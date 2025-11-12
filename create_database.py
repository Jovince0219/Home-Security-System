# create_database.py
from utils.database import init_db

if __name__ == "__main__":
    print("Creating security system database...")
    init_db()
    print("Database created successfully!")