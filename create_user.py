import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect("security_system.db")
cursor = conn.cursor()

username = "admin"
raw_password = "admin123"

hashed_password = generate_password_hash(raw_password)

cursor.execute(
    "INSERT INTO users (username, password) VALUES (?, ?)",
    (username, hashed_password)
)

conn.commit()
conn.close()
