#!/usr/bin/env python3
"""
Simple database reset script - no external dependencies
"""

import os
import sqlite3

def reset_database():
    print("WARNING: This will delete ALL data from the database!")
    print("Type 'RESET' to confirm:")
    
    confirmation = input().strip()
    
    if confirmation != 'RESET':
        print("Reset cancelled.")
        return
    
    print("Resetting database...")
    
    # Close any existing connections
    try:
        conn = sqlite3.connect('security_system.db')
        conn.close()
        print("Closed existing database connection")
    except Exception as e:
        print(f"No active connection to close: {e}")
    
    # Delete database file
    if os.path.exists('security_system.db'):
        os.remove('security_system.db')
        print("Database file deleted")
    else:
        print("Database file not found, will create new one")
    
    # Create basic database structure without face_recognition dependencies
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Create tables (simplified version from your database.py)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_type TEXT NOT NULL,
                person_name TEXT,
                confidence REAL,
                screenshot_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_level INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recipient TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration INTEGER,
                file_size INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS screenshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_type TEXT,
                file_size INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_settings (
                id INTEGER PRIMARY KEY,
                owner_email TEXT,
                police_email TEXT,
                smtp_server TEXT,
                smtp_port INTEGER,
                email_user TEXT,
                email_password TEXT,
                enable_email BOOLEAN DEFAULT 1,
                enable_police_alerts BOOLEAN DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                duration REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database schema created successfully")
        
        print("\nDatabase reset completed!")
        print("New empty database created and ready to use.")
        
    except Exception as e:
        print(f"Error during reset: {e}")

if __name__ == "__main__":
    reset_database()