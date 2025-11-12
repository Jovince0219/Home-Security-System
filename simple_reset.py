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
        
        # ========== NEW TWILIO TABLES ==========
        
        # Twilio settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS twilio_settings (
                id INTEGER PRIMARY KEY,
                account_sid TEXT,
                auth_token TEXT,
                twilio_number TEXT,
                primary_number TEXT,
                backup_number TEXT,
                test_mode BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alert events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_events (
                event_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trigger_type TEXT,
                recording_filepath TEXT,
                review_status TEXT DEFAULT 'pending',
                call_status TEXT DEFAULT 'pending',
                completed_at DATETIME
            )
        ''')
        
        # Call attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS call_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                attempt_number INTEGER,
                timestamp DATETIME,
                phone_number TEXT,
                call_sid TEXT,
                status TEXT,
                FOREIGN KEY (event_id) REFERENCES alert_events (event_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database schema created successfully")
        
        # Reset face tracking in memory (if system is running)
        reset_face_tracking()
        
        print("\nDatabase reset completed!")
        print("✅ Database tables created")
        print("✅ Twilio tables added")
        print("✅ Face tracking memory cleared")
        print("✅ New empty database ready to use")
        
    except Exception as e:
        print(f"Error during reset: {e}")

def reset_face_tracking():
    """Reset the in-memory face tracking without stopping the application"""
    try:
        # Try to import and reset the Twilio alert system
        import sys
        import importlib.util
        
        # Check if the module is loaded
        if 'utils.twilio_alert_system' in sys.modules:
            twilio_module = sys.modules['utils.twilio_alert_system']
            if hasattr(twilio_module, 'twilio_alert_system'):
                twilio_alert_system.twilio_alert_system.alerted_faces = {}
                print("✅ Face tracking memory cleared")
            else:
                print("ℹ️  Twilio alert system not initialized yet")
        else:
            print("ℹ️  Twilio module not loaded - face tracking will reset on next startup")
            
    except Exception as e:
        print(f"⚠️  Could not reset face tracking: {e}")
        print("ℹ️  Face tracking will reset when application restarts")

def reset_twilio_settings_only():
    """Reset only Twilio settings and face tracking without deleting entire database"""
    print("Resetting Twilio settings and face tracking only...")
    
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Reset Twilio settings table
        cursor.execute('DELETE FROM twilio_settings')
        cursor.execute('DELETE FROM alert_events')
        cursor.execute('DELETE FROM call_attempts')
        
        # Reset alert events and call attempts
        cursor.execute('DELETE FROM alert_events')
        cursor.execute('DELETE FROM call_attempts')
        
        conn.commit()
        conn.close()
        
        # Reset face tracking memory
        reset_face_tracking()
        
        print("✅ Twilio settings reset")
        print("✅ Alert events cleared")
        print("✅ Call attempts cleared")
        print("✅ Face tracking memory cleared")
        
    except Exception as e:
        print(f"Error resetting Twilio settings: {e}")

def show_database_stats():
    """Show current database statistics"""
    try:
        if not os.path.exists('security_system.db'):
            print("Database file does not exist")
            return
            
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Get table counts
        tables = [
            'faces', 'detections', 'alerts', 'recordings', 'screenshots',
            'alert_settings', 'audio_recordings', 'twilio_settings', 
            'alert_events', 'call_attempts'
        ]
        
        print("\n📊 Database Statistics:")
        print("-" * 40)
        
        for table in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = cursor.fetchone()[0]
                print(f"{table:20} : {count:4} records")
            except:
                print(f"{table:20} : Table not found")
        
        conn.close()
        
    except Exception as e:
        print(f"Error getting database stats: {e}")

if __name__ == "__main__":
    print("Security System Database Reset Tool")
    print("=" * 40)
    print("1. Full database reset (DELETE ALL DATA)")
    print("2. Reset Twilio settings only")
    print("3. Show database statistics")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == '1':
        reset_database()
    elif choice == '2':
        reset_twilio_settings_only()
    elif choice == '3':
        show_database_stats()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice")