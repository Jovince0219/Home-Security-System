"""
Fix user settings migration - separates settings for different users
"""
import sqlite3

def fix_user_settings():
    print("Fixing user settings isolation...")
    
    conn = sqlite3.connect('security_system.db')
    cursor = conn.cursor()
    
    try:
        # Get all users
        cursor.execute('SELECT id, username FROM users WHERE role = "admin"')
        users = cursor.fetchall()
        
        print(f"Found {len(users)} admin users")
        
        # Check current twilio_settings
        cursor.execute('SELECT COUNT(*) as count FROM twilio_settings')
        twilio_count = cursor.fetchone()[0]
        print(f"Current twilio_settings records: {twilio_count}")
        
        # If only 1 record exists but multiple users, duplicate it for each user
        if twilio_count == 1 and len(users) > 1:
            print("Duplicating twilio settings for each user...")
            
            # Get the single settings record
            cursor.execute('SELECT * FROM twilio_settings WHERE id = 1')
            settings = cursor.fetchone()
            
            if settings:
                # Delete the single record
                cursor.execute('DELETE FROM twilio_settings')
                
                # Insert a copy for each admin user
                for user_id, username in users:
                    cursor.execute('''
                        INSERT INTO twilio_settings 
                        (user_id, account_sid, auth_token, twilio_number, test_mode, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (user_id, settings[2], settings[3], settings[4], settings[5]))
                    print(f"  Created twilio settings for user {username} (ID: {user_id})")
        
        # Check current alert_settings
        cursor.execute('SELECT COUNT(*) as count FROM alert_settings')
        alert_count = cursor.fetchone()[0]
        print(f"Current alert_settings records: {alert_count}")
        
        # If only 1 record exists but multiple users, duplicate it for each user
        if alert_count == 1 and len(users) > 1:
            print("Duplicating alert settings for each user...")
            
            # Get the single settings record
            cursor.execute('SELECT * FROM alert_settings WHERE id = 1')
            settings = cursor.fetchone()
            
            if settings:
                # Delete the single record
                cursor.execute('DELETE FROM alert_settings')
                
                # Insert a copy for each admin user
                for user_id, username in users:
                    cursor.execute('''
                        INSERT INTO alert_settings 
                        (user_id, owner_email, police_email, smtp_server, smtp_port, 
                         email_user, email_password, enable_email, enable_police_alerts, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (user_id, settings[2], settings[3], settings[4], settings[5], 
                          settings[6], settings[7], settings[8], settings[9]))
                    print(f"  Created alert settings for user {username} (ID: {user_id})")
        
        # Verify each user has their own settings
        print("\nVerifying settings per user...")
        for user_id, username in users:
            # Check twilio settings
            cursor.execute('SELECT COUNT(*) FROM twilio_settings WHERE user_id = ?', (user_id,))
            twilio_exists = cursor.fetchone()[0]
            
            # Check alert settings
            cursor.execute('SELECT COUNT(*) FROM alert_settings WHERE user_id = ?', (user_id,))
            alert_exists = cursor.fetchone()[0]
            
            print(f"User {username} (ID: {user_id}):")
            print(f"  - Twilio settings: {'✓' if twilio_exists else '✗'}")
            print(f"  - Alert settings: {'✓' if alert_exists else '✗'}")
            
            # If missing settings, create empty records
            if not twilio_exists:
                cursor.execute('''
                    INSERT INTO twilio_settings (user_id, test_mode)
                    VALUES (?, 1)
                ''', (user_id,))
                print(f"  Created empty twilio settings for {username}")
            
            if not alert_exists:
                cursor.execute('''
                    INSERT INTO alert_settings (user_id, enable_email, enable_police_alerts)
                    VALUES (?, 1, 0)
                ''', (user_id,))
                print(f"  Created empty alert settings for {username}")
        
        conn.commit()
        print("\n✅ Settings isolation fixed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_user_settings()