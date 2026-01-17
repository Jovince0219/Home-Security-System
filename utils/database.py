import sqlite3
import os
import datetime

DATABASE = 'security_system.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with support for faceless human detections and known persons"""
    conn = sqlite3.connect('security_system.db')
    c = conn.cursor()
    
    # Create users table - ADD owner_id field
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'viewer',  -- 'admin' or 'viewer'
            face_id INTEGER,             -- Link to faces table
            owner_id INTEGER,            -- Link to admin who created this account
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (face_id) REFERENCES faces (id) ON DELETE SET NULL,
            FOREIGN KEY (owner_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create faces table (Authorized Persons) - ALREADY HAS user_id
    conn.execute('''    
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            image_path TEXT NOT NULL,
            relationship TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create known_persons table (Non-threatening but not authorized) - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS known_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detection_id INTEGER,
            registered_from_false_alarm BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (detection_id) REFERENCES detections (id) ON DELETE SET NULL
        )
    ''')
    
    # Create detections table - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_type TEXT NOT NULL,
            person_name TEXT,
            confidence REAL,
            screenshot_path TEXT,
            alert_level INTEGER DEFAULT 1,
            timestamp TEXT,  -- Changed to TEXT for consistent formatting
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create alerts table - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recipient TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create phone_numbers table for multiple contacts - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS phone_numbers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            phone_number TEXT NOT NULL,
            display_name TEXT,
            is_active BOOLEAN DEFAULT 1,
            sort_order INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create recordings table - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration INTEGER,
            file_size INTEGER DEFAULT 0,
            detection_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (detection_id) REFERENCES detections (id) ON DELETE SET NULL
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            description TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            detection_type TEXT,
            file_size INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create alert_settings table - UPDATED WITH user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS alert_settings (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            owner_email TEXT,
            police_email TEXT,
            smtp_server TEXT,
            smtp_port INTEGER,
            email_user TEXT,
            email_password TEXT,
            enable_email BOOLEAN DEFAULT 1,
            enable_police_alerts BOOLEAN DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id)
        )
    ''')
    
    # Create audio_recordings table - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS audio_recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            duration REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Twilio settings table - UPDATED WITH user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS twilio_settings (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            account_sid TEXT,
            auth_token TEXT,
            twilio_number TEXT,
            test_mode BOOLEAN DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id)
        )
    ''')
    
    # Alert events table - ALREADY HAS user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS alert_events (
            event_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            trigger_type TEXT,
            recording_filepath TEXT,
            review_status TEXT DEFAULT 'pending',
            call_status TEXT DEFAULT 'pending',
            person_name TEXT,
            confidence REAL,
            distance_meters REAL,
            is_authorized BOOLEAN DEFAULT 0,
            is_known_person BOOLEAN DEFAULT 0,
            completed_at DATETIME,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Call attempts table - UPDATED WITH user_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS call_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_id TEXT,
            attempt_number INTEGER,
            timestamp DATETIME,
            phone_number TEXT,
            call_sid TEXT,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (event_id) REFERENCES alert_events (event_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
def add_face(user_id, name, encoding, image_path, relationship=""):
    """Add a face with relationship field"""
    conn = get_db_connection()
    encoding_str = ','.join(map(str, encoding))
    conn.execute(
        'INSERT INTO faces (user_id, name, encoding, image_path, relationship) VALUES (?, ?, ?, ?, ?)',
        (user_id, name, encoding_str, image_path, relationship)
    )
    conn.commit()
    conn.close()

def get_all_faces(user_id):
    """Get all registered faces for a user"""
    conn = get_db_connection()
    faces = conn.execute('SELECT * FROM faces WHERE user_id = ? ORDER BY created_at DESC', (user_id,)).fetchall()
    conn.close()
    return faces

def delete_face(user_id, face_id):
    conn = get_db_connection()
    face = conn.execute('SELECT * FROM faces WHERE id = ? AND user_id = ?', (face_id, user_id)).fetchone()
    if face:
        # Delete image file
        if os.path.exists(face['image_path']):
            os.remove(face['image_path'])
        # Delete from database
        conn.execute('DELETE FROM faces WHERE id = ? AND user_id = ?', (face_id, user_id))
        conn.commit()
    conn.close()

def update_face_info(user_id, face_id, new_name=None, new_relationship=None):
    """Update face name and/or relationship"""
    conn = get_db_connection()
    
    if new_name and new_relationship:
        cursor = conn.execute(
            'UPDATE faces SET name = ?, relationship = ? WHERE id = ? AND user_id = ?', 
            (new_name, new_relationship, face_id, user_id)
        )
    elif new_name:
        cursor = conn.execute(
            'UPDATE faces SET name = ? WHERE id = ? AND user_id = ?', 
            (new_name, face_id, user_id)
        )
    elif new_relationship:
        cursor = conn.execute(
            'UPDATE faces SET relationship = ? WHERE id = ? AND user_id = ?', 
            (new_relationship, face_id, user_id)
        )
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def search_faces_by_name(user_id, query):
    """Search faces by name"""
    conn = get_db_connection()
    if query:
        faces = conn.execute(
            'SELECT * FROM faces WHERE user_id = ? AND name LIKE ? ORDER BY created_at DESC', 
            (user_id, f'%{query}%')
        ).fetchall()
    else:
        faces = conn.execute('SELECT * FROM faces WHERE user_id = ? ORDER BY created_at DESC', (user_id,)).fetchall()
    conn.close()
    return faces

def add_detection(user_id, detection_type, person_name=None, confidence=None, screenshot_path=None, alert_level=1):
    """Add detection with proper confidence handling and correct timestamp"""
    conn = get_db_connection()
    
    # Ensure confidence is a float and handle None values
    if confidence is not None:
        confidence = float(confidence)
    else:
        # Set default confidence based on detection type
        if detection_type == 'face_detection':
            confidence = 0.0  # Default for face detection
        elif detection_type == 'motion_detection':
            confidence = 0.5  # Default for motion detection
        else:
            confidence = 0.0
    
    # Use current datetime from Python (matches your system time)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"📝 DEBUG: Adding detection - User: {user_id}, Type: {detection_type}, Name: {person_name}, Confidence: {confidence}, Time: {current_time}")
    
    conn.execute(
        'INSERT INTO detections (user_id, detection_type, person_name, confidence, screenshot_path, alert_level, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, detection_type, person_name, confidence, screenshot_path, alert_level, current_time)
    )
    conn.commit()
    conn.close()

def get_recent_detections(user_id, limit=20):
    """Get recent detections with proper filtering"""
    conn = sqlite3.connect('security_system.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get all recent detections, we'll filter in the API
    c.execute('''
        SELECT * FROM detections 
        WHERE user_id = ?
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (user_id, limit * 2))  # Get extra to account for filtering
    
    detections = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return detections

def save_alert_settings(user_id, settings):
    """Enhanced alert settings persistence with better error handling"""
    conn = get_db_connection()
    
    try:
        conn.execute('BEGIN TRANSACTION')
        
        # Check if settings exist
        existing = conn.execute('SELECT id FROM alert_settings WHERE user_id = ?', (user_id,)).fetchone()
        
        if existing:
            # UPDATE existing settings
            conn.execute('''
                UPDATE alert_settings SET 
                owner_email = ?, police_email = ?, smtp_server = ?, smtp_port = ?,
                email_user = ?, email_password = ?, enable_email = ?, enable_police_alerts = ?,
                updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (
                settings['owner_email'], 
                settings.get('police_email', 'police@example.com'), 
                settings['smtp_server'],
                int(settings['smtp_port']), 
                settings['email_user'], 
                settings.get('email_password', ''),
                1 if settings['enable_email'] else 0,
                1 if settings['enable_police_alerts'] else 0,
                user_id
            ))
        else:
            # INSERT new settings
            conn.execute('''
                INSERT INTO alert_settings 
                (user_id, owner_email, police_email, smtp_server, smtp_port, email_user, email_password, enable_email, enable_police_alerts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                settings['owner_email'], 
                settings.get('police_email', 'police@example.com'), 
                settings['smtp_server'],
                int(settings['smtp_port']), 
                settings['email_user'], 
                settings.get('email_password', ''),
                1 if settings['enable_email'] else 0,
                1 if settings['enable_police_alerts'] else 0
            ))
        
        conn.commit()
        
        # Verify the save was successful
        verification = conn.execute('SELECT * FROM alert_settings WHERE user_id = ?', (user_id,)).fetchone()
        if not verification:
            raise Exception("Settings verification failed after save")
        
        print(f"Alert settings saved successfully for user {user_id}: enable_email={verification['enable_email']}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error saving alert settings for user {user_id}: {e}")
        raise e
    finally:
        conn.close()
        
def get_all_phone_numbers(user_id):
    """Get all phone numbers with sorting - UPDATED with user_id filter"""
    conn = get_db_connection()
    numbers = conn.execute('''
        SELECT * FROM phone_numbers 
        WHERE user_id = ?
        ORDER BY sort_order ASC, created_at ASC
    ''', (user_id,)).fetchall()
    conn.close()
    return numbers

def add_phone_number(user_id, phone_number, display_name=None, is_active=True):
    """Add a new phone number"""
    conn = get_db_connection()
    
    # Get the next sort order
    max_order = conn.execute('SELECT MAX(sort_order) FROM phone_numbers WHERE user_id = ?', (user_id,)).fetchone()[0] or 0
    
    conn.execute(
        'INSERT INTO phone_numbers (user_id, phone_number, display_name, is_active, sort_order) VALUES (?, ?, ?, ?, ?)',
        (user_id, phone_number, display_name, 1 if is_active else 0, max_order + 1)
    )
    conn.commit()
    conn.close()

def update_phone_number_order(user_id, phone_numbers):
    """Update the sort order of phone numbers"""
    conn = get_db_connection()
    
    for index, phone_id in enumerate(phone_numbers):
        conn.execute(
            'UPDATE phone_numbers SET sort_order = ? WHERE id = ? AND user_id = ?',
            (index, phone_id, user_id)
        )
    
    conn.commit()
    conn.close()

def toggle_phone_number_active(user_id, phone_id, is_active):
    """Toggle phone number active status"""
    conn = get_db_connection()
    conn.execute(
        'UPDATE phone_numbers SET is_active = ? WHERE id = ? AND user_id = ?',
        (1 if is_active else 0, phone_id, user_id)
    )
    conn.commit()
    conn.close()

def delete_phone_number(user_id, phone_id):
    """Delete a phone number"""
    conn = get_db_connection()
    conn.execute('DELETE FROM phone_numbers WHERE id = ? AND user_id = ?', (phone_id, user_id))
    conn.commit()
    conn.close()

def update_phone_number(user_id, phone_id, phone_number, display_name):
    """Update phone number details"""
    conn = get_db_connection()
    conn.execute(
        'UPDATE phone_numbers SET phone_number = ?, display_name = ? WHERE id = ? AND user_id = ?',
        (phone_number, display_name, phone_id, user_id)
    )
    conn.commit()
    conn.close()

def get_alert_settings(user_id):
    """Enhanced alert settings retrieval with better error handling"""
    conn = get_db_connection()
    try:
        settings = conn.execute('SELECT * FROM alert_settings WHERE user_id = ?', (user_id,)).fetchone()
        
        if settings:
            settings_dict = dict(settings)
            # Ensure boolean fields are properly converted
            settings_dict['enable_email'] = bool(settings_dict['enable_email'])
            settings_dict['enable_police_alerts'] = bool(settings_dict['enable_police_alerts'])
            return settings_dict
        else:
            return None
            
    except Exception as e:
        print(f"Error retrieving alert settings: {e}")
        return None
    finally:
        conn.close()

def add_alert(user_id, alert_type, message, recipient=None):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO alerts (user_id, alert_type, message, recipient) VALUES (?, ?, ?, ?)',
        (user_id, alert_type, message, recipient)
    )
    conn.commit()
    conn.close()

def get_alert_history(user_id, limit=50):
    """Get alert history - UPDATED with user_id filter"""
    conn = get_db_connection()
    alerts = conn.execute(
        'SELECT * FROM alerts WHERE user_id = ? ORDER BY sent_at DESC LIMIT ?', (user_id, limit)
    ).fetchall()
    conn.close()
    return alerts

def get_all_recordings(user_id):
    """Get all recordings for a user"""
    conn = get_db_connection()
    recordings = conn.execute('SELECT * FROM recordings WHERE user_id = ? ORDER BY start_time DESC', (user_id,)).fetchall()
    conn.close()
    return recordings

def delete_recording(user_id, recording_id):
    """Delete a recording"""
    conn = get_db_connection()
    recording = conn.execute('SELECT * FROM recordings WHERE id = ? AND user_id = ?', (recording_id, user_id)).fetchone()
    if recording:
        # Delete file
        if os.path.exists(recording['file_path']):
            os.remove(recording['file_path'])
        # Delete from database
        conn.execute('DELETE FROM recordings WHERE id = ? AND user_id = ?', (recording_id, user_id))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def add_screenshot(user_id, file_path, description=None, file_size=0, detection_type=None):
    """Add a screenshot to the database"""
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO screenshots (user_id, file_path, description, file_size, detection_type) VALUES (?, ?, ?, ?, ?)',
        (user_id, file_path, description, file_size, detection_type)
    )
    conn.commit()
    conn.close()

def get_all_screenshots(user_id):
    """Get all screenshots for a user"""
    conn = get_db_connection()
    screenshots = conn.execute('SELECT * FROM screenshots WHERE user_id = ? ORDER BY timestamp DESC', (user_id,)).fetchall()
    conn.close()
    return screenshots

def delete_screenshot(user_id, screenshot_id):
    """Delete a screenshot"""
    conn = get_db_connection()
    screenshot = conn.execute('SELECT * FROM screenshots WHERE id = ? AND user_id = ?', (screenshot_id, user_id)).fetchone()
    if screenshot:
        # Delete file
        if os.path.exists(screenshot['file_path']):
            os.remove(screenshot['file_path'])
        # Delete from database
        conn.execute('DELETE FROM screenshots WHERE id = ? AND user_id = ?', (screenshot_id, user_id))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def cleanup_old_recordings(user_id, days):
    """Clean up recordings older than specified days"""
    conn = get_db_connection()
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    conn.execute('DELETE FROM recordings WHERE user_id = ? AND start_time < ?', (user_id, cutoff_date))
    conn.commit()
    conn.close()

def cleanup_old_screenshots(user_id, days):
    """Clean up screenshots older than specified days"""
    conn = get_db_connection()
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    conn.execute('DELETE FROM screenshots WHERE user_id = ? AND timestamp < ?', (user_id, cutoff_date))
    conn.commit()
    conn.close()

def add_audio_recording(user_id, file_path, duration, file_size=0):
    """Add an audio recording to the database"""
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO audio_recordings (user_id, file_path, duration, file_size) VALUES (?, ?, ?, ?)',
        (user_id, file_path, duration, file_size)
    )
    conn.commit()
    conn.close()

def get_audio_recordings(user_id):
    """Get all audio recordings for a user"""
    conn = get_db_connection()
    recordings = conn.execute('SELECT * FROM audio_recordings WHERE user_id = ? ORDER BY timestamp DESC', (user_id,)).fetchall()
    conn.close()
    return recordings

def delete_audio_recording(user_id, recording_id):
    """Delete an audio recording"""
    conn = get_db_connection()
    recording = conn.execute('SELECT * FROM audio_recordings WHERE id = ? AND user_id = ?', (recording_id, user_id)).fetchone()
    if recording:
        # Delete file
        if os.path.exists(recording['file_path']):
            os.remove(recording['file_path'])
        # Delete from database
        conn.execute('DELETE FROM audio_recordings WHERE id = ? AND user_id = ?', (recording_id, user_id))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def add_face_image(user_id, name, encoding, image_path):
    """Add a face image to the database and return the face ID"""
    conn = get_db_connection()
    cursor = conn.execute(
        'INSERT INTO faces (user_id, name, encoding, image_path) VALUES (?, ?, ?, ?)',
        (user_id, name, encoding, image_path)
    )
    face_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return face_id

def get_face_by_id(user_id, face_id):
    """Get a specific face by ID"""
    conn = get_db_connection()
    face = conn.execute('SELECT * FROM faces WHERE id = ? AND user_id = ?', (face_id, user_id)).fetchone()
    conn.close()
    return face

def get_face_images(user_id, face_id):
    """Get all images for a specific face ID"""
    conn = get_db_connection()
    images = conn.execute(
        'SELECT * FROM faces WHERE id = ? AND user_id = ? ORDER BY created_at DESC', 
        (face_id, user_id)
    ).fetchall()
    conn.close()
    return images

def delete_face_image_by_index(user_id, face_id, image_index):
    """Delete a specific image from face registration"""
    conn = get_db_connection()
    
    # Get all images for this face
    images = conn.execute(
        'SELECT * FROM faces WHERE id = ? AND user_id = ? ORDER BY created_at DESC', 
        (face_id, user_id)
    ).fetchall()
    
    if image_index < len(images):
        image_to_delete = images[image_index]
        conn.execute('DELETE FROM faces WHERE id = ? AND user_id = ?', (image_to_delete['id'], user_id))
        conn.commit()
    
    conn.close()

def get_faces_grouped_by_name(user_id):
    """Get faces grouped by person name for album view"""
    conn = get_db_connection()
    faces = conn.execute('''
        SELECT name, COUNT(*) as image_count, MIN(created_at) as first_registered,
               MAX(created_at) as last_updated, GROUP_CONCAT(image_path) as image_paths
        FROM faces 
        WHERE user_id = ?
        GROUP BY name 
        ORDER BY last_updated DESC
    ''', (user_id,)).fetchall()
    conn.close()
    
    # Process the results
    result = []
    for face in faces:
        image_paths = face['image_paths'].split(',') if face['image_paths'] else []
        result.append({
            'name': face['name'],
            'image_count': face['image_count'],
            'first_registered': face['first_registered'],
            'last_updated': face['last_updated'],
            'image_paths': image_paths,
            'thumbnail_path': image_paths[0] if image_paths else None
        })
    
    return result

def search_media_by_criteria(user_id, start_date=None, end_date=None, media_type='all', detection_type=None):
    """Search recordings and screenshots by various criteria"""
    conn = get_db_connection()
    results = {'recordings': [], 'screenshots': []}
    
    try:
        # Build WHERE clauses
        date_filter = ""
        params = [user_id]
        
        if start_date and end_date:
            date_filter = " AND DATE(start_time) BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        elif start_date:
            date_filter = " AND DATE(start_time) >= ?"
            params.append(start_date)
        elif end_date:
            date_filter = " AND DATE(start_time) <= ?"
            params.append(end_date)
        
        # Search recordings
        if media_type in ['all', 'recordings']:
            recordings_query = f"SELECT * FROM recordings WHERE user_id = ?{date_filter} ORDER BY start_time DESC"
            recordings = conn.execute(recordings_query, params).fetchall()
            
            for recording in recordings:
                results['recordings'].append({
                    'id': recording['id'],
                    'file_path': recording['file_path'],
                    'filename': os.path.basename(recording['file_path']) if recording['file_path'] else 'Unknown',
                    'start_time': recording['start_time'],
                    'end_time': recording['end_time'],
                    'duration': recording['duration'],
                    'file_size': recording.get('file_size', 0),
                    'type': 'recording'
                })
        
        # Search screenshots
        if media_type in ['all', 'screenshots']:
            screenshot_params = [user_id]
            screenshot_filter = ""
            
            if start_date and end_date:
                screenshot_filter = " AND DATE(timestamp) BETWEEN ? AND ?"
                screenshot_params.extend([start_date, end_date])
            elif start_date:
                screenshot_filter = " AND DATE(timestamp) >= ?"
                screenshot_params.append(start_date)
            elif end_date:
                screenshot_filter = " AND DATE(timestamp) <= ?"
                screenshot_params.append(end_date)
            
            if detection_type:
                if screenshot_filter:
                    screenshot_filter += " AND detection_type = ?"
                else:
                    screenshot_filter = " AND detection_type = ?"
                screenshot_params.append(detection_type)
            
            screenshots_query = f"SELECT * FROM screenshots WHERE user_id = ?{screenshot_filter} ORDER BY timestamp DESC"
            screenshots = conn.execute(screenshots_query, screenshot_params).fetchall()
            
            for screenshot in screenshots:
                results['screenshots'].append({
                    'id': screenshot['id'],
                    'file_path': screenshot['file_path'],
                    'filename': os.path.basename(screenshot['file_path']) if screenshot['file_path'] else 'Unknown',
                    'timestamp': screenshot['timestamp'],
                    'detection_type': screenshot.get('detection_type', 'manual'),
                    'description': screenshot.get('description', 'Screenshot'),
                    'file_size': screenshot.get('file_size', 0),
                    'type': 'screenshot'
                })
        
        return results
        
    except Exception as e:
        print(f"Error searching media: {e}")
        return results
    finally:
        conn.close()

def get_timeline_data(user_id, date):
    """Get timeline data for a specific date"""
    conn = get_db_connection()
    timeline = []
    
    try:
        # Get recordings for the date
        recordings = conn.execute(
            "SELECT * FROM recordings WHERE user_id = ? AND DATE(start_time) = ? ORDER BY start_time",
            (user_id, date)
        ).fetchall()
        
        for recording in recordings:
            timeline.append({
                'type': 'recording',
                'id': recording['id'],
                'timestamp': recording['start_time'],
                'end_time': recording['end_time'],
                'duration': recording['duration'],
                'title': f"Recording - {recording['duration']}s",
                'file_path': recording['file_path'],
                'icon': 'video'
            })
        
        # Get screenshots for the date
        screenshots = conn.execute(
            "SELECT * FROM screenshots WHERE user_id = ? AND DATE(timestamp) = ? ORDER BY timestamp",
            (user_id, date)
        ).fetchall()
        
        for screenshot in screenshots:
            timeline.append({
                'type': 'screenshot',
                'id': screenshot['id'],
                'timestamp': screenshot['timestamp'],
                'title': screenshot.get('description', 'Screenshot'),
                'detection_type': screenshot.get('detection_type', 'manual'),
                'file_path': screenshot['file_path'],
                'icon': 'camera'
            })
        
        # Get detections for the date
        detections = conn.execute(
            "SELECT * FROM detections WHERE user_id = ? AND DATE(timestamp) = ? ORDER BY timestamp",
            (user_id, date)
        ).fetchall()
        
        for detection in detections:
            timeline.append({
                'type': 'detection',
                'id': detection['id'],
                'timestamp': detection['timestamp'],
                'title': f"{detection['detection_type'].replace('_', ' ').title()}: {detection['person_name'] or 'Unknown'}",
                'detection_type': detection['detection_type'],
                'person_name': detection['person_name'],
                'confidence': detection['confidence'],
                'alert_level': detection['alert_level'],
                'screenshot_path': detection['screenshot_path'],
                'icon': 'alert' if detection['alert_level'] > 1 else 'info'
            })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return timeline
        
    except Exception as e:
        print(f"Error getting timeline data: {e}")
        return []
    finally:
        conn.close()

def get_playback_statistics(user_id):
    """Get comprehensive playback and storage statistics"""
    conn = get_db_connection()
    
    try:
        # Recording statistics
        recording_stats = conn.execute('''
            SELECT 
                COUNT(*) as total_recordings,
                SUM(duration) as total_duration,
                SUM(file_size) as total_size,
                AVG(duration) as avg_duration,
                MIN(start_time) as oldest_recording,
                MAX(start_time) as newest_recording
            FROM recordings
            WHERE user_id = ?
        ''', (user_id,)).fetchone()
        
        # Screenshot statistics
        screenshot_stats = conn.execute('''
            SELECT 
                COUNT(*) as total_screenshots,
                SUM(file_size) as total_size,
                MIN(timestamp) as oldest_screenshot,
                MAX(timestamp) as newest_screenshot
            FROM screenshots
            WHERE user_id = ?
        ''', (user_id,)).fetchone()
        
        # Detection statistics
        detection_stats = conn.execute('''
            SELECT 
                detection_type,
                COUNT(*) as count
            FROM detections
            WHERE user_id = ?
            GROUP BY detection_type
        ''', (user_id,)).fetchall()
        
        # Recent activity (last 7 days)
        recent_activity = conn.execute('''
            SELECT 
                DATE(start_time) as date,
                COUNT(*) as recording_count,
                SUM(duration) as total_duration
            FROM recordings
            WHERE user_id = ? AND start_time >= DATE('now', '-7 days')
            GROUP BY DATE(start_time)
            ORDER BY date DESC
        ''', (user_id,)).fetchall()
        
        # Storage by month
        monthly_storage = conn.execute('''
            SELECT 
                strftime('%Y-%m', start_time) as month,
                COUNT(*) as recording_count,
                SUM(file_size) as total_size
            FROM recordings
            WHERE user_id = ? AND start_time >= DATE('now', '-12 months')
            GROUP BY strftime('%Y-%m', start_time)
            ORDER BY month DESC
        ''', (user_id,)).fetchall()
        
        return {
            'recordings': {
                'total_count': recording_stats['total_recordings'] or 0,
                'total_duration': recording_stats['total_duration'] or 0,
                'total_size': recording_stats['total_size'] or 0,
                'average_duration': recording_stats['avg_duration'] or 0,
                'oldest': recording_stats['oldest_recording'],
                'newest': recording_stats['newest_recording']
            },
            'screenshots': {
                'total_count': screenshot_stats['total_screenshots'] or 0,
                'total_size': screenshot_stats['total_size'] or 0,
                'oldest': screenshot_stats['oldest_screenshot'],
                'newest': screenshot_stats['newest_screenshot']
            },
            'detections': {
                'by_type': {det['detection_type']: det['count'] for det in detection_stats}
            },
            'recent_activity': [dict(activity) for activity in recent_activity],
            'monthly_storage': [dict(month) for month in monthly_storage],
            'total_storage': (recording_stats['total_size'] or 0) + (screenshot_stats['total_size'] or 0)
        }
        
    except Exception as e:
        print(f"Error getting playback statistics: {e}")
        return {}
    finally:
        conn.close()

def add_known_person(user_id, name, encoding, image_path, detection_id=None):
    """Add a person to known persons (non-threatening but not authorized)"""
    try:
        conn = get_db_connection()
        encoding_str = ','.join(map(str, encoding))
        
        cursor = conn.execute(
            '''INSERT INTO known_persons 
            (user_id, name, encoding, image_path, detection_id, registered_from_false_alarm) 
            VALUES (?, ?, ?, ?, ?, 1)''',
            (user_id, name, encoding_str, image_path, detection_id)
        )
        
        person_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Added known person to database: {name} (ID: {person_id})")
        return person_id
        
    except Exception as e:
        print(f"❌ Error adding known person: {e}")
        raise e


def get_all_known_persons(user_id):
    """Get all known persons for a user"""
    try:
        conn = get_db_connection()
        known_persons = conn.execute(
            'SELECT * FROM known_persons WHERE user_id = ? ORDER BY created_at DESC', (user_id,)
        ).fetchall()
        conn.close()
        return known_persons
    except Exception as e:
        print(f"Error getting known persons: {e}")
        return []


def get_known_person_by_id(user_id, person_id):
    """Get a specific known person by ID"""
    try:
        conn = get_db_connection()
        person = conn.execute(
            'SELECT * FROM known_persons WHERE id = ? AND user_id = ?', 
            (person_id, user_id)
        ).fetchone()
        conn.close()
        return person
    except Exception as e:
        print(f"Error getting known person: {e}")
        return None


def get_screenshots_by_detection_id(user_id, detection_id):
    """Get all screenshots related to a specific detection event"""
    try:
        conn = get_db_connection()
        
        # Get the detection
        detection = conn.execute(
            'SELECT screenshot_path FROM detections WHERE id = ? AND user_id = ?', 
            (detection_id, user_id)
        ).fetchone()
        
        conn.close()
        
        # Return list of screenshot paths
        if detection and detection['screenshot_path']:
            return [detection['screenshot_path']]
        else:
            return []
            
    except Exception as e:
        print(f"Error getting screenshots by detection ID: {e}")
        return []


def search_known_persons_by_name(user_id, query):
    """Search known persons by name"""
    try:
        conn = get_db_connection()
        if query:
            persons = conn.execute(
                'SELECT * FROM known_persons WHERE user_id = ? AND name LIKE ? ORDER BY created_at DESC', 
                (user_id, f'%{query}%')
            ).fetchall()
        else:
            persons = conn.execute(
                'SELECT * FROM known_persons WHERE user_id = ? ORDER BY created_at DESC', (user_id,)
            ).fetchall()
        conn.close()
        return persons
    except Exception as e:
        print(f"Error searching known persons: {e}")
        return []
    
def get_user_by_username(username):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        # Convert sqlite3.Row to dictionary
        return dict(user)
    return None

def create_user(username, password_hash, role='admin', face_id=None, owner_id=None):
    """Creates a user. Returns True if successful, False if username exists."""
    try:
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO users (username, password, role, face_id, owner_id) VALUES (?, ?, ?, ?, ?)', 
            (username, password_hash, role, face_id, owner_id)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_all_users():
    """Get all users"""
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, role, created_at FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return users

def delete_user(user_id):
    """Delete a user account"""
    conn = get_db_connection()
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()

def get_faces_without_accounts(user_id):
    """Get faces that are not linked to any user account - FOR SPECIFIC USER"""
    conn = get_db_connection()
    query = '''
        SELECT f.id, f.name, f.image_path 
        FROM faces f
        LEFT JOIN users u ON f.id = u.face_id
        WHERE u.id IS NULL AND f.user_id = ?
        ORDER BY f.name ASC
    '''
    faces = conn.execute(query, (user_id,)).fetchall()
    conn.close()
    return faces

def get_all_sub_accounts(user_id):
    """Get all users with role='viewer' and their linked face details - FOR SPECIFIC USER"""
    conn = get_db_connection()
    query = '''
        SELECT u.id, u.username, u.face_id, f.name as face_name, f.image_path
        FROM users u
        LEFT JOIN faces f ON u.face_id = f.id
        WHERE u.role = 'viewer' AND f.user_id = ?
        ORDER BY u.username ASC
    '''
    accounts = conn.execute(query, (user_id,)).fetchall()
    conn.close()
    return accounts

def update_user_credentials(user_id, username, password_hash=None):
    """Update username and optionally password"""
    conn = get_db_connection()
    if password_hash:
        conn.execute('UPDATE users SET username = ?, password = ? WHERE id = ?', 
                     (username, password_hash, user_id))
    else:
        conn.execute('UPDATE users SET username = ? WHERE id = ?', 
                     (username, user_id))
    conn.commit()
    conn.close()

def get_face_statistics(user_id):
    """Get face recognition statistics for a user"""
    conn = get_db_connection()
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_faces,
            COUNT(DISTINCT name) as unique_persons,
            MIN(created_at) as oldest_face,
            MAX(created_at) as newest_face
        FROM faces 
        WHERE user_id = ?
    ''', (user_id,)).fetchone()
    conn.close()
    return stats

def get_all_face_encodings(user_id):
    """Get all face encodings for a user"""
    conn = get_db_connection()
    faces = conn.execute('SELECT encoding FROM faces WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return faces

def add_recording(user_id, file_path, start_time, end_time, duration, file_size=0, detection_id=None):
    """Add a recording to the database"""
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO recordings (user_id, file_path, start_time, end_time, duration, file_size, detection_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, file_path, start_time, end_time, duration, file_size, detection_id)
    )
    conn.commit()
    conn.close()

def save_detection_log(user_id, detection_type, details):
    """
    Helper function to save detection logs consistently
    """
    try:
        conn = get_db_connection()
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract details
        person_name = details.get('person_name', 'Unknown')
        confidence = details.get('confidence', 0.0)
        screenshot_path = details.get('screenshot_path')
        alert_level = details.get('alert_level', 1)
        distance = details.get('distance', 0.0)
        
        # Format person name with distance if available
        if distance > 0:
            person_name = f"{person_name} ({distance}m)"
        
        print(f"📝 SAVING DETECTION LOG: User={user_id}, Type={detection_type}, Name={person_name}")
        
        conn.execute(
            'INSERT INTO detections (user_id, detection_type, person_name, confidence, screenshot_path, alert_level, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (user_id, detection_type, person_name, float(confidence), screenshot_path, alert_level, current_time)
        )
        
        conn.commit()
        conn.close()
        
        print(f"✅ Detection log saved for user {user_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving detection log: {e}")
        return False
    
def get_sub_accounts_by_owner(owner_user_id):
    """Get all sub-accounts created by a specific owner user"""
    conn = get_db_connection()
    query = '''
        SELECT u.id, u.username, u.face_id, f.name as face_name, f.image_path
        FROM users u
        LEFT JOIN faces f ON u.face_id = f.id
        WHERE u.role = 'viewer' AND f.user_id = ?
        ORDER BY u.username ASC
    '''
    accounts = conn.execute(query, (owner_user_id,)).fetchall()
    conn.close()
    return accounts