from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils.database import init_db, get_user_by_username, get_user_by_id, create_user
import cv2
import face_recognition
import numpy as np
import os
import sqlite3
import smtplib
import threading
import datetime
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import json
from utils.distance_estimation import get_distance_estimator
import time
from utils.recording_manager import recording_manager as global_recording_manager 
from utils.face_recognition_utils import load_known_faces, load_known_persons
from utils.motion_detection_utils import MotionDetector
from utils.recording_utils import RecordingManager
from utils.audio_utils import AudioManager
from utils.cctv_utils import CCTVController
from utils.twilio_alert_system import twilio_alert_system

app = Flask(__name__)
app.secret_key = 'change_this_to_a_random_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect here if not logged in

# Define User Class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

@login_manager.user_loader
def load_user(user_id):
    user_data = get_user_by_id(user_id)
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['password'])
    return None

# Create a default admin user on startup if one doesn't exist
def init_admin_user():
    user = get_user_by_username('admin')
    if not user:
        hashed_pw = generate_password_hash('admin123', method='pbkdf2:sha256')
        create_user('admin', hashed_pw)
        print("Admin user created: admin / admin123")

# Configuration
UPLOAD_FOLDER = 'static/uploads'
FACES_FOLDER = 'static/faces'
SCREENSHOTS_FOLDER = 'static/screenshots'
RECORDINGS_FOLDER = 'static/recordings'
THUMBNAILS_FOLDER = 'static/thumbnails'

# Global variables
camera = None
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
motion_detector = None
recording_manager = None
# Global variables for face recognition
known_face_encodings = []
known_face_names = []

# Global variables for known persons (non-threatening)
known_persons_encodings = []
known_persons_names = []


# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, FACES_FOLDER, SCREENSHOTS_FOLDER, RECORDINGS_FOLDER, THUMBNAILS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables
motion_detector = None
recording_manager = None

def start_background_cleanup():
    """Start background thread to clean up old face tracking data"""
    def cleanup_loop():
        while True:
            try:
                cleaned_count = twilio_alert_system.cleanup_old_faces(hours=24)
                if cleaned_count > 0:
                    print(f"🧹 Background cleanup: Removed {cleaned_count} old faces")
            except Exception as e:
                print(f"❌ Background cleanup error: {e}")
            time.sleep(3600)  # Run every hour
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()

# Start the background cleanup when your app starts
start_background_cleanup()


@app.route('/')
@login_required 
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/face_registration')
@login_required
def face_registration():
    return render_template('face_registration.html')

@app.route('/motion_detection')
@login_required
def motion_detection():
    return render_template('motion_detection.html')

@app.route('/alerts')
@login_required
def alerts():
    return render_template('alerts.html')

@app.route('/playback')
@login_required
def playback():
    return render_template('playback.html')

@app.route('/controls')
@login_required
def controls():
    return render_template('controls.html')

@app.route('/detection_logs')
@login_required
def detection_logs():
    """Display all detection logs with specific filtering"""
    return render_template('detection_logs.html')

@app.route('/api/debug_verification')
def debug_verification():
    """Debug number verification status"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        debug_info = twilio_alert_system.debug_verification_status()
        return jsonify({'success': True, 'debug': debug_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_data = get_user_by_username(username)
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data['id'], user_data['username'], user_data['password'])
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/api/twilio_settings', methods=['GET', 'POST'])
def twilio_settings():
    """Get or update Twilio settings with proper auth token handling"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        if request.method == 'GET':
            settings = twilio_alert_system.get_settings()
            return jsonify({'success': True, 'settings': settings})
        
        elif request.method == 'POST':
            settings = {
                'account_sid': request.form.get('account_sid'),
                'auth_token': request.form.get('auth_token'),
                'twilio_number': request.form.get('twilio_number'),
                'test_mode': request.form.get('test_mode') == 'true'
            }
            
            # Validate required fields
            if not settings['account_sid'] or not settings['twilio_number']:
                return jsonify({'success': False, 'error': 'Account SID and Twilio Number are required'})
            
            success = twilio_alert_system.save_settings(settings)
            if success:
                return jsonify({'success': True, 'message': 'Settings updated successfully'})
            else:
                return jsonify({'success': False, 'error': 'Failed to save settings'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/motion_tracking_status')
def get_motion_tracking_status():
    """Get motion-gated tracking status"""
    try:
        from utils.face_recognition_utils import motion_gated_tracker
        stats = motion_gated_tracker.get_tracking_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reset_motion_tracking', methods=['POST'])
def reset_motion_tracking():
    """Reset motion-gated tracking - safe version that handles both old and new code"""
    try:
        from utils.face_recognition_utils import motion_gated_tracker
        
        # Get count before reset for logging
        before_count = len(motion_gated_tracker.tracked_faces)
        
        # Use a safe reset approach
        with motion_gated_tracker.lock:
            # Clear tracked faces
            motion_gated_tracker.tracked_faces.clear()
            
            # Clear face_to_tracker if it exists (backward compatibility)
            if hasattr(motion_gated_tracker, 'face_to_tracker'):
                motion_gated_tracker.face_to_tracker.clear()
            
            # Reset motion tracking
            motion_gated_tracker.last_motion_time = None
            motion_gated_tracker.motion_active = False
            
            print(f"✅ Tracking reset - cleared {before_count} tracked faces")
        
        return jsonify({
            'success': True, 
            'message': f'Motion tracking reset - cleared {before_count} tracked faces'
        })
    except Exception as e:
        print(f"❌ Error resetting tracking: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_motion_timeout', methods=['POST'])
def update_motion_timeout():
    """Update motion timeout setting"""
    try:
        from utils.face_recognition_utils import motion_gated_tracker
        timeout = float(request.form.get('timeout', 3.0))
        motion_gated_tracker.motion_timeout = timeout
        return jsonify({'success': True, 'message': f'Motion timeout updated to {timeout}s'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/cleanup_old_trackers', methods=['POST'])
def cleanup_old_trackers():
    """Clean up old trackers manually - safe version"""
    try:
        from utils.face_recognition_utils import motion_gated_tracker
        
        # Count before cleanup
        before_count = len(motion_gated_tracker.tracked_faces)
        
        # Use the tracker's cleanup method if it exists, otherwise do manual cleanup
        if hasattr(motion_gated_tracker, '_cleanup_old_trackers'):
            motion_gated_tracker._cleanup_old_trackers()
        else:
            # Manual cleanup
            current_time = time.time()
            remove_trackers = []
            
            with motion_gated_tracker.lock:
                for tracker_id, tracker in motion_gated_tracker.tracked_faces.items():
                    time_since_last_seen = current_time - tracker.get('last_seen', 0)
                    if time_since_last_seen > 300:  # 5 minutes
                        remove_trackers.append(tracker_id)
                
                for tracker_id in remove_trackers:
                    del motion_gated_tracker.tracked_faces[tracker_id]
        
        # Count after cleanup
        after_count = len(motion_gated_tracker.tracked_faces)
        cleaned_count = before_count - after_count
        
        return jsonify({
            'success': True, 
            'cleaned_count': cleaned_count,
            'remaining': after_count,
            'message': f'Cleaned up {cleaned_count} old trackers, {after_count} remain'
        })
    except Exception as e:
        print(f"❌ Error cleaning up trackers: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/update_authorized_cooldown', methods=['POST'])
def update_authorized_cooldown():
    """Update authorized cooldown setting"""
    try:
        from utils.face_recognition_utils import motion_gated_tracker
        cooldown = float(request.form.get('cooldown', 5.0))
        motion_gated_tracker.authorized_cooldown = cooldown
        return jsonify({'success': True, 'message': f'Authorized cooldown updated to {cooldown}s'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



@app.route('/api/get_detection_logs_simple')
def get_detection_logs_simple():
    """Simple version - just filter by detection_type and person_name patterns"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        print("🔍 DEBUG: Simple version - fetching logs...")
        
        # Direct query for specific patterns
        # EXCLUDE entries with "Unauthorized (distance)" pattern
        query = '''
            SELECT * FROM detections 
            WHERE (detection_type IN ('face_detection', 'unauthorized_face'))
            AND (
                person_name LIKE '%Authorized%' OR
                person_name LIKE '%Known%' OR
                person_name LIKE '%Unauthorized%' OR
                person_name = 'Authorized Person' OR
                person_name = 'Known Person' OR
                person_name = 'Unauthorized Person'
            )
            AND person_name NOT LIKE 'Unauthorized (%'  -- Exclude distance entries
            ORDER BY timestamp DESC 
            LIMIT 200
        '''
        
        detections = conn.execute(query).fetchall()
        print(f"📊 Found {len(detections)} matching detections (excluding distance entries)")
        
        logs_list = []
        # Dictionary to track last seen timestamp for each person_name
        last_seen = {}
        
        for detection in detections:
            person_name = detection['person_name']
            timestamp_str = detection['timestamp']
            
            # Parse timestamp
            try:
                timestamp_obj = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    timestamp_obj = datetime.datetime.strptime(timestamp_str, '%m/%d/%Y %I:%M:%S %p')
                except:
                    timestamp_obj = datetime.datetime.now()
            
            # Check if we've seen this person recently (within 5 seconds)
            if person_name in last_seen:
                last_time = last_seen[person_name]
                time_diff = abs((timestamp_obj - last_time).total_seconds())
                
                # Skip if within 5 seconds of last detection
                if time_diff <= 5:
                    print(f"⏭️ Skipping duplicate '{person_name}' at {timestamp_str} (diff: {time_diff:.1f}s)")
                    continue
            
            # Update last seen time
            last_seen[person_name] = timestamp_obj
            
            # Determine person type
            if 'Authorized' in person_name or person_name == 'Authorized Person':
                person_type = 'Authorized Person'
                detection_type = 'face_detection'
            elif 'Known' in person_name or person_name == 'Known Person':
                person_type = 'Known Person'
                detection_type = 'face_detection'
            elif 'Unauthorized' in person_name or detection['detection_type'] == 'unauthorized_face':
                # Double-check for distance entries (should already be filtered by query)
                if 'Unauthorized (' in person_name:
                    print(f"⏭️ Skipping distance entry: '{person_name}'")
                    continue
                person_type = 'Unauthorized Face'
                detection_type = 'face_detection'
            else:
                continue
            
            log_data = {
                'id': detection['id'],
                'timestamp': timestamp_str,
                'detection_type': detection_type,
                'person_type': person_type,
                'person_name': person_name,
                'confidence': float(detection['confidence']) if detection['confidence'] else 0,
                'screenshot_path': detection['screenshot_path'],
                'alert_level': detection['alert_level']
            }
            logs_list.append(log_data)
            
            print(f"✅ Added: {person_type} - '{person_name}' at {timestamp_str}")
        
        conn.close()
        
        print(f"📊 Returning {len(logs_list)} unique logs (distance entries filtered + time deduplication)")
        
        return jsonify({'success': True, 'logs': logs_list})
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/debug_tracking')
def debug_tracking():
    """Debug endpoint to see tracking details"""
    try:
        from utils.face_recognition_utils import motion_gated_tracker
        
        with motion_gated_tracker.lock:
            debug_info = {
                'total_tracked': len(motion_gated_tracker.tracked_faces),
                'unique_hashes': len(motion_gated_tracker.face_to_tracker),
                'motion_active': motion_gated_tracker.motion_active,
                'last_motion_time': motion_gated_tracker.last_motion_time,
                'trackers': []
            }
            
            current_time = time.time()
            for tracker_id, data in motion_gated_tracker.tracked_faces.items():
                time_since_seen = current_time - data['last_seen']
                debug_info['trackers'].append({
                    'id': tracker_id[:8],
                    'hash': data.get('face_hash', '')[:20] + '...',
                    'state': data['state'],
                    'name': data['name'],
                    'time_since_seen': round(time_since_seen, 1),
                    'motion_active': data['motion_active'],
                    'encoding_sample': data.get('encoding_sample', '')[:50] + '...' if data.get('encoding_sample') else None
                })
        
        return jsonify({'success': True, 'debug': debug_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/start_event_recording', methods=['POST'])
def start_event_recording():
    """Start recording for specific event types with cooldown"""
    try:
        event_type = request.form.get('event_type')  # authorized, unauthorized, known, motion
        event_id = request.form.get('event_id')
        
        if not event_type or not event_id:
            return jsonify({'success': False, 'error': 'Event type and ID required'})
        
        # Start recording
        recording_path = global_recording_manager.start_recording_for_event(event_id, event_type)
        
        if recording_path:
            return jsonify({
                'success': True, 
                'recording_path': recording_path,
                'message': f'Recording started for {event_type}'
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'Recording not started (may be in cooldown)'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_filtered_alerts')
def get_filtered_alerts():
    """Get alerts filtered by type using efficient database joins - INCLUDES AUTHORIZED AND KNOWN FACES"""
    try:
        filter_type = request.args.get('filter', 'all')
        
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Updated query to properly filter face detection types
        query = '''
            SELECT 
                d.*,
                CASE 
                    WHEN d.detection_type = 'face_detection' AND d.person_name IN (SELECT name FROM faces) THEN 'authorized'
                    WHEN d.detection_type = 'face_detection' AND d.person_name IN (SELECT name FROM known_persons) THEN 'known'
                    WHEN d.detection_type = 'face_detection' AND (d.person_name = 'Unauthorized' OR d.person_name LIKE 'Unauthorized (%') THEN 'unauthorized'
                    WHEN d.detection_type = 'motion_detection' AND d.person_name LIKE 'Motion Detected%' THEN 'motion'
                    ELSE 'other'
                END as category
            FROM detections d 
            WHERE d.detection_type IN ('face_detection', 'motion_detection')
        '''
        params = []
        
        # Apply filters - FIXED: Properly filter each category
        if filter_type == 'authorized':
            query += ' AND category = "authorized"'
        elif filter_type == 'unauthorized':
            query += ' AND category = "unauthorized"'
        elif filter_type == 'known':
            query += ' AND category = "known"'
        elif filter_type == 'motion':
            query += ' AND category = "motion"'
        elif filter_type == 'all_faces':
            query += ' AND category IN ("authorized", "known", "unauthorized")'
        
        query += ' ORDER BY d.timestamp DESC LIMIT 100'
        
        detections = conn.execute(query, params).fetchall()
        
        alerts_list = []
        for detection in detections:
            # Try to find recording file by filename pattern matching
            recording_path = find_recording_by_detection(detection)
            
            # Determine display name and icon based on category
            category = detection['category']
            person_name = detection['person_name']
            
            alert_data = {
                'id': detection['id'],
                'type': detection['detection_type'],
                'person_name': person_name,
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp'],
                'screenshot_path': detection['screenshot_path'],
                'alert_level': detection['alert_level'],
                'recording_path': recording_path,
                'category': category
            }
            
            alerts_list.append(alert_data)
        
        conn.close()
        
        print(f"📊 DEBUG: Returning {len(alerts_list)} alerts for filter '{filter_type}'")
        print(f"📊 DEBUG: Categories found: {set([alert['category'] for alert in alerts_list])}")
        
        return jsonify({'success': True, 'alerts': alerts_list})
        
    except Exception as e:
        print(f"❌ ERROR in get_filtered_alerts: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
    

def get_alert_level_from_event(event):
    """Determine alert level from event data"""
    if event['is_authorized']:
        return 1
    elif event['is_known_person']:
        return 1
    elif event['trigger_type'] == 'unauthorized_face':
        return 3
    elif event['trigger_type'] == 'motion_detection':
        return 2
    else:
        return 1

def get_category_from_event(event):
    """Determine category from event data"""
    if event['is_authorized']:
        return 'authorized'
    elif event['is_known_person']:
        return 'known'
    elif event['trigger_type'] == 'unauthorized_face':
        return 'unauthorized'
    elif event['trigger_type'] == 'motion_detection':
        return 'motion'
    else:
        return 'other'

def find_recording_by_detection(detection):
    """Find recording file by detection timestamp and type"""
    try:
        import os
        import datetime
        
        detection_time = datetime.datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
        detection_date = detection_time.strftime('%Y%m%d')
        
        recordings_dir = 'static/recordings'
        if not os.path.exists(recordings_dir):
            return None
        
        # Look for recording files that match the detection time and type
        for filename in os.listdir(recordings_dir):
            if not filename.endswith(('.mp4', '.avi')):
                continue
                
            # Check if filename contains the detection date
            if detection_date in filename:
                # Determine event type based on category
                category = detection['category']
                if category == 'authorized' and 'authorized' in filename.lower():
                    return f"static/recordings/{filename}"
                elif category == 'known' and 'known' in filename.lower():
                    return f"static/recordings/{filename}"
                elif category == 'unauthorized' and 'unauthorized' in filename.lower():
                    return f"static/recordings/{filename}"
                elif category == 'motion' and 'motion' in filename.lower():
                    return f"static/recordings/{filename}"
        
        # If no exact match found, return any recording from that day
        for filename in os.listdir(recordings_dir):
            if detection_date in filename and filename.endswith(('.mp4', '.avi')):
                return f"static/recordings/{filename}"
                
        return None
        
    except Exception as e:
        print(f"Error finding recording for detection: {e}")
        return None
    
@app.route('/api/get_authorized_events')
def get_authorized_events():
    """Get only authorized face detection events with recordings"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        print("🔍 DEBUG: Fetching authorized events from database...")
        
        # Get authorized face detections
        authorized_detections = conn.execute('''
            SELECT d.* 
            FROM detections d 
            WHERE d.detection_type = 'face_detection' 
            AND d.person_name IN (SELECT name FROM faces)
            ORDER BY d.timestamp DESC 
            LIMIT 50
        ''').fetchall()
        
        print(f"🔍 DEBUG: Found {len(authorized_detections)} authorized detections")
        
        events_list = []
        for detection in authorized_detections:
            recording_path = find_recording_for_authorized_event(detection)
            
            event_data = {
                'id': detection['id'],
                'type': 'authorized_face',
                'person_name': detection['person_name'],
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp'],
                'screenshot_path': detection['screenshot_path'],
                'recording_path': recording_path,
                'category': 'authorized'
            }
            
            events_list.append(event_data)
            print(f"🔍 DEBUG: Added authorized event: {detection['person_name']}")
        
        conn.close()
        return jsonify({'success': True, 'events': events_list})
        
    except Exception as e:
        print(f"❌ ERROR in get_authorized_events: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
    
def find_recording_for_authorized_event(detection):
    """Find recording specifically for authorized events"""
    try:
        import os
        import datetime
        
        detection_time = datetime.datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
        detection_date = detection_time.strftime('%Y%m%d')
        
        recordings_dir = 'static/recordings'
        if not os.path.exists(recordings_dir):
            return None
        
        # Look for authorized recordings
        for filename in os.listdir(recordings_dir):
            if filename.endswith(('.mp4', '.avi')) and 'authorized' in filename.lower():
                if detection_date in filename:
                    return f"static/recordings/{filename}"
        
        # Fallback: any recording from that day
        for filename in os.listdir(recordings_dir):
            if detection_date in filename and filename.endswith(('.mp4', '.avi')):
                return f"static/recordings/{filename}"
                
        return None
        
    except Exception as e:
        print(f"Error finding authorized recording: {e}")
        return None

def find_recording_for_known_event(detection):
    """Find recording specifically for known person events"""
    try:
        import os
        import datetime
        
        detection_time = datetime.datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
        detection_date = detection_time.strftime('%Y%m%d')
        
        recordings_dir = 'static/recordings'
        if not os.path.exists(recordings_dir):
            return None
        
        # Look for known person recordings
        for filename in os.listdir(recordings_dir):
            if filename.endswith(('.mp4', '.avi')) and 'known' in filename.lower():
                if detection_date in filename:
                    return f"static/recordings/{filename}"
        
        # Fallback: any recording from that day
        for filename in os.listdir(recordings_dir):
            if detection_date in filename and filename.endswith(('.mp4', '.avi')):
                return f"static/recordings/{filename}"
                
        return None
        
    except Exception as e:
        print(f"Error finding known recording: {e}")
        return None
    
@app.route('/api/get_all_alert_events')
def get_all_alert_events():
    """Get ALL alert events including authorized and known"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get all events from alert_events table
        events = conn.execute('''
            SELECT * FROM alert_events 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''').fetchall()
        
        events_list = []
        for event in events:
            event_data = {
                'event_id': event['event_id'],
                'trigger_type': event['trigger_type'],
                'timestamp': event['timestamp'],
                'recording_filepath': event['recording_filepath'],
                'review_status': event['review_status'],
                'call_status': event['call_status']
            }
            events_list.append(event_data)
        
        conn.close()
        return jsonify({'success': True, 'events': events_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_alert_events_by_type')
def get_alert_events_by_type():
    """Get alert events filtered by type - INCLUDES ALL FACE TYPES"""
    try:
        event_type = request.args.get('type', 'all')
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        if event_type == 'all':
            events = conn.execute('''
                SELECT * FROM alert_events 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''').fetchall()
        else:
            # Map filter types to event types in database
            event_type_map = {
                'authorized': 'authorized_face',
                'unauthorized': 'unauthorized_face', 
                'known': 'known_face',
                'motion': 'motion_detection'
            }
            
            db_event_type = event_type_map.get(event_type, event_type)
            events = conn.execute('''
                SELECT * FROM alert_events 
                WHERE trigger_type = ?
                ORDER BY timestamp DESC 
                LIMIT 100
            ''', (db_event_type,)).fetchall()
        
        events_list = []
        for event in events:
            event_data = {
                'event_id': event['event_id'],
                'trigger_type': event['trigger_type'],
                'timestamp': event['timestamp'],
                'recording_filepath': event['recording_filepath'],
                'review_status': event['review_status'],
                'call_status': event['call_status'],
                'person_name': event.get('person_name', 'Unknown'),
                'confidence': event.get('confidence', 0),
                'distance_meters': event.get('distance_meters', 0),
                'is_authorized': bool(event.get('is_authorized', False)),
                'is_known_person': bool(event.get('is_known_person', False))
            }
            events_list.append(event_data)
        
        conn.close()
        return jsonify({'success': True, 'events': events_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_known_events')
def get_known_events():
    """Get only known person detection events with recordings"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        print("🔍 DEBUG: Fetching known events from database...")
        
        # Get known person detections
        known_detections = conn.execute('''
            SELECT d.* 
            FROM detections d 
            WHERE d.detection_type = 'face_detection' 
            AND d.person_name IN (SELECT name FROM known_persons)
            ORDER BY d.timestamp DESC 
            LIMIT 50
        ''').fetchall()
        
        print(f"🔍 DEBUG: Found {len(known_detections)} known detections")
        
        events_list = []
        for detection in known_detections:
            recording_path = find_recording_for_known_event(detection)
            
            event_data = {
                'id': detection['id'],
                'type': 'known_face',
                'person_name': detection['person_name'],
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp'],
                'screenshot_path': detection['screenshot_path'],
                'recording_path': recording_path,
                'category': 'known'
            }
            
            events_list.append(event_data)
            print(f"🔍 DEBUG: Added known event: {detection['person_name']}")
        
        conn.close()
        return jsonify({'success': True, 'events': events_list})
        
    except Exception as e:
        print(f"❌ ERROR in get_known_events: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})  
    
@app.route('/api/phone_numbers', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_phone_numbers():
    """Manage phone numbers for alerts"""
    try:
        from utils.database import (
            get_all_phone_numbers, add_phone_number, update_phone_number_order,
            toggle_phone_number_active, delete_phone_number, update_phone_number
        )
        from utils.twilio_alert_system import twilio_alert_system
        
        if request.method == 'GET':
            numbers = get_all_phone_numbers()
            numbers_list = []
            for number in numbers:
                numbers_list.append({
                    'id': number['id'],
                    'phone_number': number['phone_number'],
                    'display_name': number['display_name'],
                    'is_active': bool(number['is_active']),
                    'sort_order': number['sort_order'],
                    'created_at': number['created_at']
                })
            return jsonify({'success': True, 'phone_numbers': numbers_list})
        
        elif request.method == 'POST':
            phone_number = request.form.get('phone_number')
            display_name = request.form.get('display_name')
            
            if not phone_number:
                return jsonify({'success': False, 'error': 'Phone number required'})
            
            add_phone_number(phone_number, display_name)
            
            # Reload numbers in alert system
            twilio_alert_system.load_phone_numbers()
            
            return jsonify({'success': True, 'message': 'Phone number added'})
        
        elif request.method == 'PUT':
            if request.form.get('action') == 'reorder':
                phone_numbers = request.json.get('phone_numbers', [])
                update_phone_number_order(phone_numbers)
                
                # Reload numbers in alert system
                twilio_alert_system.load_phone_numbers()
                
                return jsonify({'success': True, 'message': 'Phone numbers reordered'})
            
            elif request.form.get('action') == 'toggle':
                phone_id = request.form.get('phone_id')
                is_active = request.form.get('is_active') == 'true'
                
                toggle_phone_number_active(phone_id, is_active)
                
                # Reload numbers in alert system
                twilio_alert_system.load_phone_numbers()
                
                return jsonify({'success': True, 'message': 'Phone number status updated'})
            
            elif request.form.get('action') == 'update':
                phone_id = request.form.get('phone_id')
                phone_number = request.form.get('phone_number')
                display_name = request.form.get('display_name')
                
                update_phone_number(phone_id, phone_number, display_name)
                
                # Reload numbers in alert system
                twilio_alert_system.load_phone_numbers()
                
                return jsonify({'success': True, 'message': 'Phone number updated'})
        
        elif request.method == 'DELETE':
            phone_id = request.args.get('phone_id')
            
            if not phone_id:
                return jsonify({'success': False, 'error': 'Phone ID required'})
            
            delete_phone_number(phone_id)
            
            # Reload numbers in alert system
            twilio_alert_system.load_phone_numbers()
            
            return jsonify({'success': True, 'message': 'Phone number deleted'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})  

@app.route('/api/get_recording_for_alert/<int:alert_id>')
def get_recording_for_alert(alert_id):
    """Get recording file for a specific alert"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        recording = conn.execute('''
            SELECT r.file_path FROM recordings r 
            JOIN detections d ON r.detection_id = d.id 
            WHERE d.id = ?
        ''', (alert_id,)).fetchone()
        
        conn.close()
        
        if recording and recording['file_path']:
            return jsonify({
                'success': True,
                'recording_path': recording['file_path'],
                'filename': os.path.basename(recording['file_path'])
            })
        else:
            return jsonify({'success': False, 'error': 'No recording found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/debug_recordings')
def debug_recordings():
    """Debug endpoint to check recording files"""
    try:
        recordings_dir = 'static/recordings'
        files = []
        
        if os.path.exists(recordings_dir):
            for filename in os.listdir(recordings_dir):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(recordings_dir, filename)
                    files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'exists': os.path.exists(filepath),
                        'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                        'web_path': f'/static/recordings/{filename}'
                    })
        
        return jsonify({
            'success': True,
            'recordings_dir': recordings_dir,
            'files': files,
            'total_files': len(files)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/alert_events')
def get_alert_events():
    """Get alert event history"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        limit = int(request.args.get('limit', 50))
        events = twilio_alert_system.get_event_history(limit)
        return jsonify({'success': True, 'events': events})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/debug_face_tracking_detailed')
def debug_face_tracking_detailed():
    """Detailed debug endpoint for face tracking"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        with twilio_alert_system.lock:
            debug_info = {
                'total_tracked_faces': len(twilio_alert_system.alerted_faces),
                'total_answered_faces': len(twilio_alert_system.answered_calls),
                'face_grouping_threshold': twilio_alert_system.face_grouping_threshold,
                'alert_cooldown': twilio_alert_system.alert_cooldown,
                'max_alerts_per_face': twilio_alert_system.max_alerts_per_face,
                'tracked_faces': [],
                'answered_faces': list(twilio_alert_system.answered_calls)
            }
            
            current_time = time.time()
            for face_hash, data in twilio_alert_system.alerted_faces.items():
                time_since_alert = current_time - data['last_alert_time']
                cooldown_remaining = max(0, twilio_alert_system.alert_cooldown - time_since_alert)
                
                debug_info['tracked_faces'].append({
                    'face_id': face_hash[:8],
                    'alert_count': data['alert_count'],
                    'last_alert_seconds_ago': int(time_since_alert),
                    'cooldown_remaining_seconds': int(cooldown_remaining),
                    'encoding_sample': data.get('encoding_sample', [])
                })
        
        return jsonify({'success': True, 'debug': debug_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/reset_answered_calls', methods=['POST'])
def reset_answered_calls():
    """Reset answered calls tracking"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        with twilio_alert_system.lock:
            count = len(twilio_alert_system.answered_calls)
            twilio_alert_system.answered_calls = set()
        return jsonify({
            'success': True, 
            'message': f'Reset {count} answered calls'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
    
@app.route('/api/face_tracking_stats')
def get_face_tracking_stats():
    """Get face tracking statistics"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        # ✅ FIX: Ensure the answered_calls attribute exists
        if not hasattr(twilio_alert_system, 'answered_calls'):
            twilio_alert_system.answered_calls = set()
            
        stats = twilio_alert_system.get_face_tracking_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleanup_face_tracking', methods=['POST'])
def cleanup_face_tracking():
    """Clean up old face tracking entries"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        hours = int(request.form.get('hours', 24))
        cleaned_count = twilio_alert_system.cleanup_old_faces(hours)
        return jsonify({'success': True, 'cleaned_count': cleaned_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/convert_recording', methods=['POST'])
def convert_recording():
    """Convert a recording to compatible format"""
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'})
        
        filepath = os.path.join('static', 'recordings', filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        from utils.recording_manager import recording_manager
        result = recording_manager.convert_to_compatible_format(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recording_info/<filename>')
def get_recording_info(filename):
    """Get detailed information about a recording"""
    try:
        filepath = os.path.join('static', 'recordings', filename)
        
        from utils.recording_manager import recording_manager
        info = recording_manager.get_recording_info(filepath)
        
        return jsonify({'success': True, 'info': info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reset_face_tracking', methods=['POST'])
def reset_face_tracking():
    """Reset all face tracking (clear memory)"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        with twilio_alert_system.lock:
            count = len(twilio_alert_system.alerted_faces)
            twilio_alert_system.alerted_faces = {}
        return jsonify({
            'success': True, 
            'message': f'Face tracking reset - cleared {count} faces'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/debug_face_tracking')
def debug_face_tracking():
    """Debug endpoint to see current face tracking state"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        with twilio_alert_system.lock:
            debug_info = {
                'total_tracked_faces': len(twilio_alert_system.alerted_faces),
                'alert_cooldown_seconds': twilio_alert_system.alert_cooldown,
                'max_alerts_per_face': twilio_alert_system.max_alerts_per_face,
                'faces': []
            }
            
            current_time = time.time()
            for face_hash, data in list(twilio_alert_system.alerted_faces.items())[:20]:
                time_since_alert = current_time - data['last_alert_time']
                cooldown_remaining = max(0, twilio_alert_system.alert_cooldown - time_since_alert)
                
                debug_info['faces'].append({
                    'face_id': face_hash[:8],
                    'alert_count': data['alert_count'],
                    'last_alert_seconds_ago': int(time_since_alert),
                    'cooldown_remaining_seconds': int(cooldown_remaining),
                    'can_alert_again': cooldown_remaining == 0 and data['alert_count'] < twilio_alert_system.max_alerts_per_face,
                    'max_alerts_reached': data['alert_count'] >= twilio_alert_system.max_alerts_per_face,
                    'first_seen': datetime.datetime.fromtimestamp(data['first_seen']).isoformat()
                })
        
        return jsonify({'success': True, 'debug': debug_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/fix_recording_url/<filename>')
def fix_recording_url(filename):
    """Fix recording URL encoding issues"""
    try:
        # Remove any URL encoding artifacts
        clean_filename = filename.replace('%02', '').replace('%03', '').replace('%04', '')
        
        # Ensure it's a valid filename
        filepath = os.path.join('static', 'recordings', clean_filename)
        
        if os.path.exists(filepath):
            return jsonify({
                'success': True, 
                'clean_filename': clean_filename,
                'url': f'/static/recordings/{clean_filename}'
            })
        else:
            return jsonify({'success': False, 'error': 'File not found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/list_recordings')
def list_recordings():
    """List all recordings with proper URLs"""
    try:
        recordings_dir = 'static/recordings'
        recordings = []
        
        for filename in os.listdir(recordings_dir):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                filepath = os.path.join(recordings_dir, filename)
                if os.path.exists(filepath):
                    recordings.append({
                        'filename': filename,
                        'url': f'/static/recordings/{filename}',
                        'file_size': os.path.getsize(filepath),
                        'file_size_mb': round(os.path.getsize(filepath) / (1024**2), 2)
                    })
        
        return jsonify({'success': True, 'recordings': recordings})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/diagnose_codecs')
def diagnose_codecs():
    """Diagnose available codecs on the system"""
    try:
        import cv2
        
        # Test different codecs
        codecs_to_test = [
            ('MJPG', 'avi', cv2.VideoWriter_fourcc(*'MJPG')),
            ('XVID', 'avi', cv2.VideoWriter_fourcc(*'XVID')),
            ('DIVX', 'avi', cv2.VideoWriter_fourcc(*'DIVX')),
            ('mp4v', 'mp4', cv2.VideoWriter_fourcc(*'mp4v')),
            ('avc1', 'mp4', cv2.VideoWriter_fourcc(*'avc1')),
        ]
        
        results = []
        test_resolution = (640, 480)
        test_fps = 15
        
        for codec_name, extension, fourcc in codecs_to_test:
            test_filename = f"test_{codec_name}.{extension}"
            test_path = os.path.join('static', 'recordings', test_filename)
            
            try:
                # Test writing
                writer = cv2.VideoWriter(test_path, fourcc, test_fps, test_resolution)
                can_write = writer.isOpened()
                
                if can_write:
                    # Write a test frame
                    test_frame = np.zeros((test_resolution[1], test_resolution[0], 3), dtype=np.uint8)
                    writer.write(test_frame)
                    writer.release()
                    
                    # Test reading
                    cap = cv2.VideoCapture(test_path)
                    can_read = cap.isOpened()
                    if can_read:
                        ret, frame = cap.read()
                        can_read = ret
                    cap.release()
                    
                    # Clean up
                    if os.path.exists(test_path):
                        os.path.getsize(test_path)
                        os.remove(test_path)
                else:
                    can_read = False
                    
                results.append({
                    'codec': codec_name,
                    'extension': extension,
                    'can_write': can_write,
                    'can_read': can_read,
                    'status': '✅ WORKING' if (can_write and can_read) else '❌ FAILED'
                })
                
            except Exception as e:
                results.append({
                    'codec': codec_name,
                    'extension': extension,
                    'can_write': False,
                    'can_read': False,
                    'status': f'❌ ERROR: {str(e)}'
                })
        
        return jsonify({'success': True, 'codec_tests': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_recording')
def test_recording():
    """Create a test recording to verify functionality"""
    try:
        from utils.recording_manager import recording_manager
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some test pattern
        cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.putText(test_frame, "TEST RECORDING", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Start recording
        event_id = f"test_{int(time.time())}"
        filepath = recording_manager.start_recording_for_event(event_id, "test")
        
        if not filepath:
            return jsonify({'success': False, 'error': 'Failed to start recording'})
        
        # Add some test frames
        for i in range(30):  # 2 seconds at 15fps
            recording_manager.add_frame(test_frame)
            time.sleep(0.066)  # ~15fps
        
        # Stop recording
        recording_manager.stop_recording()
        
        # Verify the recording
        verification = recording_manager.verify_recording_playback(filepath)
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'verification': verification,
            'url': f'/{filepath}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/start_image_recording', methods=['POST'])
def start_image_recording():
    """Record as image sequence instead of video"""
    try:
        event_id = f"images_{int(time.time())}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"recording_{event_id}_{timestamp}"
        folder_path = os.path.join('static', 'recordings', folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        
        return jsonify({
            'success': True,
            'event_id': event_id,
            'folder_path': folder_path,
            'message': 'Image recording started'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_recording_frame', methods=['POST'])
def save_recording_frame():
    """Save a frame for image sequence recording"""
    try:
        event_id = request.form.get('event_id')
        folder_path = request.form.get('folder_path')
        image_data = request.form.get('image_data')
        
        if not all([event_id, folder_path, image_data]):
            return jsonify({'success': False, 'error': 'Missing parameters'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save frame
        timestamp = datetime.now().strftime('%H%M%S_%f')[:-3]
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(folder_path, filename)
        
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/unauthorized_detections_unique')
def unauthorized_detections_unique():
    """
    ✅ NEW ENDPOINT: Get unique unauthorized detections (no duplicates)
    """
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get all recent detections
        all_detections = conn.execute('''
            SELECT * FROM detections 
            WHERE detection_type IN ('face_detection', 'motion_detection')
            ORDER BY timestamp DESC 
            LIMIT 100
        ''').fetchall()
        
        # Deduplicate by person_name + detection_type
        unique_detections = {}
        
        for detection in all_detections:
            detection_dict = dict(detection)
            person_name = detection_dict['person_name'] or 'Unknown'
            detection_type = detection_dict['detection_type']
            
            # Create unique key
            key = f"{person_name}_{detection_type}"
            
            # Keep only most recent
            if key not in unique_detections:
                unique_detections[key] = {
                    'id': detection_dict['id'],
                    'type': detection_type,
                    'person_name': person_name,
                    'confidence': detection_dict['confidence'],
                    'timestamp': detection_dict['timestamp'],
                    'screenshot_path': detection_dict['screenshot_path'],
                    'alert_level': detection_dict['alert_level']
                }
        
        # Convert to list and sort by timestamp
        detections_list = sorted(
            unique_detections.values(),
            key=lambda x: x['timestamp'],
            reverse=True
        )[:20]  # Return top 20
        
        conn.close()
        
        print(f"📊 Returning {len(detections_list)} unique detections (from {len(all_detections)} total)")
        
        return jsonify({'success': True, 'detections': detections_list})
        
    except Exception as e:
        print(f"❌ ERROR in unauthorized_detections_unique: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_review_status', methods=['POST'])
def update_review_status():
    """Update event review status"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        event_id = request.form.get('event_id')
        status = request.form.get('status')
        
        success = twilio_alert_system.update_review_status(event_id, status)
        return jsonify({'success': success})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_twilio_call', methods=['POST'])
def test_twilio_call():
    """Test Twilio call functionality"""
    try:
        from utils.twilio_alert_system import twilio_alert_system
        
        test_number = request.form.get('test_number')
        if not test_number:
            return jsonify({'success': False, 'error': 'Test number required'})
        
        result = twilio_alert_system.make_voice_call(
            test_number, 
            "This is a test call from your security system. Everything is working correctly."
        )
        
        return jsonify({'success': result['status'] == 'initiated', 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recording_stats')
def get_recording_stats():
    """Get recording statistics"""
    try:
        from utils.recording_manager import recording_manager
        
        stats = recording_manager.get_recording_stats()
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleanup_recordings', methods=['POST'])
def cleanup_recordings():
    """Clean up old recordings"""
    try:
        from utils.recording_manager import recording_manager
        
        days = int(request.form.get('days', 3))
        deleted_count = recording_manager.cleanup_old_recordings(days)
        return jsonify({'success': True, 'deleted_count': deleted_count})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/twilio_alerts')
def twilio_alerts():
    """Twilio alerts configuration page"""
    return render_template('alerts.html')

@app.route('/api/verify_recording/<filename>')
def verify_recording(filename):
    """Verify if a recording file is playable"""
    try:
        from utils.recording_manager import recording_manager
        
        filepath = os.path.join('static', 'recordings', filename)
        verification = recording_manager.verify_recording_playback(filepath)
        
        return jsonify({'success': True, 'verification': verification})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recognize_faces_with_motion', methods=['POST'])
def recognize_faces_with_motion():
    """
    Enhanced motion-gated face recognition with Twilio control
    """
    try:
        image_data = request.form.get('image_data')
        twilio_enabled = request.form.get('twilio_enabled', 'true').lower() == 'true'
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        global_recording_manager.add_frame(frame)
        
        # Get motion detector instance
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        # Get distance estimator with user settings
        from utils.distance_estimation import get_distance_estimator
        distance_estimator = get_distance_estimator()
        
        # STEP 1: Detect motion first
        motion_detected, detections, fg_mask, human_detected = motion_detector.detect_motion(frame)
        
        # STEP 2: Perform face recognition (unauthorized faces always detected)
        face_results = []
        motion_only_result = None
        
        # Always run face detection
        from utils.face_recognition_utils import detect_faces_with_motion_gate
        try:
            face_results = detect_faces_with_motion_gate(frame, motion_detector)
        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            # Fallback to basic face detection
            from utils.face_recognition_utils import detect_faces_in_frame
            face_results = detect_faces_in_frame(frame)
        
        # STEP 3: Check for motion without faces - RESPECT TWILIO SETTING
        if len(face_results) == 0 and (human_detected or motion_detector.is_human_motion_active()):
            print(f"🔴 Motion detected without faces - Twilio enabled: {twilio_enabled}")
            
            from utils.face_recognition_utils import detect_motion_without_faces
            motion_only_result = detect_motion_without_faces(frame, motion_detector, distance_estimator)
            
            # If motion without face detected, log it but respect Twilio setting
            if motion_only_result and motion_only_result['trigger_alert']:
                try:
                    screenshot_path = save_screenshot(
                        frame, 
                        f"motion_no_face_{motion_only_result['distance_meters']}m"
                    )
                    
                    # Log detection (Twilio calls handled based on frontend setting)
                    from utils.database import add_detection
                    add_detection(
                        "motion_detection", 
                        motion_only_result['person_name'],
                        float(motion_only_result['confidence']), 
                        screenshot_path, 
                        motion_only_result['alert_level']
                    )
                    
                    status_text = "WITH CALL" if twilio_enabled else "NO CALL (Blocked)"
                    print(f"✅ LOGGED MOTION-ONLY: {motion_only_result['person_name']} ({status_text})")
                    
                    # Send email alert only if Twilio is disabled
                    if not twilio_enabled:
                        send_motion_alert_async(motion_only_result, screenshot_path)
                    # If Twilio is enabled, the normal alert flow will handle it
                    
                except Exception as e:
                    print(f"❌ Error logging motion-only detection: {e}")
        
        # Ensure all data is JSON serializable
        clean_motion_only_result = None
        if motion_only_result:
            clean_motion_only_result = {
                'type': str(motion_only_result.get('type', '')),
                'distance_meters': float(motion_only_result.get('distance_meters', 0)),
                'distance_feet': float(motion_only_result.get('distance_feet', 0)),
                'confidence': float(motion_only_result.get('confidence', 0)),
                'bbox': [int(x) for x in motion_only_result.get('bbox', [])],
                'zone': str(motion_only_result.get('zone', '')),
                'trigger_alert': bool(motion_only_result.get('trigger_alert', False)),
                'alert_level': int(motion_only_result.get('alert_level', 0)),
                'within_detection_range': bool(motion_only_result.get('within_detection_range', False)),
                'motion_area': int(motion_only_result.get('motion_area', 0)),
                'person_name': 'Motion Detected, Unauthorized Person',
                'twilio_blocked': not twilio_enabled  # Add this flag
            }
        
        return jsonify({
            'success': True, 
            'faces': face_results,
            'motion_only_detection': clean_motion_only_result,
            'motion_detections': detections,
            'motion_status': {
                'human_motion_detected': human_detected,
                'motion_active': motion_detector.is_human_motion_active(),
                'face_detection_enabled': len(face_results) > 0 or motion_detector.is_human_motion_active(),
                'motion_only_alert': motion_only_result is not None,
                'twilio_enabled': twilio_enabled,  # Echo back the setting
                'distance_settings': {
                    'max_distance': float(distance_estimator.MAX_DETECTION_DISTANCE),
                    'warning_distance': float(distance_estimator.WARNING_DISTANCE),
                    'critical_distance': float(distance_estimator.CRITICAL_DISTANCE)
                }
            }
        })
        
    except Exception as e:
        print(f"Error in recognize_faces_with_motion: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/log_motion_detection', methods=['POST'])
def log_motion_detection():
    """Direct endpoint to log motion detection for testing"""
    try:
        # Create a simple black image for testing
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Save screenshot
        screenshot_path = save_screenshot(
            test_frame, 
            "test_motion"
        )
        
        # Add detection to database
        from utils.database import add_detection
        add_detection(
            "motion_detection", 
            "Motion Detected, Unauthorized Person",
            0.8,  # confidence
            screenshot_path, 
            2  # alert level
        )
        
        return jsonify({
            'success': True, 
            'message': 'Motion detection logged successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
    
@app.route('/api/get_detection_screenshots/<int:detection_id>')
def get_detection_screenshots(detection_id):
    """Get all screenshots from a detection event"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get the detection
        detection = conn.execute(
            'SELECT screenshot_path FROM detections WHERE id = ?', 
            (detection_id,)
        ).fetchone()
        
        conn.close()
        
        if detection and detection['screenshot_path']:
            screenshots = [detection['screenshot_path']]
        else:
            screenshots = []
        
        return jsonify({
            'success': True,
            'screenshots': screenshots,
            'detection_id': detection_id
        })
        
    except Exception as e:
        print(f"Error getting detection screenshots: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/register_known_person', methods=['POST'])
def register_known_person():
    """Register a face as a Known Person (from false alarm)"""
    try:
        name = request.form.get('name')
        detection_id = request.form.get('detection_id')
        selected_images = request.form.getlist('selected_images[]')
        
        print(f"📝 Registering Known Person: name={name}, detection_id={detection_id}, images={len(selected_images)}")
        
        if not name or not selected_images:
            return jsonify({'success': False, 'error': 'Name and images required'})
        
        # Process each selected image
        registered_count = 0
        from utils.database import add_known_person
        
        for image_path in selected_images:
            # Remove leading slash if present
            clean_path = image_path.lstrip('/')
            
            if os.path.exists(clean_path):
                try:
                    # Load image and extract face encoding
                    image = face_recognition.load_image_file(clean_path)
                    face_locations = face_recognition.face_locations(image)
                    
                    if len(face_locations) > 0:
                        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                        
                        # Save to known_persons database
                        add_known_person(name, face_encoding, clean_path, detection_id)
                        registered_count += 1
                        print(f"✅ Registered image: {clean_path}")
                    else:
                        print(f"⚠️ No face found in image: {clean_path}")
                        
                except Exception as e:
                    print(f"❌ Error processing image {clean_path}: {e}")
            else:
                print(f"❌ Image not found: {clean_path}")
        
        if registered_count == 0:
            return jsonify({
                'success': False, 
                'error': 'No faces could be extracted from selected images'
            })
        
        # Reload face recognition model to include known persons
        from utils.face_recognition_utils import load_known_faces, load_known_persons
        load_known_faces()
        load_known_persons()
        
        print(f"✅ Successfully registered {registered_count} images for {name}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully registered {registered_count} image(s) for {name} as Known Person',
            'registered_count': registered_count
        })
        
    except Exception as e:
        print(f"❌ Error in register_known_person: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get_known_persons')
def get_known_persons():
    """Get all registered known persons"""
    try:
        from utils.database import get_all_known_persons
        known_persons = get_all_known_persons()
        
        persons_list = []
        for person in known_persons:
            persons_list.append({
                'id': person['id'],
                'name': person['name'],
                'image_path': person['image_path'],
                'created_at': person['created_at'],
                'registered_from_false_alarm': bool(person['registered_from_false_alarm'])
            })
        
        return jsonify({
            'success': True, 
            'known_persons': persons_list,
            'total_count': len(persons_list)
        })
        
    except Exception as e:
        print(f"Error getting known persons: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/delete_known_person/<int:person_id>', methods=['DELETE'])
def delete_known_person(person_id):
    """Delete a known person"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get person details first
        person = conn.execute(
            'SELECT * FROM known_persons WHERE id = ?', 
            (person_id,)
        ).fetchone()
        
        if person:
            # Delete image file if it exists
            if os.path.exists(person['image_path']):
                os.remove(person['image_path'])
            
            # Delete from database
            conn.execute('DELETE FROM known_persons WHERE id = ?', (person_id,))
            conn.commit()
            
            # Reload known persons
            from utils.face_recognition_utils import load_known_persons
            load_known_persons()
            
            conn.close()
            return jsonify({'success': True, 'message': 'Known person deleted successfully'})
        else:
            conn.close()
            return jsonify({'success': False, 'error': 'Person not found'})
        
    except Exception as e:
        print(f"Error deleting known person: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/debug_detections')
def debug_detections():
    """Debug endpoint to see all detections in database"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get ALL detections
        all_detections = conn.execute('''
            SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10
        ''').fetchall()
        
        detections_list = []
        for detection in all_detections:
            detections_list.append({
                'id': detection['id'],
                'detection_type': detection['detection_type'],
                'person_name': detection['person_name'],
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp'],
                'alert_level': detection['alert_level']
            })
        
        conn.close()
        
        return jsonify({
            'success': True, 
            'total_detections': len(detections_list),
            'detections': detections_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def send_motion_alert_async(motion_info, screenshot_path):
    """Send alert for motion without face detection"""
    try:
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        message = f"Motion detected without face recognition at {motion_info['distance_meters']}m - Possible unauthorized person"
        
        alert_result = alert_system.send_alert(
            motion_info['alert_level'], 
            "motion_detection", 
            "Motion Detected, Unauthorized Person", 
            screenshot_path=screenshot_path,
            custom_message=message
        )
        print(f"Motion-only alert sent: {alert_result}")
    except Exception as e:
        print(f"Error sending motion-only alert: {e}")
    
@app.route('/api/motion_detection_status_detailed')
def motion_detection_status_detailed():
    """Get detailed motion detection status including human motion flag"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        stats = motion_detector.get_stats()
        
        # Add human motion status
        stats['human_motion_active'] = motion_detector.is_human_motion_active()
        stats['last_human_detection'] = motion_detector.last_human_detection_time.isoformat() if motion_detector.last_human_detection_time else None
        stats['recent_unauthorized_count'] = len(motion_detector.recent_unauthorized_faces)
        
        return jsonify({'success': True, 'status': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/clear_unauthorized_cache', methods=['POST'])
def clear_unauthorized_cache():
    """Manually clear the cache of recent unauthorized detections"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        cleared_count = len(motion_detector.recent_unauthorized_faces)
        motion_detector.recent_unauthorized_faces = {}
        
        return jsonify({
            'success': True, 
            'message': f'Cleared {cleared_count} cached unauthorized detections'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/debug_unauthorized_cache')
def debug_unauthorized_cache():
    """Debug endpoint to see current unauthorized face cache"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        cache_info = {
            'total_cached_faces': len(motion_detector.recent_unauthorized_faces),
            'cache_entries': []
        }
        
        current_time = datetime.datetime.now()
        for encoding_key, timestamp in motion_detector.recent_unauthorized_faces.items():
            elapsed = (current_time - timestamp).total_seconds()
            cache_info['cache_entries'].append({
                'encoding_key': str(encoding_key)[:20] + '...',
                'last_detected': timestamp.isoformat(),
                'elapsed_seconds': round(elapsed, 1),
                'status': 'ACTIVE' if elapsed < 600 else 'EXPIRED'
            })
        
        return jsonify({'success': True, 'cache_info': cache_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/register_face', methods=['POST'])
def register_face():
    """Register a new face from uploaded image or camera"""
    try:
        name = request.form.get('name')
        if not name:
            return jsonify({'success': False, 'error': 'Name is required'})
        
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(FACES_FOLDER, filename)
                file.save(filepath)
                
                image = face_recognition.load_image_file(filepath)
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) != 1:
                    os.remove(filepath)
                    return jsonify({'success': False, 'error': 'Please ensure exactly one face is visible in the image'})
                
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                
                from utils.database import add_face
                add_face(name, face_encoding, filepath)
                
                from utils.face_recognition_utils import load_known_faces
                load_known_faces()
                
                return jsonify({'success': True, 'message': f'Face registered successfully for {name}'})
        
        return jsonify({'success': False, 'error': 'No image provided'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_face', methods=['POST'])
def capture_face():
    """Capture face from camera for registration"""
    try:
        name = request.form.get('name')
        image_data = request.form.get('image_data')
        
        if not name or not image_data:
            return jsonify({'success': False, 'error': 'Name and image data required'})
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(FACES_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) != 1:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Please ensure exactly one face is visible'})
        
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        from utils.database import add_face
        add_face(name, face_encoding, filepath)
        
        from utils.face_recognition_utils import load_known_faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': f'Face registered successfully for {name}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_faces')
def get_faces():
    """Get all registered faces"""
    try:
        from utils.database import get_all_faces
        faces = get_all_faces()
        
        faces_list = []
        for face in faces:
            faces_list.append({
                'id': face['id'],
                'name': face['name'],
                'image_path': face['image_path'],
                'created_at': face['created_at']
            })
        
        return jsonify({'success': True, 'faces': faces_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_face/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Delete a registered face"""
    try:
        from utils.database import delete_face
        delete_face(face_id)
        
        # Reload known faces
        from utils.face_recognition_utils import load_known_faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': 'Face deleted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bulk_delete_faces', methods=['POST'])
def bulk_delete_faces():
    """Delete multiple faces at once"""
    try:
        face_ids = request.json.get('face_ids', [])
        if not face_ids:
            return jsonify({'success': False, 'error': 'No face IDs provided'})
        
        from utils.database import delete_face
        deleted_count = 0
        
        for face_id in face_ids:
            try:
                delete_face(face_id)
                deleted_count += 1
            except Exception as e:
        
                # Reload known faces
                from utils.face_recognition_utils import load_known_faces
                load_known_faces()
                
                return jsonify({
                    'success': True, 
                    'message': f'Successfully deleted {deleted_count} faces',
                    'deleted_count': deleted_count
                })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_face', methods=['POST'])
def update_face():
    """Update face information"""
    try:
        face_id = request.form.get('face_id')
        new_name = request.form.get('new_name')
        
        if not face_id or not new_name:
            return jsonify({'success': False, 'error': 'Face ID and new name required'})
        
        from utils.database import update_face_name
        result = update_face_name(face_id, new_name)
        
        if result:
            # Reload known faces
            from utils.face_recognition_utils import load_known_faces
            load_known_faces()
            return jsonify({'success': True, 'message': 'Face updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Face not found'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search_faces')
def search_faces():
    """Search faces by name"""
    try:
        query = request.args.get('query', '').strip()
        
        from utils.database import search_faces_by_name
        faces = search_faces_by_name(query)
        
        faces_list = []
        for face in faces:
            faces_list.append({
                'id': face['id'],
                'name': face['name'],
                'image_path': face['image_path'],
                'created_at': face['created_at']
            })
        
        return jsonify({'success': True, 'faces': faces_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_faces')
def export_faces():
    """Export face data"""
    try:
        from utils.database import get_all_faces
        faces = get_all_faces()
        
        export_data = []
        for face in faces:
            export_data.append({
                'name': face['name'],
                'created_at': face['created_at'],
                'image_path': face['image_path']
            })
        
        return jsonify({'success': True, 'faces': export_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/face_statistics')
def face_statistics():
    """Get face recognition statistics"""
    try:
        from utils.database import get_face_statistics
        stats = get_face_statistics()
        
        return jsonify({'success': True, 'statistics': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recognize_faces', methods=['POST'])
def recognize_faces():
    """Recognize faces in uploaded image"""
    try:
        image_data = request.form.get('image_data')
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Recognize faces on processed frame
        from utils.face_recognition_utils import detect_faces_in_frame
        results = detect_faces_in_frame(frame)
        
        # Convert numpy types to Python types for JSON serialization
        for result in results:
            result['confidence'] = float(result['confidence'])
            result['location'] = [int(x) for x in result['location']]
            result['face_covered'] = bool(result['face_covered'])
            result['is_authorized'] = bool(result['is_authorized'])
        
        return jsonify({'success': True, 'faces': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/recent_detections')
def recent_detections():
    """Get recent face and motion detections"""
    try:
        from utils.database import get_recent_detections
        detections = get_recent_detections()
        
        detections_list = []
        for detection in detections:
            detections_list.append({
                'id': detection['id'],
                'detection_type': detection['detection_type'],
                'person_name': detection['person_name'],
                'confidence': detection['confidence'],
                'screenshot_path': detection['screenshot_path'],
                'timestamp': detection['timestamp'],
                'alert_level': detection['alert_level']
            })
        
        return jsonify({'success': True, 'detections': detections_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_screenshot', methods=['POST'])
def save_screenshot():
    """Save screenshot from camera"""
    try:
        if 'screenshot' not in request.files:
            return jsonify({'success': False, 'error': 'No screenshot provided'})
        
        file = request.files['screenshot']
        filename = f"manual_screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(SCREENSHOTS_FOLDER, filename)
        file.save(filepath)
        
        return jsonify({'success': True, 'filepath': filepath})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_motion_detection', methods=['POST'])
def start_motion_detection():
    """Start motion detection on camera feed"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        # Start motion detection in background thread
        motion_detector.start_detection()
        return jsonify({'success': True, 'message': 'Motion detection started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_motion_detection', methods=['POST'])
def stop_motion_detection():
    """Stop motion detection"""
    try:
        global motion_detector
        if motion_detector:
            motion_detector.stop_detection()
        return jsonify({'success': True, 'message': 'Motion detection stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detect_motion', methods=['POST'])
def detect_motion():
    """Detect motion in uploaded frame"""
    try:
        image_data = request.form.get('image_data')
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect motion
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        # UPDATE THIS LINE - now expecting 4 return values
        motion_detected, detections, fg_mask, human_detected = motion_detector.detect_motion(frame)
        
        return jsonify({
            'success': True, 
            'motion_detected': motion_detected,
            'detections': detections,
            'human_detected': human_detected  # Optional: include this if needed
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_motion_model', methods=['POST'])
def train_motion_model():
    """Train motion detection model with collected data"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        result = motion_detector.train_model()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/collect_motion_data', methods=['POST'])
def collect_motion_data():
    """Collect training data for motion detection"""
    try:
        image_data = request.form.get('image_data')
        label = request.form.get('label')  # 'human', 'animal', 'object'
        
        if not image_data or not label:
            return jsonify({'success': False, 'error': 'Image data and label required'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save training data
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        result = motion_detector.collect_training_data(frame, label)
        return jsonify({'success': True, 'message': f'Training data collected for {label}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/motion_stats')
def motion_stats():
    """Get motion detection statistics"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        stats = motion_detector.get_stats()
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/send_alert', methods=['POST'])
def send_alert():
    """Send alert with specified level"""
    try:
        alert_level = int(request.form.get('alert_level', 1))
        detection_type = request.form.get('detection_type', 'manual')
        person_name = request.form.get('person_name')
        message = request.form.get('message', 'Security alert triggered')
        
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        result = alert_system.send_alert(alert_level, detection_type, person_name, message)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_email_settings', methods=['POST'])
def test_email_settings():
    """Test email configuration"""
    try:
        test_email = request.form.get('test_email')
        if not test_email:
            return jsonify({'success': False, 'error': 'Test email address required'})
        
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        result = alert_system.test_email_config(test_email)
        
        return jsonify({
            'success': 'successfully' in result.lower(),
            'result': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_alert_settings', methods=['POST'])
def update_alert_settings():
    """Update alert system settings with enhanced persistence"""
    try:
        settings = {
            'owner_email': request.form.get('owner_email'),
            'police_email': request.form.get('police_email'),
            'smtp_server': request.form.get('smtp_server'),
            'smtp_port': int(request.form.get('smtp_port', 587)),
            'email_user': request.form.get('email_user'),
            'email_password': request.form.get('email_password'),
            'enable_email': request.form.get('enable_email') == 'true',
            'enable_police_alerts': request.form.get('enable_police_alerts') == 'true'
        }
        
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        # Save settings and force reload
        result = alert_system.update_settings(settings)
        
        # Verify settings were saved by reloading
        updated_settings = alert_system.get_settings()
        
        return jsonify({
            'success': True, 
            'result': result, 
            'settings': updated_settings,
            'message': 'Settings saved and verified successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_alert_history')
def get_alert_history():
    """Get alert history"""
    try:
        limit = int(request.args.get('limit', 50))
        
        from utils.database import get_alert_history
        alerts = get_alert_history(limit)
        
        alerts_list = []
        for alert in alerts:
            alerts_list.append({
                'id': alert['id'],
                'alert_type': alert['alert_type'],
                'message': alert['message'],
                'sent_at': alert['sent_at'],
                'recipient': alert['recipient']
            })
        
        return jsonify({'success': True, 'alerts': alerts_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_alert_settings')
def get_alert_settings():
    """Get current alert settings"""
    try:
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        settings = alert_system.get_settings()
        return jsonify({'success': True, 'settings': settings})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trigger_emergency', methods=['POST'])
def trigger_emergency():
    """Trigger emergency protocol (Level 3 alert)"""
    try:
        reason = request.form.get('reason', 'Emergency protocol activated manually')
        
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        result = alert_system.trigger_emergency_protocol(reason)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start video recording"""
    try:
        global recording_manager
        if recording_manager is None:
            from utils.recording_utils import RecordingManager
            recording_manager = RecordingManager()
        
        result = recording_manager.start_recording()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop video recording"""
    try:
        global recording_manager
        if recording_manager is None:
            from utils.recording_utils import RecordingManager
            recording_manager = RecordingManager()
        
        result = recording_manager.stop_recording()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_recordings')
def get_recordings():
    """Get list of all recordings with enhanced metadata"""
    try:
        from utils.database import get_all_recordings
        recordings = get_all_recordings()
        
        recordings_list = []
        for recording in recordings:
            file_exists = os.path.exists(recording['file_path']) if recording['file_path'] else False
            
            recordings_list.append({
                'id': recording['id'],
                'file_path': recording['file_path'],
                'filename': os.path.basename(recording['file_path']) if recording['file_path'] else 'Unknown',
                'start_time': recording['start_time'],
                'end_time': recording['end_time'],
                'duration': recording['duration'],
                'duration_formatted': format_duration(recording['duration']) if recording['duration'] else '0:00',
                'file_size': recording.get('file_size', 0),
                'file_size_formatted': format_file_size(recording.get('file_size', 0)),
                'file_exists': file_exists,
                'playback_url': f"/static/recordings/{os.path.basename(recording['file_path'])}" if file_exists else None,
                'thumbnail_url': generate_video_thumbnail(recording['file_path']) if file_exists else None
            })
        
        return jsonify({'success': True, 'recordings': recordings_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_recording/<int:recording_id>', methods=['DELETE'])
def delete_recording(recording_id):
    """Delete a recording"""
    try:
        from utils.database import delete_recording
        result = delete_recording(recording_id)
        
        if result:
            return jsonify({'success': True, 'message': 'Recording deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Recording not found'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_screenshots')
def get_screenshots():
    """Get list of all screenshots with enhanced metadata for gallery view"""
    try:
        from utils.database import get_all_screenshots
        screenshots = get_all_screenshots()
        
        screenshots_list = []
        for screenshot in screenshots:
            file_exists = os.path.exists(screenshot['file_path']) if screenshot['file_path'] else False
            
            screenshots_list.append({
                'id': screenshot['id'],
                'file_path': screenshot['file_path'],
                'filename': os.path.basename(screenshot['file_path']) if screenshot['file_path'] else 'Unknown',
                'timestamp': screenshot['timestamp'],
                'detection_type': screenshot.get('detection_type', 'manual'),
                'description': screenshot.get('description', 'Screenshot'),
                'file_size': screenshot.get('file_size', 0),
                'file_size_formatted': format_file_size(screenshot.get('file_size', 0)),
                'file_exists': file_exists,
                'thumbnail_url': f"/static/screenshots/{os.path.basename(screenshot['file_path'])}" if file_exists else None,
                'view_url': f"/static/screenshots/{os.path.basename(screenshot['file_path'])}" if file_exists else None
            })
        
        return jsonify({'success': True, 'screenshots': screenshots_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_screenshot/<int:screenshot_id>', methods=['DELETE'])
def delete_screenshot(screenshot_id):
    """Delete a screenshot"""
    try:
        from utils.database import delete_screenshot
        result = delete_screenshot(screenshot_id)
        
        if result:
            return jsonify({'success': True, 'message': 'Screenshot deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Screenshot not found'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_screenshot', methods=['POST'])
def capture_screenshot():
    """Capture screenshot from live camera"""
    try:
        description = request.form.get('description', 'Manual screenshot')
        
        global recording_manager
        if recording_manager is None:
            from utils.recording_utils import RecordingManager
            recording_manager = RecordingManager()
        
        result = recording_manager.capture_screenshot(description)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recording_status')
def recording_status():
    """Get current recording status"""
    try:
        global recording_manager
        if recording_manager is None:
            from utils.recording_utils import RecordingManager
            recording_manager = RecordingManager()
        
        status = recording_manager.get_status()
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/storage_stats')
def storage_stats():
    """Get storage statistics"""
    try:
        global recording_manager
        if recording_manager is None:
            from utils.recording_utils import RecordingManager
            recording_manager = RecordingManager()
        
        stats = recording_manager.get_storage_stats()
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleanup_old_files', methods=['POST'])
def cleanup_old_files():
    """Clean up old recordings and screenshots"""
    try:
        days = int(request.form.get('days', 30))
        
        global recording_manager
        if recording_manager is None:
            from utils.recording_utils import RecordingManager
            recording_manager = RecordingManager()
        
        result = recording_manager.cleanup_old_files(days)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera_control', methods=['POST'])
def camera_control():
    """Control camera pan/tilt movements"""
    try:
        action = request.form.get('action')  # 'pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'center'
        speed = int(request.form.get('speed', 5))  # 1-10 speed scale
        
        from utils.cctv_utils import CCTVController
        controller = CCTVController()
        
        result = controller.move_camera(action, speed)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera_preset', methods=['POST'])
def camera_preset():
    """Save or load camera position preset"""
    try:
        action = request.form.get('action')  # 'save' or 'load'
        preset_name = request.form.get('preset_name')
        
        if not preset_name:
            return jsonify({'success': False, 'error': 'Preset name required'})
        
        from utils.cctv_utils import CCTVController
        controller = CCTVController()
        
        if action == 'save':
            result = controller.save_preset(preset_name)
        elif action == 'load':
            result = controller.load_preset(preset_name)
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_presets')
def get_presets():
    """Get all saved camera presets"""
    try:
        from utils.cctv_utils import CCTVController
        controller = CCTVController()
        
        presets = controller.get_presets()
        return jsonify({'success': True, 'presets': presets})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_preset/<preset_name>', methods=['DELETE'])
def delete_preset(preset_name):
    """Delete a camera preset"""
    try:
        from utils.cctv_utils import CCTVController
        controller = CCTVController()
        
        result = controller.delete_preset(preset_name)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera_settings', methods=['GET', 'POST'])
def camera_settings():
    """Get or update camera settings"""
    try:
        from utils.cctv_utils import CCTVController
        controller = CCTVController()
        
        if request.method == 'GET':
            settings = controller.get_camera_settings()
            return jsonify({'success': True, 'settings': settings})
        
        elif request.method == 'POST':
            settings = {
                'brightness': int(request.form.get('brightness', 50)),
                'contrast': int(request.form.get('contrast', 50)),
                'saturation': int(request.form.get('saturation', 50)),
                'zoom': int(request.form.get('zoom', 100)),
                'night_vision': request.form.get('night_vision') == 'true',
                'auto_tracking': request.form.get('auto_tracking') == 'true'
            }
            
            result = controller.update_camera_settings(settings)
            return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_audio_recording', methods=['POST'])
def start_audio_recording():
    """Start audio recording"""
    try:
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        
        result = audio_manager.start_recording()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_audio_recording', methods=['POST'])
def stop_audio_recording():
    """Stop audio recording"""
    try:
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        
        result = audio_manager.stop_recording()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/play_audio_message', methods=['POST'])
def play_audio_message():
    """Play pre-recorded audio message through speakers"""
    try:
        message_type = request.form.get('message_type')  # 'warning', 'greeting', 'custom'
        custom_text = request.form.get('custom_text')
        
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        
        result = audio_manager.play_message(message_type, custom_text)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_two_way_audio', methods=['POST'])
def start_two_way_audio():
    """Start two-way audio communication"""
    try:
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        
        result = audio_manager.start_two_way_communication()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_two_way_audio', methods=['POST'])
def stop_two_way_audio():
    """Stop two-way audio communication"""
    try:
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        
        result = audio_manager.stop_two_way_communication()
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_audio_recordings')
def get_audio_recordings():
    """Get list of audio recordings"""
    try:
        from utils.database import get_audio_recordings
        recordings = get_audio_recordings()
        
        recordings_list = []
        for recording in recordings:
            recordings_list.append({
                'id': recording['id'],
                'file_path': recording['file_path'],
                'duration': recording['duration'],
                'timestamp': recording['timestamp'],
                'file_size': recording.get('file_size', 0)
            })
        
        return jsonify({'success': True, 'recordings': recordings_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_audio_recording/<int:recording_id>', methods=['DELETE'])
def delete_audio_recording(recording_id):
    """Delete an audio recording"""
    try:
        from utils.database import delete_audio_recording
        result = delete_audio_recording(recording_id)
        
        if result:
            return jsonify({'success': True, 'message': 'Audio recording deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Audio recording not found'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/audio_status')
def audio_status():
    """Get current audio system status"""
    try:
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        
        status = audio_manager.get_status()
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_multiple_faces', methods=['POST'])
def capture_multiple_faces():
    """Capture multiple faces for better training"""
    try:
        name = request.form.get('name')
        image_data = request.form.get('image_data')
        
        if not name or not image_data:
            return jsonify({'success': False, 'error': 'Name and image data required'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Find faces in the image
        import face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) != 1:
            return jsonify({'success': False, 'error': 'Please ensure exactly one face is visible'})
        
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        # Save image with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(FACES_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        
        # Save to database
        from utils.database import add_face
        add_face(name, face_encoding, filepath)
        
        # Reload known faces
        from utils.face_recognition_utils import load_known_faces
        load_known_faces()
        
        return jsonify({
            'success': True, 
            'message': f'Face captured for {name}',
            'image_path': filepath,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/finalize_face_registration', methods=['POST'])
def finalize_face_registration():
    """Finalize face registration and retrain model"""
    try:
        # Reload known faces to include all new captures
        from utils.face_recognition_utils import load_known_faces
        load_known_faces()
        
        return jsonify({'success': True, 'message': 'Face registration completed and model updated'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dashboard_feed')
def dashboard_feed():
    """Get unified dashboard feed with face and motion detection"""
    try:
        from utils.database import get_recent_detections
        
        # Get more detections to filter through
        detections = get_recent_detections(50)
        
        # DEBUG: Print all detections to see what we're working with
        for i, detection in enumerate(detections):
        
            recent_detections = []
            for detection in detections:
                # Include face detections where person is unauthorized/unknown
                if detection['detection_type'] == 'face_detection':
                    # More flexible matching for unauthorized faces
                    person_name = detection['person_name']
                    is_unauthorized = (
                        person_name is None or 
                        person_name == 'Unknown' or 
                        person_name == 'Unauthorized' or
                        'unauthorized' in str(person_name).lower()
                    )
                
                if is_unauthorized:
                    recent_detections.append({
                        'id': detection['id'],
                        'type': detection['detection_type'],
                        'person_name': 'Unauthorized',
                        'confidence': detection['confidence'],
                        'timestamp': detection['timestamp'],
                        'screenshot_path': detection['screenshot_path'],
                        'alert_level': detection['alert_level']
                    })
                # Include ALL motion detections
                elif detection['detection_type'] == 'motion_detection':
                    recent_detections.append({
                        'id': detection['id'],
                        'type': detection['detection_type'],
                        'person_name': detection['person_name'] or 'motion',
                        'confidence': detection['confidence'],
                        'timestamp': detection['timestamp'],
                        'screenshot_path': detection['screenshot_path'],
                        'alert_level': detection['alert_level']
                    })
        
        # Return only the 20 most recent
        recent_detections = recent_detections[:20]
        
        
        return jsonify({'success': True, 'detections': recent_detections})
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dashboard_action', methods=['POST'])
def dashboard_action():
    """Handle dashboard quick actions"""
    try:
        action = request.form.get('action')
        
        if action == 'start_motion_detection':
            global motion_detector
            if motion_detector is None:
                from utils.motion_detection_utils import MotionDetector
                motion_detector = MotionDetector()
            result = motion_detector.start_detection()
            return jsonify({'success': True, 'result': result})
            
        elif action == 'stop_motion_detection':
            if motion_detector:
                result = motion_detector.stop_detection()
                return jsonify({'success': True, 'result': result})
            return jsonify({'success': False, 'error': 'Motion detector not initialized'})
            
        elif action == 'start_face_recognition':
            from utils.face_recognition_utils import load_known_faces
            load_known_faces()
            return jsonify({'success': True, 'result': 'Face recognition started'})
            
        elif action == 'start_recording':
            return start_recording()
            
        elif action == 'stop_recording':
            return stop_recording()
            
        elif action == 'play_message':
            message_type = request.form.get('message_type', 'warning')
            custom_text = request.form.get('custom_text')
            
            from utils.audio_utils import AudioManager
            audio_manager = AudioManager()
            result = audio_manager.play_message(message_type, custom_text)
            return jsonify({'success': True, 'result': result})
            
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/unauthorized_detections')
def unauthorized_detections():
    """Get all security-related detections - INCLUDES AUTHORIZED AND KNOWN FACES"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get ALL recent detections including authorized and known faces
        all_detections = conn.execute('''
            SELECT 
                d.*,
                CASE 
                    WHEN d.detection_type = 'face_detection' AND d.person_name IN (SELECT name FROM faces) THEN 'authorized'
                    WHEN d.detection_type = 'face_detection' AND d.person_name IN (SELECT name FROM known_persons) THEN 'known'
                    WHEN d.detection_type = 'face_detection' AND (d.person_name = 'Unauthorized' OR d.person_name LIKE 'Unauthorized (%') THEN 'unauthorized'
                    WHEN d.detection_type = 'motion_detection' THEN 'motion'
                    ELSE 'other'
                END as category
            FROM detections d 
            WHERE d.detection_type IN ('face_detection', 'motion_detection')
            ORDER BY d.timestamp DESC 
            LIMIT 50
        ''').fetchall()
        
        detections_list = []
        for detection in all_detections:
            detection_dict = dict(detection)
            
            # For debugging, print every detection
            print(f"📋 DETECTION: type={detection_dict['detection_type']}, name='{detection_dict['person_name']}', category={detection_dict['category']}")
            
            # Include ALL detections in the results
            detections_list.append({
                'id': detection_dict['id'],
                'type': detection_dict['detection_type'],
                'person_name': detection_dict['person_name'],
                'confidence': detection_dict['confidence'],
                'timestamp': detection_dict['timestamp'],
                'screenshot_path': detection_dict['screenshot_path'],
                'alert_level': detection_dict['alert_level'],
                'category': detection_dict['category']
            })
        
        print(f"📊 Returning {len(detections_list)} total detections")
        
        conn.close()
        return jsonify({'success': True, 'detections': detections_list})
        
    except Exception as e:
        print(f"❌ ERROR in unauthorized_detections: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
def save_screenshot(frame, filename_prefix):
    """Save screenshot to screenshots folder"""
    try:
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.jpg"
        filepath = os.path.join(SCREENSHOTS_FOLDER, filename)
        
        # Save the frame as image
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        print(f"✅ Screenshot saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving screenshot: {e}")
        return None

@app.route('/api/register_face_multiple', methods=['POST'])
def register_face_multiple():
    """Register multiple face images for better recognition"""
    try:
        name = request.form.get('name')
        if not name:
            return jsonify({'success': False, 'error': 'Name is required'})
        
        # Start multiple face capture
        from utils.face_recognition_utils import register_multiple_faces_from_camera, register_face_multiple_images
        
        captured_faces = register_multiple_faces_from_camera(name)
        
        if captured_faces is None:
            return jsonify({'success': False, 'error': 'Face capture cancelled'})
        
        if len(captured_faces) == 0:
            return jsonify({'success': False, 'error': 'No faces captured'})
        
        # Register all captured faces
        result = register_face_multiple_images(name, captured_faces)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_face_image/<int:face_id>/<int:image_index>', methods=['DELETE'])
def delete_face_image_api(face_id, image_index):
    """Delete a specific image from face registration"""
    try:
        from utils.face_recognition_utils import delete_face_image
        result = delete_face_image(face_id, image_index)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
        
        return jsonify({'success': True, 'message': result['message']})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_face_album')
def get_face_album():
    """Get face album with privacy protection"""
    try:
        from utils.database import get_faces_grouped_by_name
        from utils.face_recognition_utils import create_privacy_thumbnail
        
        faces = get_faces_grouped_by_name()
        
        # Create privacy thumbnails for album view
        for face in faces:
            if face['thumbnail_path']:
                privacy_thumbnail = create_privacy_thumbnail(face['thumbnail_path'], blur_faces=True)
                face['privacy_thumbnail'] = privacy_thumbnail
        
        return jsonify({'success': True, 'faces': faces})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_face_images/<name>')
def get_face_images_api(name):
    """Get all images for a specific person (for modal view)"""
    try:
        from utils.database import get_all_faces
        
        # Get all faces for this person
        all_faces = get_all_faces()
        person_faces = [face for face in all_faces if face['name'] == name]
        
        face_images = []
        for face in person_faces:
            face_images.append({
                'id': face['id'],
                'image_path': face['image_path'],
                'created_at': face['created_at']
            })
        
        return jsonify({'success': True, 'images': face_images})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/playback_search')
def playback_search():
    """Search recordings and screenshots by date range and type"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        media_type = request.args.get('type', 'all')  # 'recordings', 'screenshots', 'all'
        detection_type = request.args.get('detection_type')  # 'face_detection', 'motion_detection', etc.
        
        from utils.database import search_media_by_criteria
        results = search_media_by_criteria(start_date, end_date, media_type, detection_type)
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/playback_timeline')
def playback_timeline():
    """Get timeline data for recordings and events"""
    try:
        date = request.args.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
        
        from utils.database import get_timeline_data
        timeline_data = get_timeline_data(date)
        
        return jsonify({'success': True, 'timeline': timeline_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/create_video_clip', methods=['POST'])
def create_video_clip():
    """Create a video clip from a recording with start and end times"""
    try:
        recording_id = request.form.get('recording_id')
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        clip_name = request.form.get('clip_name', 'clip')
        
        from utils.recording_utils import RecordingManager
        recording_manager = RecordingManager()
        
        result = recording_manager.create_video_clip(recording_id, start_time, end_time, clip_name)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_media', methods=['POST'])
def export_media():
    """Export selected recordings and screenshots"""
    try:
        media_ids = request.json.get('media_ids', [])
        media_type = request.json.get('type', 'recordings')  # 'recordings' or 'screenshots'
        export_format = request.json.get('format', 'zip')  # 'zip' or 'folder'
        
        from utils.recording_utils import RecordingManager
        recording_manager = RecordingManager()
        
        result = recording_manager.export_media(media_ids, media_type, export_format)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/playback_stats')
def playback_stats():
    """Get comprehensive playback and storage statistics"""
    try:
        from utils.database import get_playback_statistics
        stats = get_playback_statistics()
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/motion_detection_status')
def motion_detection_status():
    """Get current motion detection status"""
    try:
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        stats = motion_detector.get_stats()
        return jsonify({'success': True, 'status': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_training_data', methods=['POST'])
def clear_training_data():
    """Clear motion detection training data"""
    try:
        category = request.form.get('category')  # 'human', 'animal', or None for all
        
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        result = motion_detector.clear_training_data(category)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/upload_training_images', methods=['POST'])
def upload_training_images():
    """Upload training images for motion detection"""
    try:
        label = request.form.get('label')  # 'human' or 'animal'
        
        if not label or label not in ['human', 'animal']:
            return jsonify({'success': False, 'error': 'Valid label (human/animal) required'})
        
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No images provided'})
        
        files = request.files.getlist('images')
        uploaded_count = 0
        
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        for file in files:
            if file.filename != '':
                result = motion_detector.upload_training_image(file, label)
                if result['success']:
                    uploaded_count += 1
        
        return jsonify({
            'success': True, 
            'message': f'Successfully uploaded {uploaded_count} images for {label}',
            'uploaded_count': uploaded_count
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_training_images')
def get_training_images():
    """Get all training images"""
    try:
        category = request.args.get('category')  # 'human', 'animal', or None for all
        
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        images = motion_detector.get_training_images(category)
        return jsonify({'success': True, 'images': images})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_training_image', methods=['DELETE'])
def delete_training_image():
    """Delete a training image"""
    try:
        filename = request.json.get('filename')
        category = request.json.get('category')
        
        if not filename or not category:
            return jsonify({'success': False, 'error': 'Filename and category required'})
        
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        result = motion_detector.delete_training_image(filename, category)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bulk_upload_training', methods=['POST'])
def bulk_upload_training():
    """Bulk upload training images from a folder or multiple files"""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'})
        
        files = request.files.getlist('files')
        label = request.form.get('label')  # 'human' or 'animal'
        
        if not label or label not in ['human', 'animal']:
            return jsonify({'success': False, 'error': 'Valid label (human/animal) required'})
        
        global motion_detector
        if motion_detector is None:
            from utils.motion_detection_utils import MotionDetector
            motion_detector = MotionDetector()
        
        uploaded_count = 0
        failed_count = 0
        
        for file in files:
            if file.filename != '' and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    result = motion_detector.upload_training_image(file, label)
                    if result['success']:
                        uploaded_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
        
        return jsonify({
            'success': True,
            'uploaded_count': uploaded_count,
            'failed_count': failed_count,
            'message': f'Uploaded {uploaded_count} images, {failed_count} failed'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/calibrate_distance', methods=['POST'])
def calibrate_distance():
    """Calibrate distance estimation for your camera"""
    try:
        known_distance = float(request.form.get('known_distance'))  # in meters
        face_width_pixels = float(request.form.get('face_width_pixels'))
        
        distance_estimator = get_distance_estimator()
        
        # Convert to cm for calibration
        known_distance_cm = known_distance * 100
        
        focal_length = distance_estimator.calibrate_focal_length(known_distance_cm, face_width_pixels)
        
        return jsonify({
            'success': True,
            'focal_length': focal_length,
            'message': f'Camera calibrated successfully. Focal length: {focal_length:.2f}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/update_distance_settings', methods=['POST'])
def update_distance_settings():
    """Update distance detection settings"""
    try:
        max_distance = float(request.form.get('max_distance', 6.0))
        
        # Calculate warning and critical distances based on max_distance
        warning_distance = max_distance * 0.5   # 50% of max distance
        critical_distance = max_distance * 0.25  # 25% of max distance
        
        distance_estimator = get_distance_estimator()
        distance_estimator.update_settings(
            max_distance=max_distance,
            warning_distance=warning_distance,
            critical_distance=critical_distance
        )
        
        return jsonify({
            'success': True,
            'message': 'Distance settings updated successfully',
            'settings': {
                'max_distance': max_distance,
                'warning_distance': warning_distance,
                'critical_distance': critical_distance
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get_distance_settings')
def get_distance_settings():
    """Get current distance estimation settings"""
    try:
        distance_estimator = get_distance_estimator()
        
        return jsonify({
            'success': True,
            'settings': {
                'max_detection_distance': distance_estimator.MAX_DETECTION_DISTANCE,
                'warning_distance': distance_estimator.WARNING_DISTANCE,
                'critical_distance': distance_estimator.CRITICAL_DISTANCE,
                'focal_length': distance_estimator.focal_length,
                'known_face_width_cm': distance_estimator.KNOWN_FACE_WIDTH_CM
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/estimate_distance', methods=['POST'])
def estimate_distance():
    """Estimate distance for a test image"""
    try:
        image_data = request.form.get('image_data')
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces and estimate distance
        from utils.face_recognition_utils import detect_faces_in_frame_optimized
        results = detect_faces_in_frame_optimized(frame)
        
        # Convert numpy types to Python types for JSON serialization
        for result in results:
            result['confidence'] = float(result['confidence'])
            result['location'] = [int(x) for x in result['location']]
            result['face_covered'] = bool(result['face_covered'])
            result['is_authorized'] = bool(result['is_authorized'])
            result['distance_meters'] = float(result['distance_meters'])
            result['distance_feet'] = float(result['distance_feet'])
            result['trigger_alert'] = bool(result['trigger_alert'])
            result['alert_level'] = int(result['alert_level'])
            result['within_detection_range'] = bool(result['within_detection_range'])
        
        return jsonify({'success': True, 'faces': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    from utils.database import init_db
    from utils.face_recognition_utils import load_known_faces, load_known_persons
    from utils.motion_detection_utils import MotionDetector
    from utils.recording_utils import RecordingManager
    
    # Initialize database
    init_db()
    
    # Load known faces (Authorized Persons)
    print("📋 Loading authorized persons...")
    load_known_faces()
    
    # Load known persons (Non-threatening)
    print("📋 Loading known persons...")
    load_known_persons()
    
    # Initialize motion detector
    motion_detector = MotionDetector()
    
    # Initialize recording manager
    recording_manager = RecordingManager()
    
    print("✅ System initialized successfully!")
    print(f"   - Authorized Persons: {len(known_face_names)}")
    print(f"   - Known Persons: {len(known_persons_names)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

def format_duration(seconds):
    """Format duration in seconds to MM:SS or HH:MM:SS"""
    if not seconds:
        return "0:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def format_file_size(bytes_size):
    """Format file size in bytes to human readable format"""
    if bytes_size == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    
    return f"{bytes_size:.1f} TB"

def generate_video_thumbnail(video_path):
    """Generate thumbnail for video file"""
    try:
        if not os.path.exists(video_path):
            return None
        
        # Create thumbnails directory
        thumbnails_dir = THUMBNAILS_FOLDER # Use the defined constant
        os.makedirs(thumbnails_dir, exist_ok=True)
        
        # Generate thumbnail filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        thumbnail_path = os.path.join(thumbnails_dir, f"{video_name}_thumb.jpg")
        
        # Check if thumbnail already exists
        if os.path.exists(thumbnail_path):
            return f"/static/thumbnails/{video_name}_thumb.jpg"
        
        # Generate thumbnail using OpenCV
        cap = cv2.VideoCapture(video_path)
        
        # Get frame from middle of video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to thumbnail size
                height, width = frame.shape[:2]
                max_size = 200
                
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                thumbnail = cv2.resize(frame, (new_width, new_height))
                cv2.imwrite(thumbnail_path, thumbnail)
                
                cap.release()
                return f"/static/thumbnails/{video_name}_thumb.jpg"
        
        cap.release()
        return None
        
    except Exception as e:
        return None
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
