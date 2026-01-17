import os
import time
import uuid
import sqlite3
from datetime import datetime, timedelta
import datetime as dt
from threading import Thread, Lock
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import logging
import re
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TwilioAlertSystem:
    def __init__(self):
        self.setup_logging()
        
        self.client = None
        self.test_mode = False
        self.primary_number = None
        self.backup_number = None
        self.twilio_number = None
        self.account_sid = None
        self.auth_token = None
        
        # Phone numbers loaded per user
        self.user_phone_numbers = {}
        
        # ✅ FACE TRACKING: Track faces that triggered alerts - PER USER
        self.user_alerted_faces = {}  # {user_id: {face_hash: data}}
        self.user_answered_calls = {}  # {user_id: set(face_hashes)}
        
        self.alert_cooldown = 300  # 5 minutes (300 seconds)
        self.max_alerts_per_face = 3  # Maximum 3 calls per face
        
        # ✅ FACE GROUPING: For better face recognition
        self.face_grouping_threshold = 0.7  # 70% similarity to group faces
        self.similarity_threshold = 0.85  # 85% similarity to consider same face
        
        # ✅ THREAD SAFETY: Lock for concurrent access
        self.lock = Lock()
        
        # Don't load settings globally - load per user as needed
        self.user_settings = {}
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('twilio_alerts.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_settings(self, user_id):
        """Load Twilio settings for a specific user"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS twilio_settings (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    account_sid TEXT,
                    auth_token TEXT,
                    twilio_number TEXT,
                    test_mode BOOLEAN DEFAULT 1,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id)
                )
            ''')
            
            settings = cursor.execute('SELECT * FROM twilio_settings WHERE user_id = ?', (user_id,)).fetchone()
            
            if settings:
                user_settings = {
                    'account_sid': settings[2],
                    'auth_token': settings[3],
                    'twilio_number': settings[4],
                    'test_mode': bool(settings[5]),
                    'user_id': user_id
                }
                
                self.user_settings[user_id] = user_settings
                
                # Only initialize Twilio client if we have valid credentials
                if user_settings['account_sid'] and user_settings['auth_token'] and user_settings['twilio_number']:
                    try:
                        client = Client(user_settings['account_sid'], user_settings['auth_token'])
                        # Test the client by making a simple API call
                        client.api.accounts(user_settings['account_sid']).fetch()
                        user_settings['client'] = client
                        self.logger.info(f"✅ Twilio client initialized for user {user_id}")
                    except Exception as e:
                        self.logger.error(f"❌ Twilio authentication failed for user {user_id}: {e}")
                        user_settings['client'] = None
                        user_settings['test_mode'] = True
                else:
                    user_settings['client'] = None
                    user_settings['test_mode'] = True
                    self.logger.info(f"Twilio credentials incomplete for user {user_id} - using test mode")
            else:
                # Create default settings for this user
                user_settings = {
                    'account_sid': None,
                    'auth_token': None,
                    'twilio_number': None,
                    'test_mode': True,
                    'client': None,
                    'user_id': user_id
                }
                self.user_settings[user_id] = user_settings
                self.logger.info(f"No settings found for user {user_id} - using test mode")
            
            conn.close()
            return user_settings
            
        except Exception as e:
            self.logger.error(f"Error loading settings for user {user_id}: {e}")
            user_settings = {
                'account_sid': None,
                'auth_token': None,
                'twilio_number': None,
                'test_mode': True,
                'client': None,
                'user_id': user_id
            }
            self.user_settings[user_id] = user_settings
            return user_settings
    
    def _get_user_settings(self, user_id):
        """Get user settings, loading if necessary"""
        if user_id not in self.user_settings:
            return self.load_settings(user_id)
        return self.user_settings[user_id]
    
    def load_phone_numbers(self, user_id):
        """Load phone numbers for a specific user"""
        try:
            from utils.database import get_all_phone_numbers
            numbers = get_all_phone_numbers(user_id)
            
            user_numbers = []
            for number in numbers:
                if number['is_active']:
                    user_numbers.append({
                        'id': number['id'],
                        'phone_number': number['phone_number'],
                        'display_name': number['display_name'],
                        'sort_order': number['sort_order']
                    })
            
            # Sort by sort_order
            user_numbers.sort(key=lambda x: x['sort_order'])
            
            self.user_phone_numbers[user_id] = user_numbers
            
            self.logger.info(f"✅ Loaded {len(user_numbers)} active phone numbers for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error loading phone numbers for user {user_id}: {e}")
            self.user_phone_numbers[user_id] = []
    
    def get_active_phone_numbers(self, user_id):
        """Get list of active phone numbers for a user"""
        if user_id not in self.user_phone_numbers:
            self.load_phone_numbers(user_id)
        
        return [num['phone_number'] for num in self.user_phone_numbers.get(user_id, [])]

    def _get_face_hash(self, face_encoding):
        """Create unique hash from face encoding"""
        if face_encoding is None:
            return None
        
        # Use first 30 values for hash with good precision
        encoding_str = ','.join([f"{x:.6f}" for x in face_encoding[:30]])
        return hashlib.md5(encoding_str.encode()).hexdigest()
    
    def _find_similar_face(self, user_id, new_encoding):
        """Find if a similar face already exists in tracking for a user"""
        if user_id not in self.user_alerted_faces:
            return None
            
        user_faces = self.user_alerted_faces[user_id]
        if not user_faces:
            return None
            
        for face_hash, face_data in user_faces.items():
            stored_encoding = face_data.get('encoding')
            if stored_encoding is None:
                continue
                
            # Calculate cosine similarity between encodings
            similarity = self._calculate_similarity(stored_encoding, new_encoding)
            
            if similarity >= self.similarity_threshold:
                print(f"✅ FACE MATCH for user {user_id}: {face_hash[:8]} - Similarity: {similarity:.3f}")
                return face_hash
                
        return None
    
    def _calculate_similarity(self, encoding1, encoding2):
        """Calculate cosine similarity between two face encodings"""
        try:
            # Convert to numpy arrays if they aren't already
            enc1 = np.array(encoding1).reshape(1, -1)
            enc2 = np.array(encoding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(enc1, enc2)[0][0]
            return max(0.0, min(1.0, similarity))  # Ensure between 0-1
        except Exception as e:
            print(f"❌ Similarity calculation error: {e}")
            return 0.0
    
    def should_trigger_alert_for_face(self, user_id, face_encoding):
        """
        ✅ MAIN LOGIC: Check if we should trigger alert for this face for a specific user
        Returns: True if alert should be triggered, False if duplicate/cooldown
        """
        user_settings = self._get_user_settings(user_id)
        if user_settings.get('test_mode', True):
            self.logger.info(f"TEST MODE for user {user_id}: Allowing alert")
            return True
        
        if face_encoding is None:
            self.logger.warning(f"No face encoding provided for user {user_id} - denying alert")
            return False
        
        current_time = time.time()
        
        with self.lock:  # Thread-safe access
            # Initialize user tracking if not exists
            if user_id not in self.user_alerted_faces:
                self.user_alerted_faces[user_id] = {}
            if user_id not in self.user_answered_calls:
                self.user_answered_calls[user_id] = set()
            
            user_faces = self.user_alerted_faces[user_id]
            user_answered = self.user_answered_calls[user_id]
            
            # ✅ STEP 1: Check if similar face already exists
            similar_face_hash = self._find_similar_face(user_id, face_encoding)
            
            if similar_face_hash:
                # Use the existing face ID
                face_data = user_faces[similar_face_hash]
                face_hash = similar_face_hash
                print(f"🔄 USING EXISTING FACE for user {user_id}: {face_hash[:8]}")
            else:
                # Create new face ID
                face_hash = self._get_face_hash(face_encoding)
                print(f"🆕 NEW FACE DETECTED for user {user_id}: {face_hash[:8]}")
            
            # ✅ CHECK 1: Has this face been answered already?
            if face_hash in user_answered:
                self.logger.info(f"✅ ANSWERED: Face {face_hash[:8]} for user {user_id} - Call was answered, no more alerts")
                return False
            
            # ✅ CHECK 2: Face tracking logic
            if face_hash in user_faces:
                face_data = user_faces[face_hash]
                
                # Cooldown period
                time_since_last_alert = current_time - face_data['last_alert_time']
                if time_since_last_alert < self.alert_cooldown:
                    remaining = self.alert_cooldown - time_since_last_alert
                    self.logger.info(f"🔄 COOLDOWN: Face {face_hash[:8]} for user {user_id} - {int(remaining)}s remaining of {self.alert_cooldown}s")
                    return False
                
                # Maximum alerts per face
                if face_data['alert_count'] >= self.max_alerts_per_face:
                    # Reset after extended period (e.g., 1 hour)
                    if time_since_last_alert > 3600:  # 1 hour
                        self.logger.info(f"🔄 RESET: Face {face_hash[:8]} for user {user_id} - Resetting after 1 hour")
                        face_data['alert_count'] = 1
                        face_data['last_alert_time'] = current_time
                        return True
                    else:
                        self.logger.info(f"🚫 MAX REACHED: Face {face_hash[:8]} for user {user_id} - {face_data['alert_count']}/{self.max_alerts_per_face} calls made")
                        return False
                
                # ✅ UPDATE: Increment count and time
                face_data['alert_count'] += 1
                face_data['last_alert_time'] = current_time
                # Update encoding to latest version
                face_data['encoding'] = face_encoding
                self.logger.info(f"✅ ALERT #{face_data['alert_count']}: Face {face_hash[:8]} for user {user_id}")
                return True
            
            else:
                # ✅ NEW FACE: Add to tracking
                user_faces[face_hash] = {
                    'encoding': face_encoding,
                    'last_alert_time': current_time,
                    'alert_count': 1,
                    'first_seen': current_time
                }
                self.logger.info(f"✅ NEW FACE: {face_hash[:8]} for user {user_id} - First alert")
                return True
    
    def mark_call_answered(self, user_id, face_encoding):
        """Mark that a call for this face was answered - stop future alerts"""
        if face_encoding is None:
            return
            
        face_hash = self._get_face_hash(face_encoding)
        with self.lock:
            if user_id not in self.user_answered_calls:
                self.user_answered_calls[user_id] = set()
            self.user_answered_calls[user_id].add(face_hash)
            self.logger.info(f"📞 CALL ANSWERED: Face {face_hash[:8]} for user {user_id} - No more alerts")
            
            # Also remove from tracking to free memory
            if user_id in self.user_alerted_faces and face_hash in self.user_alerted_faces[user_id]:
                del self.user_alerted_faces[user_id][face_hash]
    
    def mark_call_not_answered(self, user_id, face_encoding):
        """Mark that a call for this face was NOT answered - continue escalation"""
        if face_encoding is None:
            return
            
        face_hash = self._get_face_hash(face_encoding)
        self.logger.info(f"📞 CALL NOT ANSWERED: Face {face_hash[:8]} for user {user_id} - Will retry if under limit")
    
    def make_voice_call(self, user_id, to_number, message="There is an unauthorized person detected."):
        """Make Twilio voice call for specific user"""
        user_settings = self._get_user_settings(user_id)
        
        if user_settings.get('test_mode', True):
            self.logger.info(f"TEST MODE for user {user_id}: Would call {to_number}")
            return {'status': 'test_mode', 'call_sid': 'test_' + str(uuid.uuid4())}
        
        formatted_number = self._validate_and_format_phone_number(to_number)
        if not formatted_number:
            return {'status': 'error', 'error': f'Invalid phone number: {to_number}'}
            
        client = user_settings.get('client')
        twilio_number = user_settings.get('twilio_number')
        
        if not client or not twilio_number:
            return {'status': 'error', 'error': 'Twilio not configured for this user'}
            
        try:
            call = client.calls.create(
                twiml=f'<Response><Say>{message}</Say></Response>',
                to=formatted_number,
                from_=twilio_number
            )
            
            self.logger.info(f"📞 CALL INITIATED for user {user_id}: {formatted_number} - {call.sid}")
            return {'status': 'initiated', 'call_sid': call.sid}
            
        except TwilioRestException as e:
            self.logger.error(f"Twilio error for user {user_id}: {e}")
            return {'status': 'error', 'error': str(e)}
                
    def _validate_and_format_phone_number(self, phone_number):
        """Validate and format phone number to E.164"""
        if not phone_number:
            return None
            
        cleaned = re.sub(r'[\s\-\(\)]', '', phone_number)
        
        if cleaned.startswith('+'):
            return cleaned
        
        # Philippine format
        if cleaned.startswith('0') and len(cleaned) == 11:
            return '+63' + cleaned[1:]
        elif cleaned.startswith('63') and len(cleaned) == 12:
            return '+' + cleaned
        elif cleaned.startswith('9') and len(cleaned) == 10:
            return '+63' + cleaned
        
        return cleaned
            
    def trigger_alert_escalation(self, user_id, event_id, trigger_type, recording_filepath=None, face_encoding=None):
            """Trigger voice call escalation with face tracking for specific user"""
            
            # Check if we have any active phone numbers for this user
            active_numbers = self.get_active_phone_numbers(user_id)
            
            if not active_numbers:
                # Try reloading just in case
                self.load_phone_numbers(user_id)
                active_numbers = self.get_active_phone_numbers(user_id)
                
                if not active_numbers:
                    print(f"❌ ERROR: No active phone numbers found for user {user_id}. Call aborted.")
                    return {'status': 'error', 'error': 'No phone numbers configured'}
                
            self.log_event(user_id, event_id, trigger_type, recording_filepath)
            
            # Start in background
            thread = Thread(target=self._execute_escalation, args=(user_id, event_id, face_encoding))
            thread.daemon = True
            thread.start()
            
            return {'status': 'escalation_started', 'event_id': event_id}
        
    def _execute_escalation(self, user_id, event_id, face_encoding=None):
        """Execute escalation logic with multiple numbers for specific user"""
        try:
            # ✅ FIX: Get active phone numbers for this user
            active_numbers = self.get_active_phone_numbers(user_id)
            if not active_numbers:
                self.logger.error(f"No active phone numbers available for user {user_id}")
                return
                
            escalation_result = {
                'event_id': event_id,
                'start_time': datetime.now(),
                'attempts': [],
                'final_status': 'failed',
                'answered': False
            }
            
            max_attempts_per_number = 3
            attempt_interval = 30  # 1 minute
            
            # For each number in sequence
            for number_index, phone_number in enumerate(active_numbers):
                if escalation_result['answered']:
                    break
                    
                # Attempt 1-3 for current number
                for attempt in range(1, max_attempts_per_number + 1):
                    # ✅ FIX: Call with correct user_id
                    result = self._make_call_attempt(user_id, event_id, attempt, phone_number, number_index + 1)
                    escalation_result['attempts'].append(result)
                    
                    # ✅ CHECK IF CALL WAS ANSWERED
                    if result['status'] == 'answered':
                        escalation_result['final_status'] = 'answered'
                        escalation_result['answered'] = True
                        
                        # Mark this face as answered - stop future alerts
                        if face_encoding is not None:
                            self.mark_call_answered(user_id, face_encoding)
                            
                        self.update_event_status(user_id, event_id, 'answered', escalation_result)
                        return
                        
                    # If not answered and we have more attempts, wait
                    if attempt < max_attempts_per_number:
                        self.logger.info(f"🔄 Waiting {attempt_interval}s before next attempt for user {user_id}...")
                        time.sleep(attempt_interval)
                
                # If we have more numbers to try, log the transition
                if number_index < len(active_numbers) - 1 and not escalation_result['answered']:
                    next_number = active_numbers[number_index + 1]
                    self.logger.info(f"🔄 Escalating to next number for user {user_id}: {next_number}")
            
            # Mark as not answered if all attempts failed
            if not escalation_result['answered'] and face_encoding is not None:
                self.mark_call_not_answered(user_id, face_encoding)
                
            self.update_event_status(user_id, event_id, escalation_result['final_status'], escalation_result)
            
        except Exception as e:
            self.logger.error(f"Error in escalation for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
        
    def _make_call_attempt(self, user_id, event_id, attempt_number, phone_number, number_sequence=None):
        """Make single call attempt with enhanced logging for specific user"""
        attempt_result = {
            'attempt_number': attempt_number,
            'phone_number': phone_number,
            'number_sequence': number_sequence,
            'timestamp': datetime.now(),
            'status': 'failed'
        }
        
        for retry in range(2):
            call_result = self.make_voice_call(user_id, phone_number)
            
            if call_result['status'] == 'initiated':
                call_status = self._wait_for_call_completion(user_id, call_result['call_sid'])
                attempt_result['status'] = call_status
                attempt_result['call_sid'] = call_result['call_sid']
                break
            elif call_result['status'] == 'test_mode':
                attempt_result['status'] = 'answered'
                break
            else:
                time.sleep(5)
                
        self.log_call_attempt(user_id, event_id, attempt_result)
        return attempt_result
        
    def _wait_for_call_completion(self, user_id, call_sid, timeout=30):
        """Wait for call completion for specific user"""
        user_settings = self._get_user_settings(user_id)
        client = user_settings.get('client')
        
        if not client:
            return 'failed'
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                call = client.calls(call_sid).fetch()
                if call.status in ['completed', 'busy', 'no-answer', 'failed', 'canceled']:
                    return 'answered' if call.status == 'completed' else 'failed'
            except:
                pass
            time.sleep(2)
        return 'timeout'
        
    def cleanup_old_faces(self, user_id=None, hours=24):
        """Clean up old face tracking entries for user(s)"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        with self.lock:
            if user_id:
                # Cleanup for specific user
                if user_id in self.user_alerted_faces:
                    user_faces = self.user_alerted_faces[user_id]
                    faces_to_remove = []
                    for face_hash, face_data in user_faces.items():
                        if face_data['first_seen'] < cutoff_time:
                            faces_to_remove.append(face_hash)
                    
                    for face_hash in faces_to_remove:
                        del user_faces[face_hash]
                    
                    if faces_to_remove:
                        self.logger.info(f"🧹 CLEANUP for user {user_id}: Removed {len(faces_to_remove)} old faces")
                    
                    return len(faces_to_remove)
                return 0
            else:
                # Cleanup for all users
                total_removed = 0
                for uid in list(self.user_alerted_faces.keys()):
                    user_faces = self.user_alerted_faces[uid]
                    faces_to_remove = []
                    for face_hash, face_data in user_faces.items():
                        if face_data['first_seen'] < cutoff_time:
                            faces_to_remove.append(face_hash)
                    
                    for face_hash in faces_to_remove:
                        del user_faces[face_hash]
                    
                    total_removed += len(faces_to_remove)
                    if faces_to_remove:
                        self.logger.info(f"🧹 CLEANUP for user {uid}: Removed {len(faces_to_remove)} old faces")
                
                return total_removed
    
    def get_face_tracking_stats(self, user_id):
        """Get face tracking statistics for a user"""
        with self.lock:
            # ✅ FIX: Ensure user tracking exists
            if user_id not in self.user_alerted_faces:
                self.user_alerted_faces[user_id] = {}
            if user_id not in self.user_answered_calls:
                self.user_answered_calls[user_id] = set()
                
            user_faces = self.user_alerted_faces.get(user_id, {})
            user_answered = self.user_answered_calls.get(user_id, set())
                
            return {
                'total_tracked_faces': len(user_faces),
                'total_answered_faces': len(user_answered),
                'alert_cooldown_seconds': self.alert_cooldown,
                'max_alerts_per_face': self.max_alerts_per_face,
                'face_grouping_threshold': self.face_grouping_threshold,
                'similarity_threshold': self.similarity_threshold,
                'tracked_faces': [
                    {
                        'face_id': face_hash[:8],
                        'alert_count': data['alert_count'],
                        'last_alert_seconds_ago': int(time.time() - data['last_alert_time']),
                        'time_until_reset': max(0, int(self.alert_cooldown - (time.time() - data['last_alert_time']))),
                        'answered': face_hash in user_answered
                    }
                    for face_hash, data in list(user_faces.items())[:10]
                ]
            }
    
    def log_event(self, user_id, event_id, trigger_type, recording_filepath):
        """Log alert event with correct timestamp for specific user"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    timestamp DATETIME,
                    trigger_type TEXT,
                    recording_filepath TEXT,
                    review_status TEXT DEFAULT 'pending',
                    call_status TEXT DEFAULT 'pending',
                    completed_at DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            
            # ✅ FIX: Use datetime correctly
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute(
                'INSERT INTO alert_events (event_id, user_id, timestamp, trigger_type, recording_filepath) VALUES (?, ?, ?, ?, ?)',
                (event_id, user_id, current_time, trigger_type, recording_filepath)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging event for user {user_id}: {e}")
            
    def log_call_attempt(self, user_id, event_id, attempt_result):
        """Log call attempt for specific user"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
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
            
            cursor.execute(
                '''INSERT INTO call_attempts 
                (user_id, event_id, attempt_number, timestamp, phone_number, call_sid, status) 
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (
                    user_id,
                    event_id,
                    attempt_result['attempt_number'],
                    attempt_result['timestamp'].isoformat(),
                    attempt_result['phone_number'],
                    attempt_result.get('call_sid'),
                    attempt_result['status']
                )
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging call attempt for user {user_id}: {e}")
            
    def update_event_status(self, user_id, event_id, call_status, escalation_result=None):
        """Update event status for specific user"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE alert_events SET call_status = ?, completed_at = CURRENT_TIMESTAMP WHERE event_id = ? AND user_id = ?',
                (call_status, event_id, user_id)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating event for user {user_id}: {e}")
    
    def save_settings(self, user_id, settings):
        """Save Twilio settings for a specific user"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            # Get current settings to preserve auth token if not changed
            current_settings = cursor.execute('SELECT * FROM twilio_settings WHERE user_id = ?', (user_id,)).fetchone()
            
            account_sid = settings.get('account_sid', '').strip()
            auth_token = settings.get('auth_token', '').strip()
            twilio_number = settings.get('twilio_number', '').strip()
            test_mode = 1 if settings.get('test_mode', True) else 0
            
            # If auth token is "***" or empty, use the existing one
            if auth_token in ['***', ''] and current_settings:
                auth_token = current_settings[3]  # Use existing auth token (index 3)
            
            if current_settings:
                # UPDATE existing settings
                cursor.execute('''
                    UPDATE twilio_settings 
                    SET account_sid = ?, auth_token = ?, twilio_number = ?, test_mode = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (account_sid, auth_token, twilio_number, test_mode, user_id))
            else:
                # INSERT new settings - let SQLite auto-generate the ID
                cursor.execute('''
                    INSERT INTO twilio_settings 
                    (user_id, account_sid, auth_token, twilio_number, test_mode)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, account_sid, auth_token, twilio_number, test_mode))
            
            conn.commit()
            conn.close()
            
            # Reload settings to update the client
            self.load_settings(user_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings for user {user_id}: {e}")
            return False
    
    def get_settings(self, user_id):
        """Get current settings for a user - mask auth token for display"""
        user_settings = self._get_user_settings(user_id)
        return {
            'account_sid': user_settings.get('account_sid', ''),
            'auth_token': '***' if user_settings.get('auth_token') else '',  # Always mask for display
            'twilio_number': user_settings.get('twilio_number', ''),
            'test_mode': user_settings.get('test_mode', True),
            'configured': bool(user_settings.get('account_sid') and user_settings.get('auth_token') and user_settings.get('twilio_number')),
            'authenticated': user_settings.get('client') is not None
        }
    
    def get_event_history(self, user_id, limit=50):
        """Get event history for a specific user"""
        try:
            conn = sqlite3.connect('security_system.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            events = cursor.execute('''
                SELECT * FROM alert_events 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit)).fetchall()
            
            events_with_attempts = []
            for event in events:
                event_dict = dict(event)
                attempts = cursor.execute('''
                    SELECT * FROM call_attempts 
                    WHERE event_id = ? AND user_id = ?
                    ORDER BY attempt_number
                ''', (event_dict['event_id'], user_id)).fetchall()
                
                event_dict['call_attempts'] = [dict(a) for a in attempts]
                events_with_attempts.append(event_dict)
                
            conn.close()
            return events_with_attempts
            
        except Exception as e:
            self.logger.error(f"Error getting history for user {user_id}: {e}")
            return []
    
    def update_review_status(self, user_id, event_id, status):
        """Update review status for specific user"""
        try:
            valid_statuses = ['pending', 'false_alarm', 'confirmed']
            if status not in valid_statuses:
                return False
                
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE alert_events SET review_status = ? WHERE event_id = ? AND user_id = ?',
                (status, event_id, user_id)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating review for user {user_id}: {e}")
            return False

# Global instance
twilio_alert_system = TwilioAlertSystem()