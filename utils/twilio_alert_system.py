import os
import time
import uuid
import sqlite3
from datetime import datetime, timedelta
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
        
        # ✅ FACE TRACKING: Track faces that triggered alerts
        self.alerted_faces = {}  # {face_hash: {'last_alert_time': timestamp, 'alert_count': count, 'encoding': face_encoding}}
        self.alert_cooldown = 300  # 5 minutes (300 seconds)
        self.max_alerts_per_face = 3  # Maximum 3 calls per face
        
        # ✅ ANSWERED CALLS TRACKING
        self.answered_calls = set()  # Track which faces have been answered
        
        # ✅ FACE GROUPING: For better face recognition
        self.face_grouping_threshold = 0.7  # 70% similarity to group faces
        self.similarity_threshold = 0.85  # 85% similarity to consider same face
        
        # ✅ THREAD SAFETY: Lock for concurrent access
        self.lock = Lock()
        
        self.load_settings()
        
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
        
    def load_settings(self):
        """Load Twilio settings from database"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
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
            
            settings = cursor.execute('SELECT * FROM twilio_settings WHERE id = 1').fetchone()
            
            if settings:
                self.account_sid = settings[1]
                self.auth_token = settings[2]
                self.twilio_number = settings[3]
                self.primary_number = settings[4]
                self.backup_number = settings[5]
                self.test_mode = bool(settings[6])
                
                if self.account_sid and self.auth_token:
                    self.client = Client(self.account_sid, self.auth_token)
                    self.logger.info("✅ Twilio client initialized")
            else:
                self.test_mode = True
                self.logger.info("No settings found - using test mode")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.test_mode = True
    
    def _get_face_hash(self, face_encoding):
        """Create unique hash from face encoding"""
        if face_encoding is None:
            return None
        
        # Use first 30 values for hash with good precision
        encoding_str = ','.join([f"{x:.6f}" for x in face_encoding[:30]])
        return hashlib.md5(encoding_str.encode()).hexdigest()
    
    def _find_similar_face(self, new_encoding):
        """Find if a similar face already exists in tracking"""
        if not self.alerted_faces:
            return None
            
        for face_hash, face_data in self.alerted_faces.items():
            stored_encoding = face_data.get('encoding')
            if stored_encoding is None:
                continue
                
            # Calculate cosine similarity between encodings
            similarity = self._calculate_similarity(stored_encoding, new_encoding)
            
            if similarity >= self.similarity_threshold:
                print(f"✅ FACE MATCH: {face_hash[:8]} - Similarity: {similarity:.3f}")
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
    
    def should_trigger_alert_for_face(self, face_encoding):
        """
        ✅ MAIN LOGIC: Check if we should trigger alert for this face
        Returns: True if alert should be triggered, False if duplicate/cooldown
        """
        if self.test_mode:
            self.logger.info("TEST MODE: Allowing alert")
            return True
        
        if face_encoding is None:
            self.logger.warning("No face encoding provided - denying alert")
            return False
        
        current_time = time.time()
        
        with self.lock:  # Thread-safe access
            # ✅ STEP 1: Check if similar face already exists
            similar_face_hash = self._find_similar_face(face_encoding)
            
            if similar_face_hash:
                # Use the existing face ID
                face_data = self.alerted_faces[similar_face_hash]
                face_hash = similar_face_hash
                print(f"🔄 USING EXISTING FACE: {face_hash[:8]}")
            else:
                # Create new face ID
                face_hash = self._get_face_hash(face_encoding)
                print(f"🆕 NEW FACE DETECTED: {face_hash[:8]}")
            
            # ✅ CHECK 1: Has this face been answered already?
            if face_hash in self.answered_calls:
                self.logger.info(f"✅ ANSWERED: Face {face_hash[:8]} - Call was answered, no more alerts")
                return False
            
            # ✅ CHECK 2: Face tracking logic
            if face_hash in self.alerted_faces:
                face_data = self.alerted_faces[face_hash]
                
                # Cooldown period
                time_since_last_alert = current_time - face_data['last_alert_time']
                if time_since_last_alert < self.alert_cooldown:
                    remaining = self.alert_cooldown - time_since_last_alert
                    self.logger.info(f"🔄 COOLDOWN: Face {face_hash[:8]} - {int(remaining)}s remaining of {self.alert_cooldown}s")
                    return False
                
                # Maximum alerts per face
                if face_data['alert_count'] >= self.max_alerts_per_face:
                    # Reset after extended period (e.g., 1 hour)
                    if time_since_last_alert > 3600:  # 1 hour
                        self.logger.info(f"🔄 RESET: Face {face_hash[:8]} - Resetting after 1 hour")
                        face_data['alert_count'] = 1
                        face_data['last_alert_time'] = current_time
                        return True
                    else:
                        self.logger.info(f"🚫 MAX REACHED: Face {face_hash[:8]} - {face_data['alert_count']}/{self.max_alerts_per_face} calls made")
                        return False
                
                # ✅ UPDATE: Increment count and time
                face_data['alert_count'] += 1
                face_data['last_alert_time'] = current_time
                # Update encoding to latest version
                face_data['encoding'] = face_encoding
                self.logger.info(f"✅ ALERT #{face_data['alert_count']}: Face {face_hash[:8]}")
                return True
            
            else:
                # ✅ NEW FACE: Add to tracking
                self.alerted_faces[face_hash] = {
                    'encoding': face_encoding,
                    'last_alert_time': current_time,
                    'alert_count': 1,
                    'first_seen': current_time
                }
                self.logger.info(f"✅ NEW FACE: {face_hash[:8]} - First alert")
                return True
    
    def mark_call_answered(self, face_encoding):
        """Mark that a call for this face was answered - stop future alerts"""
        if face_encoding is None:
            return
            
        face_hash = self._get_face_hash(face_encoding)
        with self.lock:
            self.answered_calls.add(face_hash)
            self.logger.info(f"📞 CALL ANSWERED: Face {face_hash[:8]} - No more alerts")
            
            # Also remove from tracking to free memory
            if face_hash in self.alerted_faces:
                del self.alerted_faces[face_hash]
    
    def mark_call_not_answered(self, face_encoding):
        """Mark that a call for this face was NOT answered - continue escalation"""
        if face_encoding is None:
            return
            
        face_hash = self._get_face_hash(face_encoding)
        self.logger.info(f"📞 CALL NOT ANSWERED: Face {face_hash[:8]} - Will retry if under limit")
    
    def make_voice_call(self, to_number, message="There is an unauthorized person detected."):
        """Make Twilio voice call"""
        if self.test_mode:
            self.logger.info(f"TEST MODE: Would call {to_number}")
            return {'status': 'test_mode', 'call_sid': 'test_' + str(uuid.uuid4())}
        
        formatted_number = self._validate_and_format_phone_number(to_number)
        if not formatted_number:
            return {'status': 'error', 'error': f'Invalid phone number: {to_number}'}
            
        if not self.client or not self.twilio_number:
            return {'status': 'error', 'error': 'Twilio not configured'}
            
        try:
            call = self.client.calls.create(
                twiml=f'<Response><Say>{message}</Say></Response>',
                to=formatted_number,
                from_=self.twilio_number
            )
            
            self.logger.info(f"📞 CALL INITIATED: {formatted_number} - {call.sid}")
            return {'status': 'initiated', 'call_sid': call.sid}
            
        except TwilioRestException as e:
            self.logger.error(f"Twilio error: {e}")
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
            
    def trigger_alert_escalation(self, event_id, trigger_type, recording_filepath=None, face_encoding=None):
        """Trigger voice call escalation with face tracking"""
        if not self.primary_number:
            return {'status': 'error', 'error': 'No phone numbers configured'}
            
        self.log_event(event_id, trigger_type, recording_filepath)
        
        # Start in background
        thread = Thread(target=self._execute_escalation, args=(event_id, face_encoding))
        thread.daemon = True
        thread.start()
        
        return {'status': 'escalation_started', 'event_id': event_id}
        
    def _execute_escalation(self, event_id, face_encoding=None):
        """Execute escalation logic with answer tracking"""
        escalation_result = {
            'event_id': event_id,
            'start_time': datetime.now(),
            'attempts': [],
            'final_status': 'failed',
            'answered': False
        }
        
        max_attempts = 3
        attempt_interval = 60  # 1 minute
        
        # Attempt 1-3: Primary number
        for attempt in range(1, max_attempts + 1):
            result = self._make_call_attempt(event_id, attempt, self.primary_number)
            escalation_result['attempts'].append(result)
            
            # ✅ CHECK IF CALL WAS ANSWERED
            if result['status'] == 'answered':
                escalation_result['final_status'] = 'answered'
                escalation_result['answered'] = True
                
                # Mark this face as answered - stop future alerts
                if face_encoding is not None:
                    self.mark_call_answered(face_encoding)
                    
                self.update_event_status(event_id, 'answered', escalation_result)
                return
                
            # If not answered and we have more attempts, wait
            if attempt < max_attempts:
                self.logger.info(f"🔄 Waiting {attempt_interval}s before next attempt...")
                time.sleep(attempt_interval)
                
        # If primary number failed all attempts, try backup
        if not escalation_result['answered'] and self.backup_number:
            result = self._make_call_attempt(event_id, max_attempts + 1, self.backup_number)
            escalation_result['attempts'].append(result)
            
            if result['status'] == 'answered':
                escalation_result['final_status'] = 'answered'
                escalation_result['answered'] = True
                if face_encoding is not None:
                    self.mark_call_answered(face_encoding)
        
        # Mark as not answered if all attempts failed
        if not escalation_result['answered'] and face_encoding is not None:
            self.mark_call_not_answered(face_encoding)
            
        self.update_event_status(event_id, escalation_result['final_status'], escalation_result)
        
    def _make_call_attempt(self, event_id, attempt_number, phone_number):
        """Make single call attempt"""
        attempt_result = {
            'attempt_number': attempt_number,
            'phone_number': phone_number,
            'timestamp': datetime.now(),
            'status': 'failed'
        }
        
        for retry in range(2):
            call_result = self.make_voice_call(phone_number)
            
            if call_result['status'] == 'initiated':
                call_status = self._wait_for_call_completion(call_result['call_sid'])
                attempt_result['status'] = call_status
                attempt_result['call_sid'] = call_result['call_sid']
                break
            elif call_result['status'] == 'test_mode':
                attempt_result['status'] = 'answered'
                break
            else:
                time.sleep(5)
                
        self.log_call_attempt(event_id, attempt_result)
        return attempt_result
        
    def _wait_for_call_completion(self, call_sid, timeout=30):
        """Wait for call completion"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                call = self.client.calls(call_sid).fetch()
                if call.status in ['completed', 'busy', 'no-answer', 'failed', 'canceled']:
                    return 'answered' if call.status == 'completed' else 'failed'
            except:
                pass
            time.sleep(2)
        return 'timeout'
        
    def cleanup_old_faces(self, hours=24):
        """Clean up old face tracking entries"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        with self.lock:
            faces_to_remove = []
            for face_hash, face_data in self.alerted_faces.items():
                if face_data['first_seen'] < cutoff_time:
                    faces_to_remove.append(face_hash)
            
            for face_hash in faces_to_remove:
                del self.alerted_faces[face_hash]
            
            if faces_to_remove:
                self.logger.info(f"🧹 CLEANUP: Removed {len(faces_to_remove)} old faces")
            
            return len(faces_to_remove)
    
    def get_face_tracking_stats(self):
        """Get face tracking statistics"""
        with self.lock:
            # ✅ FIX: Ensure answered_calls exists
            if not hasattr(self, 'answered_calls'):
                self.answered_calls = set()
                
            return {
                'total_tracked_faces': len(self.alerted_faces),
                'total_answered_faces': len(self.answered_calls),
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
                        'answered': face_hash in self.answered_calls
                    }
                    for face_hash, data in list(self.alerted_faces.items())[:10]
                ]
            }
    
    def log_event(self, event_id, trigger_type, recording_filepath):
        """Log alert event with correct timestamp"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    trigger_type TEXT,
                    recording_filepath TEXT,
                    review_status TEXT DEFAULT 'pending',
                    call_status TEXT DEFAULT 'pending',
                    completed_at DATETIME
                )
            ''')
            
            # Use current datetime from Python
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute(
                'INSERT INTO alert_events (event_id, timestamp, trigger_type, recording_filepath) VALUES (?, ?, ?, ?)',
                (event_id, current_time, trigger_type, recording_filepath)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging event: {e}")
            
    def log_call_attempt(self, event_id, attempt_result):
        """Log call attempt"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
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
            
            cursor.execute(
                '''INSERT INTO call_attempts 
                (event_id, attempt_number, timestamp, phone_number, call_sid, status) 
                VALUES (?, ?, ?, ?, ?, ?)''',
                (
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
            self.logger.error(f"Error logging call attempt: {e}")
            
    def update_event_status(self, event_id, call_status, escalation_result=None):
        """Update event status"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE alert_events SET call_status = ?, completed_at = CURRENT_TIMESTAMP WHERE event_id = ?',
                (call_status, event_id)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating event: {e}")
    
    def save_settings(self, settings):
        """Save Twilio settings"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO twilio_settings 
                (id, account_sid, auth_token, twilio_number, primary_number, backup_number, test_mode)
                VALUES (1, ?, ?, ?, ?, ?, ?)
            ''', (
                settings.get('account_sid', '').strip(),
                settings.get('auth_token', '').strip(),
                settings.get('twilio_number', '').strip(),
                settings.get('primary_number', '').strip(),
                settings.get('backup_number', '').strip(),
                1 if settings.get('test_mode', True) else 0
            ))
            
            conn.commit()
            conn.close()
            
            self.load_settings()
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
    
    def get_settings(self):
        """Get current settings"""
        return {
            'account_sid': self.account_sid,
            'auth_token': '***' if self.auth_token else '',
            'twilio_number': self.twilio_number,
            'primary_number': self.primary_number,
            'backup_number': self.backup_number,
            'test_mode': self.test_mode,
            'configured': bool(self.account_sid and self.auth_token and self.twilio_number)
        }
    
    def get_event_history(self, limit=50):
        """Get event history"""
        try:
            conn = sqlite3.connect('security_system.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            events = cursor.execute('''
                SELECT * FROM alert_events 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,)).fetchall()
            
            events_with_attempts = []
            for event in events:
                event_dict = dict(event)
                attempts = cursor.execute('''
                    SELECT * FROM call_attempts 
                    WHERE event_id = ? 
                    ORDER BY attempt_number
                ''', (event_dict['event_id'],)).fetchall()
                
                event_dict['call_attempts'] = [dict(a) for a in attempts]
                events_with_attempts.append(event_dict)
                
            conn.close()
            return events_with_attempts
            
        except Exception as e:
            self.logger.error(f"Error getting history: {e}")
            return []
    
    def update_review_status(self, event_id, status):
        """Update review status"""
        try:
            valid_statuses = ['pending', 'false_alarm', 'confirmed']
            if status not in valid_statuses:
                return False
                
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE alert_events SET review_status = ? WHERE event_id = ?',
                (status, event_id)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating review: {e}")
            return False

# Global instance
twilio_alert_system = TwilioAlertSystem()