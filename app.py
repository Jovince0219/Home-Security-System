from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
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

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
FACES_FOLDER = 'static/faces'
SCREENSHOTS_FOLDER = 'static/screenshots'
RECORDINGS_FOLDER = 'static/recordings'
THUMBNAILS_FOLDER = 'static/thumbnails'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, FACES_FOLDER, SCREENSHOTS_FOLDER, RECORDINGS_FOLDER, THUMBNAILS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables
camera = None
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
motion_detector = None
recording_manager = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/face_registration')
def face_registration():
    return render_template('face_registration.html')

@app.route('/motion_detection')
def motion_detection():
    return render_template('motion_detection.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/playback')
def playback():
    return render_template('playback.html')

@app.route('/controls')
def controls():
    return render_template('controls.html')

@app.route('/api/recognize_faces_with_motion', methods=['POST'])
def recognize_faces_with_motion():
    """
    Enhanced motion-gated face recognition with independent unauthorized face detection
    """
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
        
        # Always run face detection, but authorized faces are filtered in the function
        from utils.face_recognition_utils import detect_faces_with_motion_gate
        face_results = detect_faces_with_motion_gate(frame, motion_detector)
        
        # STEP 3: Check for motion without faces using user distance settings
        # Only check for motion-only if no faces were detected AND human motion is active
        if len(face_results) == 0 and (human_detected or motion_detector.is_human_motion_active()):
            print(f"🔴 DEBUG: No faces but human motion detected - checking for motion without faces")
            
            from utils.face_recognition_utils import detect_motion_without_faces
            
            motion_only_result = detect_motion_without_faces(frame, motion_detector, distance_estimator)
            
            print(f"🔴 DEBUG: Motion only result: {motion_only_result}")
            
            # If motion without face detected within user's distance, log it
            if motion_only_result and motion_only_result['trigger_alert']:
                try:
                    # Save screenshot
                    screenshot_path = save_screenshot(
                        frame, 
                        f"motion_no_face_{motion_only_result['distance_meters']}m"
                    )
                    
                    # Add detection to database with proper person_name
                    from utils.database import add_detection
                    add_detection(
                        "motion_detection", 
                        motion_only_result['person_name'],  # Use the person_name from the result
                        float(motion_only_result['confidence']), 
                        screenshot_path, 
                        motion_only_result['alert_level']
                    )
                    
                    print(f"✅ LOGGED MOTION-ONLY ALERT: {motion_only_result['person_name']} at {motion_only_result['distance_meters']}m")
                    
                    # Send alert
                    send_motion_alert_async(motion_only_result, screenshot_path)
                    
                except Exception as e:
                    print(f"❌ Error logging motion-only detection: {e}")
            else:
                print(f"🔴 DEBUG: Motion detected but not logged - trigger_alert: {motion_only_result.get('trigger_alert') if motion_only_result else 'No result'}")
        
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
                'person_name': 'Motion Detected, Unauthorized Person'  # Add this for frontend
            }
        
        return jsonify({
            'success': True, 
            'faces': face_results,
            'motion_only_detection': clean_motion_only_result,
            'motion_detections': detections,  # Only contains animals now
            'motion_status': {
                'human_motion_detected': human_detected,
                'motion_active': motion_detector.is_human_motion_active(),
                'face_detection_enabled': len(face_results) > 0 or motion_detector.is_human_motion_active(),
                'motion_only_alert': motion_only_result is not None,
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
    """Get all security-related detections"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # Get ALL recent detections - no filtering
        all_detections = conn.execute('''
            SELECT * FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 20
        ''').fetchall()
        
        detections_list = []
        for detection in all_detections:
            detection_dict = dict(detection)
            
            # For debugging, print every detection
            print(f"📋 DATABASE DETECTION: type={detection_dict['detection_type']}, name='{detection_dict['person_name']}'")
            
            # Include ALL detections in the results
            detections_list.append({
                'id': detection_dict['id'],
                'type': detection_dict['detection_type'],
                'person_name': detection_dict['person_name'],
                'confidence': detection_dict['confidence'],
                'timestamp': detection_dict['timestamp'],
                'screenshot_path': detection_dict['screenshot_path'],
                'alert_level': detection_dict['alert_level']
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
    from utils.face_recognition_utils import load_known_faces
    from utils.motion_detection_utils import MotionDetector
    from utils.recording_utils import RecordingManager
    
    # Initialize database
    init_db()
    
    # Load known faces
    load_known_faces()
    
    # Initialize motion detector
    motion_detector = MotionDetector()
    
    # Initialize recording manager
    recording_manager = RecordingManager()
    
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
    
