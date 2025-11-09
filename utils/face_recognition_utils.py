import face_recognition
import cv2
import numpy as np
import os
from utils.database import get_all_faces, add_detection, get_face_by_id, delete_face_image_by_index, add_face_image, get_face_images
import datetime
from utils.alert_system import AlertSystem
from utils.distance_estimation import get_distance_estimator

# Global variables for face recognition
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load all registered faces from database"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    faces = get_all_faces()
    for face in faces:
        try:
            encoding = np.array([float(x) for x in face['encoding'].split(',')])
            known_face_encodings.append(encoding)
            known_face_names.append(face['name'])
        except Exception as e:
            print(f"Error loading face encoding for {face['name']}: {e}")
            
def detect_faces_with_motion_gate(frame, motion_detector, scale_factor=1.0):
    """
    UPDATED: Face detection that works independently for unauthorized persons
    Authorized faces: Only detected when human motion is active
    Unauthorized faces: Always detected regardless of motion
    """
    global known_face_encodings, known_face_names
    
    # Get distance estimator with user settings
    distance_estimator = get_distance_estimator()
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    
    if len(face_locations) == 0:
        return []
    
    # Encode all faces at once
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # STEP 1: Estimate distance FIRST using user settings
        face_location = [top, right, bottom, left]
        face_width_pixels = right - left
        distance_meters = distance_estimator.estimate_distance(face_width_pixels)
        
        # STEP 2: Check if within user's configured MAX Detection Distance
        if distance_meters > distance_estimator.MAX_DETECTION_DISTANCE:
            continue  # Skip face recognition if beyond user's max distance
        
        # STEP 3: Perform face recognition
        matches = face_recognition.compare_faces(
            known_face_encodings, 
            face_encoding, 
            tolerance=0.55
        )
        
        name = "Unknown"
        confidence = 0.0
        
        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            matched_encodings = [known_face_encodings[i] for i in matched_indices]
            
            face_distances = face_recognition.face_distance(matched_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            actual_index = matched_indices[best_match_index]
            
            name = known_face_names[actual_index]
            confidence = float(1.0 - face_distances[best_match_index])
            confidence = max(0.0, min(1.0, confidence))
            
            print(f"DEBUG: Authorized face '{name}' detected at {distance_meters:.2f}m with confidence: {confidence:.3f}")
            
            # FOR AUTHORIZED FACES: Only process if human motion is active
            if not motion_detector.is_human_motion_active():
                print(f"SKIPPING AUTHORIZED FACE: No human motion active")
                continue  # Skip authorized faces when no human motion
            
        else:
            name = "Unauthorized"
            # For unauthorized faces, estimate confidence based on face size/quality
            face_width_pixels = right - left
            face_height_pixels = bottom - top
            face_area = face_width_pixels * face_height_pixels
            
            # Simple confidence estimation for unauthorized faces
            if face_area > 10000:
                confidence = 0.85
            elif face_area > 5000:
                confidence = 0.70
            else:
                confidence = 0.55
            
            print(f"DEBUG: Unauthorized face detected at {distance_meters:.2f}m with confidence: {confidence:.3f}")
            # UNAUTHORIZED FACES: Always process regardless of motion

        is_authorized = name != "Unauthorized"
        
        # Use distance estimator with user settings
        distance_analysis = distance_estimator.analyze_face_detection(face_location, is_authorized)
        
        # Check face covering
        face_region = rgb_frame[top:bottom, left:right]
        face_covered = detect_face_covering_fast(face_region)
        
        # Better face box fitting
        face_width = right - left
        face_height = bottom - top
        
        width_reduction = int(face_width * 0.20)
        height_top_reduction = int(face_height * 0.15)
        height_bottom_reduction = int(face_height * 0.05)
        
        adjusted_left = max(0, left + width_reduction)
        adjusted_right = min(rgb_frame.shape[1], right - width_reduction)
        adjusted_top = max(0, top + height_top_reduction)
        adjusted_bottom = min(rgb_frame.shape[0], bottom - height_bottom_reduction)
        
        # Determine display name based on distance and authorization
        if is_authorized:
            display_name = str(name)
        else:
            if distance_analysis['within_detection_range']:
                display_name = 'Unauthorized'
            else:
                display_name = 'Too Far'
        
        # Create result
        result = {
            'name': str(name),
            'confidence': float(confidence),
            'location': [int(adjusted_top), int(adjusted_right), int(adjusted_bottom), int(adjusted_left)],
            'face_covered': bool(face_covered),
            'is_authorized': bool(is_authorized),
            'display_name': display_name,
            'distance_meters': float(distance_analysis['distance_meters']),
            'distance_feet': float(distance_analysis['distance_feet']),
            'zone': str(distance_analysis['zone']),
            'trigger_alert': bool(distance_analysis['trigger_alert']),
            'alert_level': int(distance_analysis['alert_level']),
            'within_detection_range': bool(distance_analysis['within_detection_range'])
        }
        
        results.append(result)
        
        # STEP 4: Handle unauthorized faces (check for duplicates)
        if not is_authorized and distance_analysis['trigger_alert']:
            # Create a simple hash for the face encoding (using first few values)
            encoding_key = hash(tuple(face_encoding[:10]))
            
            # Check if this face was recently logged
            if encoding_key in motion_detector.recent_unauthorized_faces:
                last_detection = motion_detector.recent_unauthorized_faces[encoding_key]
                elapsed = (datetime.datetime.now() - last_detection).total_seconds()
                
                if elapsed < motion_detector.duplicate_detection_window:
                    print(f"Skipping duplicate unauthorized face (last seen {elapsed:.1f}s ago)")
                    continue  # Skip logging this duplicate
            
            # DETECTION - Log it
            try:
                screenshot_path = save_screenshot(
                    frame, 
                    f"unauthorized_{distance_analysis['distance_meters']}m_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                alert_level = distance_analysis['alert_level']
                
                # Add detection to database
                add_detection(
                    "face_detection", 
                    f"Unauthorized ({distance_analysis['distance_meters']}m)", 
                    float(confidence), 
                    screenshot_path, 
                    alert_level
                )
                
                # Mark this face as recently detected
                motion_detector.recent_unauthorized_faces[encoding_key] = datetime.datetime.now()
                
                # Send alert in background
                import threading
                alert_thread = threading.Thread(
                    target=send_alert_async,
                    args=(alert_level, "face_detection", str(name), screenshot_path, distance_analysis)
                )
                alert_thread.daemon = True
                alert_thread.start()
                
            except Exception as e:
                print(f"Error logging detection: {e}")
    
    # Cleanup old detections periodically
    motion_detector.cleanup_old_detections()
    
    return results


def estimate_distance_from_area(self, motion_area):
    """
    Estimate distance based on motion detection area using user's max distance as reference
    """
    # Use the user's max detection distance as the baseline
    max_distance = self.MAX_DETECTION_DISTANCE
    
    # Calibration: adjust these ratios based on your camera
    # These are approximate ratios - you may need to calibrate
    if motion_area > 50000:
        return max_distance * 0.08  # 8% of max distance = very close
    elif motion_area > 30000:
        return max_distance * 0.17  # 17% of max distance
    elif motion_area > 20000:
        return max_distance * 0.25  # 25% of max distance = critical zone
    elif motion_area > 10000:
        return max_distance * 0.42  # 42% of max distance = warning zone
    elif motion_area > 5000:
        return max_distance * 0.67  # 67% of max distance = detection zone
    else:
        return max_distance * 1.1   # Beyond max distance

def analyze_motion_detection(self, bbox, motion_area, is_human=True):
    """
    Analyze motion detection and determine distance-based alerts using user settings
    """
    try:
        # Estimate distance from motion area
        distance = self.estimate_distance_from_area(motion_area)
        
        # Determine if alert should be triggered based on user's max distance
        trigger_alert = distance <= self.MAX_DETECTION_DISTANCE
        
        # Get alert level based on user's warning/critical distances
        alert_level = self.get_alert_level(distance) if trigger_alert else 0
        
        return {
            'distance_meters': round(distance, 2),
            'distance_feet': round(distance * 3.28084, 2),
            'trigger_alert': trigger_alert,
            'alert_level': alert_level,
            'zone': self.get_distance_zone(distance),
            'within_detection_range': distance <= self.MAX_DETECTION_DISTANCE,
            'motion_area': motion_area
        }
        
    except Exception as e:
        print(f"Error in motion detection analysis: {e}")
        return {
            'distance_meters': self.MAX_DETECTION_DISTANCE + 1,
            'distance_feet': (self.MAX_DETECTION_DISTANCE + 1) * 3.28084,
            'trigger_alert': False,
            'alert_level': 0,
            'zone': 'SAFE ZONE',
            'within_detection_range': False,
            'motion_area': motion_area
        }

def detect_motion_without_faces(frame, motion_detector, distance_estimator):
    """
    Detect motion without faces and determine if it should trigger an alert
    """
    try:
        print(f"🟡 DEBUG: detect_motion_without_faces called - checking for motion without faces")
        
        # Use the motion detector to check for human motion
        motion_detected, detections, fg_mask, human_detected = motion_detector.detect_motion(frame)
        
        print(f"🟡 DEBUG: Motion detected: {motion_detected}, Human detected: {human_detected}")
        
        # Only proceed if human motion is detected but no faces were found
        if not human_detected:
            print(f"🔴 DEBUG: No human motion detected, skipping motion-only detection")
            return None
        
        # Find the largest human motion detection
        human_motions = []
        for detection in detections:
            if detection.get('type') == 'human':
                human_motions.append(detection)
        
        if not human_motions:
            print(f"🔴 DEBUG: No human motion detections found")
            return None
        
        # Use the largest human motion
        largest_motion = max(human_motions, key=lambda x: x.get('area', 0))
        
        # Estimate distance from motion area using user settings
        motion_area = largest_motion.get('area', 0)
        estimated_distance = distance_estimator.estimate_distance_from_area(motion_area)
        
        # Check if within user's detection range
        within_range = estimated_distance <= distance_estimator.MAX_DETECTION_DISTANCE
        
        # Determine alert level based on user's distance settings
        alert_level = distance_estimator.get_alert_level(estimated_distance) if within_range else 0
        trigger_alert = within_range and alert_level > 0
        
        zone = distance_estimator.get_distance_zone(estimated_distance)
        
        result = {
            'type': 'motion_no_face',
            'distance_meters': round(estimated_distance, 2),
            'distance_feet': round(estimated_distance * 3.28084, 2),
            'confidence': largest_motion.get('confidence', 0.7),
            'bbox': largest_motion.get('bbox', [100, 100, 300, 300]),
            'zone': zone,
            'trigger_alert': trigger_alert,
            'alert_level': alert_level,
            'motion_area': motion_area,
            'within_detection_range': within_range,
            'person_name': 'Motion Detected, Unauthorized Person'
        }
        
        print(f"✅ DEBUG: Motion-only detection created: trigger_alert={result['trigger_alert']}, distance={result['distance_meters']}m, zone={result['zone']}")
        return result
        
    except Exception as e:
        print(f"❌ ERROR in detect_motion_without_faces: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_faces_in_frame_optimized(frame, scale_factor=1.0):
    """
    OPTIMIZED VERSION with DISTANCE ESTIMATION
    Detect and recognize faces, estimate distance, and only alert for close unauthorized persons
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        scale_factor: Scale factor if frame was resized (1.0 = original size)
    
    Returns:
        List of detected faces with locations, info, and distance
    """
    global known_face_encodings, known_face_names
    
    # Get distance estimator
    distance_estimator = get_distance_estimator()
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    
    if len(face_locations) == 0:
        return []
    
    # Encode all faces at once
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    authorized_detected = False
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Face recognition
        matches = face_recognition.compare_faces(
            known_face_encodings, 
            face_encoding, 
            tolerance=0.55
        )
        
        name = "Unknown"
        confidence = 0.0
        
        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            matched_encodings = [known_face_encodings[i] for i in matched_indices]
            
            face_distances = face_recognition.face_distance(matched_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            actual_index = matched_indices[best_match_index]
            
            name = known_face_names[actual_index]
            confidence = float(1.0 - face_distances[best_match_index])  # Ensure it's a float
            
            # Ensure confidence is within valid range [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
        else:
            name = "Unauthorized"
            # For unauthorized faces, we can estimate confidence based on face quality
            # Use a default confidence or calculate based on face size/quality
            face_width_pixels = right - left
            face_height_pixels = bottom - top
            face_area = face_width_pixels * face_height_pixels
            
            # Simple confidence estimation for unauthorized faces
            if face_area > 10000:  # Large face = high confidence
                confidence = 0.85
            elif face_area > 5000:  # Medium face = medium confidence
                confidence = 0.70
            else:  # Small face = lower confidence
                confidence = 0.55
            

        is_authorized = name != "Unauthorized"
        
        # DISTANCE ESTIMATION
        face_location = [top, right, bottom, left]
        distance_analysis = distance_estimator.analyze_face_detection(face_location, is_authorized)
        
        # Check face covering
        face_region = rgb_frame[top:bottom, left:right]
        face_covered = detect_face_covering_fast(face_region)
        
        # Better face box fitting
        face_width = right - left
        face_height = bottom - top
        
        width_reduction = int(face_width * 0.20)
        height_top_reduction = int(face_height * 0.15)
        height_bottom_reduction = int(face_height * 0.05)
        
        adjusted_left = max(0, left + width_reduction)
        adjusted_right = min(rgb_frame.shape[1], right - width_reduction)
        adjusted_top = max(0, top + height_top_reduction)
        adjusted_bottom = min(rgb_frame.shape[0], bottom - height_bottom_reduction)
        
        # Determine display name based on distance and authorization
        if is_authorized:
            display_name = str(name)
        else:
            if distance_analysis['within_detection_range']:
                display_name = 'Unauthorized'
            else:
                display_name = 'Too Far'
        
        results.append({
            'name': str(name),
            'confidence': float(confidence),
            'location': [int(adjusted_top), int(adjusted_right), int(adjusted_bottom), int(adjusted_left)],
            'face_covered': bool(face_covered),
            'is_authorized': bool(is_authorized),
            'display_name': display_name,
            'distance_meters': distance_analysis['distance_meters'],
            'distance_feet': distance_analysis['distance_feet'],
            'zone': distance_analysis['zone'],
            'trigger_alert': distance_analysis['trigger_alert'],
            'alert_level': distance_analysis['alert_level'],
            'within_detection_range': distance_analysis['within_detection_range']
        })
        
        # Only log and alert for unauthorized persons within detection range
        if not is_authorized and distance_analysis['trigger_alert']:
            try:
                screenshot_path = save_screenshot(
                    frame, 
                    f"unauthorized_{distance_analysis['distance_meters']}m_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                alert_level = distance_analysis['alert_level']
                
                # Add detection to database
                add_detection(
                    "face_detection", 
                    f"Unauthorized ({distance_analysis['distance_meters']}m)", 
                    float(confidence), 
                    screenshot_path, 
                    alert_level
                )
                
                # Send alert in background
                import threading
                alert_thread = threading.Thread(
                    target=send_alert_async,
                    args=(alert_level, "face_detection", str(name), screenshot_path, distance_analysis)
                )
                alert_thread.daemon = True
                alert_thread.start()
                
            except Exception as e:
                print(f"Error logging detection: {e}")
    
    return results

def send_alert_async(alert_level, detection_type, name, screenshot_path, distance_info):
    """Send alert asynchronously with distance information"""
    try:
        alert_system = AlertSystem()
        
        message = f"Unauthorized person detected at {distance_info['distance_meters']}m (CRITICAL ZONE - Within {distance_info['distance_meters']}m)"
        
        alert_result = alert_system.send_alert(
            alert_level, 
            detection_type, 
            name, 
            screenshot_path=screenshot_path,
            custom_message=message
        )
        print(f"Alert sent: {alert_result}")
    except Exception as e:
        print(f"Error sending alert: {e}")

def detect_face_covering_fast(face_region):
    """
    OPTIMIZED: Faster face covering detection
    """
    if face_region.size == 0:
        return False
        
    try:
        h, w = face_region.shape[:2]
        
        if h < 20 or w < 20:
            return False
        
        lower_face = face_region[int(h*0.6):, :]
        
        if len(lower_face.shape) == 3:
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_RGB2GRAY)
        else:
            gray_lower = lower_face
        
        variance = np.var(gray_lower)
        edges = cv2.Canny(gray_lower, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        is_covered = (variance < 250) and (edge_density > 0.15)
        
        return is_covered
        
    except Exception as e:
        print(f"Error in fast face covering detection: {e}")
        return False

def detect_faces_in_frame(frame):
    """
    Original function kept for backward compatibility
    Calls the optimized version with scale_factor=1.0
    """
    return detect_faces_in_frame_optimized(frame, scale_factor=1.0)

def detect_face_covering(face_region):
    """
    Original function kept for backward compatibility
    Calls the optimized version
    """
    return detect_face_covering_fast(face_region)

def save_screenshot(frame, filename):
    """Save screenshot to screenshots folder"""
    try:
        filepath = os.path.join('static/screenshots', f"{filename}.jpg")
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filepath
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return None

def train_face_recognition_model():
    """Train/retrain the face recognition model with all registered faces"""
    print("Training face recognition model...")
    load_known_faces()
    print(f"Loaded {len(known_face_names)} registered faces")
    return len(known_face_names)

def get_face_recognition_stats():
    """Get statistics about face recognition system"""
    return {
        'total_registered_faces': len(known_face_names),
        'registered_names': known_face_names,
        'model_loaded': len(known_face_encodings) > 0
    }

def register_multiple_faces_from_camera(name):
    """Capture multiple faces from camera for better training"""
    cap = cv2.VideoCapture(0)
    captured_faces = []
    
    print(f"Multiple Face Registration Mode for {name}")
    print("Press SPACE to capture photo, ENTER to finish, ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, f"Registering: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {len(captured_faces)} photos", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | ENTER: Finish | ESC: Cancel", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if len(captured_faces) >= 1:
            cv2.putText(frame, "Ready to finish! Press ENTER", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            face_width = right - left
            face_height = bottom - top
            width_reduction = int(face_width * 0.20)
            height_top_reduction = int(face_height * 0.15)
            height_bottom_reduction = int(face_height * 0.05)
            
            adjusted_left = max(0, left + width_reduction)
            adjusted_right = min(frame.shape[1], right - width_reduction)
            adjusted_top = max(0, top + height_top_reduction)
            adjusted_bottom = min(frame.shape[0], bottom - height_bottom_reduction)
            
            cv2.rectangle(frame, (adjusted_left, adjusted_top), (adjusted_right, adjusted_bottom), (0, 255, 0), 2)
        
        cv2.imshow('Multiple Face Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE
            if len(face_locations) == 1:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                
                captured_faces.append({
                    'encoding': face_encoding,
                    'frame': frame.copy(),
                    'timestamp': timestamp
                })
                print(f"Captured face {len(captured_faces)}")
            else:
                print("Please ensure exactly one face is visible")
        elif key == 13:  # ENTER
            if len(captured_faces) > 0:
                break
            else:
                print("Please capture at least one face")
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_faces

def register_face_multiple_images(name, captured_faces):
    """Register multiple face images for a single person"""
    try:
        if not captured_faces:
            return {'error': 'No faces captured'}
        
        registered_images = []
        
        for i, face_data in enumerate(captured_faces):
            timestamp = face_data['timestamp']
            filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join('static/faces', filename)
            
            cv2.imwrite(image_path, face_data['frame'])
            
            encoding_str = ','.join(map(str, face_data['encoding']))
            face_id = add_face_image(name, encoding_str, image_path)
            
            registered_images.append({
                'face_id': face_id,
                'image_path': image_path,
                'timestamp': timestamp
            })
        
        load_known_faces()
        
        return {
            'success': True,
            'message': f'Successfully registered {len(registered_images)} images for {name}',
            'registered_images': registered_images
        }
        
    except Exception as e:
        return {'error': str(e)}

def delete_face_image(face_id, image_index):
    """Delete a specific image from a face registration"""
    try:
        face = get_face_by_id(face_id)
        if not face:
            return {'error': 'Face not found'}
        
        face_images = get_face_images(face_id)
        
        if image_index >= len(face_images):
            return {'error': 'Image index out of range'}
        
        image_path = face_images[image_index]['image_path']
        if os.path.exists(image_path):
            os.remove(image_path)
        
        delete_face_image_by_index(face_id, image_index)
        
        return {'success': True, 'message': 'Image deleted successfully'}
        
    except Exception as e:
        return {'error': str(e)}

def create_privacy_thumbnail(image_path, blur_faces=True):
    """Create a privacy-protected thumbnail with blurred faces"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        if blur_faces:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            for (top, right, bottom, left) in face_locations:
                face_region = image[top:bottom, left:right]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                image[top:bottom, left:right] = blurred_face
        
        height, width = image.shape[:2]
        max_size = 200
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        thumbnail = cv2.resize(image, (new_width, new_height))
        
        thumbnail_dir = 'static/thumbnails'
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        thumbnail_path = os.path.join(thumbnail_dir, f"{name}_thumb{ext}")
        
        cv2.imwrite(thumbnail_path, thumbnail)
        
        return thumbnail_path
        
    except Exception as e:
        print(f"Error creating privacy thumbnail: {e}")
        return None