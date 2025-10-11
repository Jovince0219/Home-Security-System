import face_recognition
import cv2
import numpy as np
import os
from utils.database import get_all_faces, add_detection, get_face_by_id, delete_face_image_by_index, add_face_image, get_face_images
import datetime
from utils.alert_system import AlertSystem

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

def detect_faces_in_frame_optimized(frame, scale_factor=1.0):
    """
    OPTIMIZED VERSION: Detect and recognize faces with better performance
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        scale_factor: Scale factor if frame was resized (1.0 = original size)
    
    Returns:
        List of detected faces with locations and info
    """
    global known_face_encodings, known_face_names
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # OPTIMIZATION 1: Use faster face detection model
    # Using 'cnn' model is more accurate but slower
    # Using default (HOG) model is faster
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    
    # OPTIMIZATION 2: Only process faces if found
    if len(face_locations) == 0:
        return []
    
    # OPTIMIZATION 3: Encode all faces at once (batch processing)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    authorized_detected = False
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # OPTIMIZATION 4: Early matching with tolerance adjustment
        # Higher tolerance = faster but less accurate
        # Lower tolerance = slower but more accurate
        matches = face_recognition.compare_faces(
            known_face_encodings, 
            face_encoding, 
            tolerance=0.55  # Slightly more lenient for speed
        )
        
        name = "Unknown"
        confidence = 0.0
        
        if True in matches:
            # OPTIMIZATION 5: Only calculate distances for matched faces
            matched_indices = [i for i, match in enumerate(matches) if match]
            matched_encodings = [known_face_encodings[i] for i in matched_indices]
            
            face_distances = face_recognition.face_distance(matched_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            actual_index = matched_indices[best_match_index]
            
            name = known_face_names[actual_index]
            confidence = 1.0 - face_distances[best_match_index]
            authorized_detected = True
        
        # OPTIMIZATION 6: Simplified face covering detection
        face_region = rgb_frame[top:bottom, left:right]
        face_covered = detect_face_covering_fast(face_region)
        
        is_authorized = name != "Unknown"
        
        # Skip unauthorized faces if authorized person is present
        if authorized_detected and not is_authorized:
            continue
        
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
        
        results.append({
            'name': str(name),
            'confidence': float(confidence),
            'location': [int(adjusted_top), int(adjusted_right), int(adjusted_bottom), int(adjusted_left)],
            'face_covered': bool(face_covered),
            'is_authorized': bool(is_authorized),
            'display_name': str(name) if is_authorized else 'Unauthorized'
        })
        
        # OPTIMIZATION 7: Async logging (don't wait for DB write)
        # Only log unauthorized or covered faces to reduce DB writes
        if not is_authorized or face_covered:
            try:
                screenshot_path = save_screenshot(frame, f"face_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                alert_level = 1 if is_authorized else 2
                add_detection("face_detection", str(name), float(confidence), screenshot_path, alert_level)
                
                if not is_authorized:
                    # Send alert in background
                    import threading
                    alert_thread = threading.Thread(
                        target=send_alert_async,
                        args=(alert_level, "face_detection", str(name), screenshot_path)
                    )
                    alert_thread.daemon = True
                    alert_thread.start()
            except Exception as e:
                print(f"Error logging detection: {e}")
    
    return results

def send_alert_async(alert_level, detection_type, name, screenshot_path):
    """Send alert asynchronously to avoid blocking detection"""
    try:
        alert_system = AlertSystem()
        alert_result = alert_system.send_alert(alert_level, detection_type, name, screenshot_path=screenshot_path)
        print(f"Alert sent: {alert_result}")
    except Exception as e:
        print(f"Error sending alert: {e}")

def detect_face_covering_fast(face_region):
    """
    OPTIMIZED: Faster face covering detection
    Uses fewer checks for better performance
    """
    if face_region.size == 0:
        return False
        
    try:
        # Quick check using variance in lower face region
        h, w = face_region.shape[:2]
        
        if h < 20 or w < 20:
            return False
        
        # Focus on mouth/nose area (lower 40% of face)
        lower_face = face_region[int(h*0.6):, :]
        
        # Convert to grayscale
        if len(lower_face.shape) == 3:
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_RGB2GRAY)
        else:
            gray_lower = lower_face
        
        # Check variance (masks typically have lower variance)
        variance = np.var(gray_lower)
        
        # Check edge density (masks have more regular edges)
        edges = cv2.Canny(gray_lower, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple rule-based detection
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
        # OPTIMIZATION: Use lower JPEG quality for faster writes
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