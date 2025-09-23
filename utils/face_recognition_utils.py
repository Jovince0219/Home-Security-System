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
            # Convert string back to numpy array
            encoding = np.array([float(x) for x in face['encoding'].split(',')])
            known_face_encodings.append(encoding)
            known_face_names.append(face['name'])
        except Exception as e:
            print(f"Error loading face encoding for {face['name']}: {e}")

def register_face_from_camera():
    """Capture face from camera and return encoding"""
    cap = cv2.VideoCapture(0)
    
    print("Face Registration Mode")
    print("Press SPACE to capture photo, ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display instructions on frame
        cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Find faces in current frame and draw rectangles
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow('Face Registration - Press SPACE to capture, ESC to cancel', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            return None, None
        elif key == 32:  # SPACE key
            if len(face_locations) == 1:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                cap.release()
                cv2.destroyAllWindows()
                return face_encoding, frame
            else:
                print("Please ensure exactly one face is visible in the frame")
    
    cap.release()
    cv2.destroyAllWindows()
    return None, None

def detect_faces_in_frame(frame):
    """Detect and recognize faces in a frame with enhanced JSON serialization and better box fitting"""
    global known_face_encodings, known_face_names
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    authorized_detected = False
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if face matches known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0.0
        
        if True in matches:
            # Find the best match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1.0 - face_distances[best_match_index]
                authorized_detected = True
        
        # Check for face coverings (enhanced detection)
        face_region = rgb_frame[top:bottom, left:right]
        face_covered = detect_face_covering(face_region)
        
        is_authorized = name != "Unknown"
        
        # If authorized person detected, prioritize them over unauthorized
        if authorized_detected and not is_authorized:
            continue  # Skip unauthorized faces when authorized person is present
        
        face_width = right - left
        face_height = bottom - top
        
        # More precise face fitting - reduce width more and adjust height better
        width_reduction = int(face_width * 0.20)  # Reduce width by 20%
        height_top_reduction = int(face_height * 0.15)  # Reduce top by 15%
        height_bottom_reduction = int(face_height * 0.05)  # Reduce bottom by 5%
        
        adjusted_left = left + width_reduction
        adjusted_right = right - width_reduction
        adjusted_top = top + height_top_reduction
        adjusted_bottom = bottom - height_bottom_reduction
        
        # Ensure boundaries are valid
        adjusted_left = max(0, adjusted_left)
        adjusted_right = min(rgb_frame.shape[1], adjusted_right)
        adjusted_top = max(0, adjusted_top)
        adjusted_bottom = min(rgb_frame.shape[0], adjusted_bottom)
        
        results.append({
            'name': str(name),
            'confidence': float(confidence),  # Convert numpy float to Python float
            'location': [int(adjusted_top), int(adjusted_right), int(adjusted_bottom), int(adjusted_left)],
            'face_covered': bool(face_covered),  # Convert numpy bool to Python bool
            'is_authorized': bool(is_authorized),  # Convert numpy bool to Python bool
            'display_name': str(name) if is_authorized else 'Unauthorized'
        })
        
        # Log detection and send alert for unauthorized faces
        screenshot_path = save_screenshot(frame, f"face_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        alert_level = 1 if is_authorized else 2
        add_detection("face_detection", str(name), float(confidence), screenshot_path, alert_level)
        
        if not is_authorized:
            try:
                alert_system = AlertSystem()
                alert_result = alert_system.send_alert(alert_level, "face_detection", str(name), screenshot_path=screenshot_path)
                print(f"Unauthorized face alert sent: {alert_result}")
            except Exception as e:
                print(f"Error sending unauthorized face alert: {e}")
    
    return results

def detect_face_covering(face_region):
    """Enhanced detection for face coverings like masks, helmets, scarves"""
    if face_region.size == 0:
        return False
        
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        
        # Multiple detection methods
        covering_indicators = []
        
        # Method 1: Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        covering_indicators.append(edge_density > 0.3)
        
        # Method 2: Color variance in mouth/nose region
        h, w = gray.shape
        mouth_region = gray[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            color_variance = np.var(mouth_region)
            covering_indicators.append(color_variance < 200)
        
        # Method 3: Texture analysis using Local Binary Pattern
        try:
            # Simple texture analysis
            texture_score = np.std(gray)
            covering_indicators.append(texture_score > 40)
        except:
            pass
        
        # Method 4: Symmetry analysis (masks/helmets often have regular patterns)
        try:
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            if left_half.shape == right_half.shape:
                symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
                covering_indicators.append(symmetry_diff < 30)
        except:
            pass
        
        # Combine indicators - if 2 or more methods indicate covering
        covering_score = sum(covering_indicators)
        return covering_score >= 2
        
    except Exception as e:
        print(f"Error in face covering detection: {e}")
        return False

def save_screenshot(frame, filename):
    """Save screenshot to screenshots folder"""
    try:
        filepath = os.path.join('static/screenshots', f"{filename}.jpg")
        cv2.imwrite(filepath, frame)
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
    """Capture multiple faces from camera for better training with enhanced UI"""
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
        
        # Find faces in current frame and draw better fitting rectangles
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            # Apply same better fitting logic as detection
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
        if key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE key
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
                print("Please ensure exactly one face is visible in the frame")
        elif key == 13:  # ENTER key
            if len(captured_faces) > 0:
                break
            else:
                print("Please capture at least one face before finishing")
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_faces

def delete_face_image(face_id, image_index):
    """Delete a specific image from a face registration"""
    try:
        face = get_face_by_id(face_id)
        if not face:
            return {'error': 'Face not found'}
        
        # Get all images for this face
        face_images = get_face_images(face_id)
        
        if image_index >= len(face_images):
            return {'error': 'Image index out of range'}
        
        # Delete the specific image file
        image_path = face_images[image_index]['image_path']
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Update database to remove this image reference
        delete_face_image_by_index(face_id, image_index)
        
        return {'success': True, 'message': 'Image deleted successfully'}
        
    except Exception as e:
        return {'error': str(e)}

def create_privacy_thumbnail(image_path, blur_faces=True):
    """Create a privacy-protected thumbnail with blurred faces"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        if blur_faces:
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            # Blur each face region
            for (top, right, bottom, left) in face_locations:
                # Extract face region
                face_region = image[top:bottom, left:right]
                
                # Apply strong blur
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                
                # Replace face region with blurred version
                image[top:bottom, left:right] = blurred_face
        
        # Create thumbnail
        height, width = image.shape[:2]
        max_size = 200
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        thumbnail = cv2.resize(image, (new_width, new_height))
        
        # Save thumbnail
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

def register_face_multiple_images(name, captured_faces):
    """Register multiple face images for a single person"""
    try:
        if not captured_faces:
            return {'error': 'No faces captured'}
        
        registered_images = []
        
        for i, face_data in enumerate(captured_faces):
            # Save image
            timestamp = face_data['timestamp']
            filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join('static/faces', filename)
            
            cv2.imwrite(image_path, face_data['frame'])
            
            # Convert encoding to string
            encoding_str = ','.join(map(str, face_data['encoding']))
            
            # Add to database
            face_id = add_face_image(name, encoding_str, image_path)
            
            registered_images.append({
                'face_id': face_id,
                'image_path': image_path,
                'timestamp': timestamp
            })
        
        # Reload known faces
        load_known_faces()
        
        return {
            'success': True,
            'message': f'Successfully registered {len(registered_images)} images for {name}',
            'registered_images': registered_images
        }
        
    except Exception as e:
        return {'error': str(e)}
