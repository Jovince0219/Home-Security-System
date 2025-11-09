import cv2
import numpy as np
from utils.database import add_detection
import datetime
import os
import threading
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imutils import resize
import warnings
import base64
warnings.filterwarnings('ignore')

class MotionDetector:
    def __init__(self):
        # Enhanced background subtractor with better parameters
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,  # Shorter history for faster adaptation
            varThreshold=8,  # Lower threshold for more sensitivity
            detectShadows=False  # Disable shadows for cleaner detection
        )
        # Lower minimum area to catch more human movements
        self.min_contour_area = 200  # Reduced from 500
        self.motion_threshold = 0.001  # More sensitive threshold
        
        # Human-favoring detection thresholds
        self.human_min_area = 500     # Reduced from 2000
        self.human_min_confidence = 0.3  # Lower confidence threshold
        
        self.animal_min_area = 2000   # Keep animals higher
        self.animal_min_confidence = 0.7
        
        # Motion history for better human detection
        self.motion_frames = []
        self.max_motion_frames = 5
        
        self.is_detecting = False
        self.detection_thread = None
        
        # Motion without face tracking
        self.last_motion_without_face = None
        self.motion_without_face_cooldown = 30  # seconds between duplicate alerts
        
        # Training data management
        self.training_data_folder = 'static/training_data'
        os.makedirs(self.training_data_folder, exist_ok=True)
        os.makedirs(os.path.join(self.training_data_folder, 'human'), exist_ok=True)
        os.makedirs(os.path.join(self.training_data_folder, 'animal'), exist_ok=True)
        
        # Machine learning model
        self.model = None
        self.model_path = 'motion_classifier.pkl'
        self.scaler_path = 'feature_scaler.pkl'
        
        # Motion history for temporal analysis
        self.motion_history = []
        self.max_history = 10
        
        # Statistics
        self.detection_count = {'human': 0, 'animal': 0}
        self.training_data_count = {'human': 0, 'animal': 0}
        
        self.human_motion_detected = False
        self.last_human_detection_time = None
        self.human_motion_timeout = 5.0  # seconds - how long to keep face detection active
        
        # Track recent detections to avoid duplicates
        self.recent_unauthorized_faces = {}  # {face_encoding_key: timestamp}
        self.duplicate_detection_window = 60  # seconds - ignore duplicates within this window
        
        # Load model if it exists
        self.load_model()
        
    def load_model(self):
        """Load trained model if it exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("Motion detection model loaded successfully")
                # Print model info
                if hasattr(self.model, 'classes_'):
                    print(f"Model classes: {self.model.classes_}")
                    print(f"Number of features: {self.model.n_features_in_}")
            else:
                print("No trained model found. Using rule-based classification.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
    def detect_motion_without_faces(self, frame, face_detections):
        """
        Check for human motion without face detection within critical distance
        """
        try:
            # Skip if we recently alerted for this
            if self.last_motion_without_face:
                elapsed = (datetime.datetime.now() - self.last_motion_without_face).total_seconds()
                if elapsed < self.motion_without_face_cooldown:
                    return None
            
            # Check if human motion is active but no faces detected
            if (self.human_motion_detected and 
                (not face_detections or len(face_detections) == 0)):
                
                # Get the latest motion detection
                motion_detected, detections, fg_mask, human_detected = self.detect_motion(frame)
                
                if human_detected and detections:
                    # Find human detections
                    human_detections = [d for d in detections if d['type'] == 'human']
                    
                    if human_detections:
                        # Use the largest human detection
                        largest_human = max(human_detections, key=lambda x: x['area'])
                        
                        # Estimate distance (simplified)
                        bbox = largest_human['bbox']
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        motion_area = width * height
                        
                        # Simple area-based distance estimation
                        if motion_area > 15000:  # Likely within critical distance
                            self.last_motion_without_face = datetime.datetime.now()
                            return {
                                'type': 'motion_no_face',
                                'distance_meters': 1.5,  # Approximate critical distance
                                'distance_feet': 4.9,
                                'confidence': largest_human['confidence'],
                                'bbox': bbox,
                                'zone': 'CRITICAL ZONE',
                                'trigger_alert': True,
                                'alert_level': 3
                            }
            
            return None
            
        except Exception as e:
            print(f"Error detecting motion without faces: {e}")
            return None


    def detect_motion(self, frame):
        """Enhanced motion detection that respects user distance settings"""
        try:
            # Get distance estimator to check user settings
            from utils.distance_estimation import get_distance_estimator
            distance_estimator = get_distance_estimator()
            
            # Resize frame for faster processing
            processed_frame = resize(frame, width=800)
            original_height, original_width = frame.shape[:2]
            processed_height, processed_width = processed_frame.shape[:2]
            
            # Calculate scale factors
            scale_x = original_width / processed_width
            scale_y = original_height / processed_height
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(processed_frame, learningRate=0.01)
            
            # Noise removal
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
            fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)
            
            _, fg_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            detections = []
            human_detected = False
            
            current_motion_areas = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    motion_detected = True
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    x_orig, y_orig = int(x * scale_x), int(y * scale_y)
                    w_orig, h_orig = int(w * scale_x), int(h * scale_y)
                    
                    # Calculate scaled area for distance estimation
                    scaled_area = area * scale_x * scale_y
                    
                    # Estimate distance from motion area using user settings
                    estimated_distance = self.estimate_distance_from_motion_area(scaled_area, distance_estimator.MAX_DETECTION_DISTANCE)
                    
                    # Only process if within user's max detection distance
                    if estimated_distance <= distance_estimator.MAX_DETECTION_DISTANCE:
                        current_motion_areas.append({
                            'area': float(area),
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'scaled_area': float(scaled_area),
                            'estimated_distance': float(estimated_distance)
                        })
                        
                        # Extract features
                        features = self.extract_enhanced_features(processed_frame, contour, x, y, w, h)
                        
                        # Classify motion
                        if self.model is not None:
                            motion_type, confidence = self.classify_with_ml(features)
                        else:
                            motion_type, confidence = self.classify_motion_advanced(contour, area, w, h, features)
                        
                        # Only add to detections if within user's distance and is human
                        if motion_type == 'human' and confidence >= self.human_min_confidence:
                            human_detected = True
                            # ADD human detections to the detections list so they can be used by motion-only detection
                            detections.append({
                                'type': str(motion_type),
                                'bbox': (int(x_orig), int(y_orig), int(w_orig), int(h_orig)),
                                'area': float(scaled_area),
                                'confidence': float(confidence),
                                'estimated_distance': float(estimated_distance)
                            })
                            print(f"HUMAN MOTION: Distance {estimated_distance:.2f}m, Area {scaled_area:.0f}px, Conf {confidence:.3f}")
                        
                        elif motion_type == 'animal' and scaled_area >= self.animal_min_area and confidence >= self.animal_min_confidence:
                            # Still show animal detections with boxes
                            detections.append({
                                'type': str(motion_type),
                                'bbox': (int(x_orig), int(y_orig), int(w_orig), int(h_orig)),
                                'area': float(scaled_area),
                                'confidence': float(confidence),
                                'estimated_distance': float(estimated_distance)
                            })
            
            # Temporal analysis for better human detection
            self.motion_frames.append({
                'motion_detected': motion_detected,
                'human_detected': human_detected,
                'motion_areas': current_motion_areas,
                'timestamp': datetime.datetime.now()
            })
            
            if len(self.motion_frames) > self.max_motion_frames:
                self.motion_frames.pop(0)
            
            # Force human detection for consistent motion within distance
            if motion_detected and not human_detected and len(self.motion_frames) >= 3:
                recent_motion_frames = [f for f in self.motion_frames[-3:] if f['motion_detected']]
                if len(recent_motion_frames) >= 2:
                    # Check if we have consistent motion within user's distance
                    valid_motion_areas = []
                    for frame in recent_motion_frames:
                        for area in frame['motion_areas']:
                            if area['estimated_distance'] <= distance_estimator.MAX_DETECTION_DISTANCE:
                                valid_motion_areas.append(area['scaled_area'])
                    
                    if valid_motion_areas:
                        avg_motion_area = sum(valid_motion_areas) / len(valid_motion_areas)
                        
                        if avg_motion_area > 1000:  # Reasonable human size
                            human_detected = True
                            print(f"FORCING HUMAN DETECTION: Consistent motion within {distance_estimator.MAX_DETECTION_DISTANCE}m (avg area: {avg_motion_area:.0f}px)")
            
            # Update human motion status
            if human_detected:
                self.human_motion_detected = True
                self.last_human_detection_time = datetime.datetime.now()
            else:
                if self.last_human_detection_time:
                    elapsed = (datetime.datetime.now() - self.last_human_detection_time).total_seconds()
                    if elapsed > self.human_motion_timeout:
                        self.human_motion_detected = False
            
            return motion_detected, detections, fg_mask, human_detected
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return False, [], np.zeros((100, 100), dtype=np.uint8), False

    def estimate_distance_from_motion_area(self, motion_area, max_detection_distance):
        """
        Estimate distance based on motion area relative to user's max detection distance
        """
        # Scale estimation based on user's max distance
        if motion_area > 50000:
            return max_detection_distance * 0.08   # Very close
        elif motion_area > 30000:
            return max_detection_distance * 0.17   # Close
        elif motion_area > 20000:
            return max_detection_distance * 0.25   # Critical distance
        elif motion_area > 10000:
            return max_detection_distance * 0.42   # Warning distance  
        elif motion_area > 5000:
            return max_detection_distance * 0.67   # Detection range
        else:
            return max_detection_distance * 1.1    # Beyond max distance
            
    def is_human_motion_active(self):
        """Check if human motion is currently active"""
        if not self.last_human_detection_time:
            return False
            
        elapsed = (datetime.datetime.now() - self.last_human_detection_time).total_seconds()
        return elapsed <= self.human_motion_timeout
    
    def cleanup_old_detections(self):
        """Remove old entries from recent detections cache"""
        current_time = datetime.datetime.now()
        expired_keys = []
        
        for key, timestamp in self.recent_unauthorized_faces.items():
            if (current_time - timestamp).total_seconds() > self.duplicate_detection_window:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.recent_unauthorized_faces[key]

    def extract_enhanced_features(self, frame, contour, x, y, w, h):
        """Extract comprehensive features for human/animal classification"""
        try:
            # Basic geometric features
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Advanced contour features
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Bounding box features
            rect = cv2.minAreaRect(contour)
            box_width, box_height = rect[1]
            box_aspect_ratio = max(box_width, box_height) / min(box_width, box_height) if min(box_width, box_height) > 0 else 0
            
            # Region of interest analysis
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                # Color features in different color spaces
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                
                # Mean and std in different color spaces
                bgr_mean = np.mean(roi, axis=(0, 1))
                bgr_std = np.std(roi, axis=(0, 1))
                hsv_mean = np.mean(hsv_roi, axis=(0, 1))
                lab_mean = np.mean(lab_roi, axis=(0, 1))
                
                # Texture features
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Edge density
                edges = cv2.Canny(gray_roi, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
                
                # Gradient features
                grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                mean_gradient = np.mean(gradient_magnitude) if gradient_magnitude.size > 0 else 0
                
                # Histogram of Oriented Gradients (HOG) like features
                gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1)
                magnitude, angle = cv2.cartToPolar(gx, gy)
                hist, _ = np.histogram(angle, bins=8, range=(0, 2*np.pi))
                hist = hist / (hist.sum() + 1e-6)  # Normalize
                
            else:
                # Default values for small or invalid ROIs
                bgr_mean = [0, 0, 0]
                bgr_std = [0, 0, 0]
                hsv_mean = [0, 0, 0]
                lab_mean = [0, 0, 0]
                edge_density = 0
                mean_gradient = 0
                hist = np.zeros(8)
            
            # Combine all features
            features = [
                area, aspect_ratio, extent, solidity, circularity,
                w, h, perimeter, box_aspect_ratio, edge_density, mean_gradient,
                *bgr_mean, *bgr_std, *hsv_mean, *lab_mean, *hist
            ]
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting enhanced features: {e}")
            return np.zeros(35)  # Return default feature vector

    def classify_with_ml(self, features):
        """ML classification with strong human bias"""
        try:
            features_array = np.array(features).reshape(1, -1)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(features_array)[0]
            prediction = self.model.predict(features_array)[0]
            
            # Get confidence from raw probabilities
            confidence = np.max(probabilities)
            
            # HEAVY HUMAN BIAS: If ML says animal but features suggest human, override
            if prediction == 'animal':
                # Check if this might actually be human
                if len(features) > 5:
                    area = features[0]
                    aspect_ratio = features[1]
                    height = features[6]
                    width = features[5]
                    
                    # Override to human if:
                    # 1. Reasonable size for human
                    # 2. Human-like proportions
                    # 3. Not clearly animal
                    is_human_like = (
                        area > 3000 and                    # Not too small
                        aspect_ratio < 1.0 and             # Not wider than tall
                        height > width * 1.1 and           # Taller than wide
                        confidence < 0.8                   # ML isn't very confident
                    )
                    
                    if is_human_like:
                        # Reduce confidence slightly since we're overriding
                        return 'human', confidence * 0.9
            
            return prediction, confidence
                
        except Exception as e:
            print(f"Error in ML classification: {e}")
            # Default to human with good confidence
            return 'human', 0.75

    def classify_motion_advanced(self, contour, area, width, height, features):
        """Human-biased classification - when in doubt, classify as human"""
        try:
            aspect_ratio = width / height if height > 0 else 0
            
            # Start with human bias
            human_score = 2  # Starting bonus for humans
            animal_score = 0
            
            # Aspect ratio - Humans are taller than wide
            if 0.3 <= aspect_ratio <= 0.9:  # Very wide range for humans
                human_score += 3
            elif 0.2 <= aspect_ratio <= 1.0:  # Even wider range
                human_score += 2
            elif aspect_ratio > 1.5:  # Only clearly wide objects are animals
                animal_score += 2
            elif aspect_ratio > 1.2:  # Moderately wide
                animal_score += 1
            
            # Size - Humans are generally larger
            if area > 5000:  # Medium to large size strongly favors human
                human_score += 3
            elif area > 2000:  # Small to medium could be either
                human_score += 1
            elif area < 1000:  # Very small likely animal
                animal_score += 1
            
            # Height-to-width ratio - Critical human indicator
            if height > width * 1.5:  # Clearly taller (strong human)
                human_score += 4
            elif height > width * 1.2:  # Taller than wide (human)
                human_score += 2
            elif width > height * 1.8:  # Very wide (strong animal)
                animal_score += 3
            elif width > height * 1.4:  # Wider than tall (animal)
                animal_score += 2
            
            # Contour solidity - Humans have more solid contours
            solidity = features[3] if len(features) > 3 else 0.5
            if solidity > 0.7:  # Solid contour (human)
                human_score += 2
            elif solidity < 0.4:  # Fragmented (animal)
                animal_score += 1
            
            # Motion pattern analysis
            if len(self.motion_history) > 0:
                recent_humans = sum(1 for det in self.motion_history[-3:] 
                                if det.get('type') == 'human')
                if recent_humans > 0:
                    human_score += 2  # Bonus if recent human detections
            
            # SIMPLE DECISION: If it looks even slightly human, call it human
            total_score = human_score + animal_score
            if total_score == 0:
                return 'human', 0.7  # Default to human with good confidence
                
            human_ratio = human_score / total_score
            
            # Very liberal human classification
            if human_ratio >= 0.4:  # Even slight human indication -> human
                confidence = 0.6 + (human_ratio * 0.3)  # 60-90% confidence
                return 'human', min(confidence, 0.95)
            else:  # Only classify as animal if clearly not human
                confidence = 0.5 + (animal_score / 10)  # 50-80% confidence for animals
                return 'animal', min(confidence, 0.8)
                        
        except Exception as e:
            print(f"Error in rule-based classification: {e}")
            # Always default to human when uncertain
            return 'human', 0.7

    def upload_training_image(self, image_file, label):
        """Upload a single training image"""
        try:
            if label not in ['human', 'animal']:
                return {'success': False, 'error': 'Invalid label. Use "human" or "animal".'}
            
            # Read the uploaded image
            if hasattr(image_file, 'read'):
                # File-like object
                image_bytes = image_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # File path
                frame = cv2.imread(image_file)
            
            if frame is None:
                return {'success': False, 'error': 'Could not read image file'}
            
            # Save to training data folder
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(self.training_data_folder, label, filename)
            
            cv2.imwrite(filepath, frame)
            self.training_data_count[label] += 1
            
            return {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'message': f'Training image uploaded for {label}'
            }
            
        except Exception as e:
            print(f"Error uploading training image: {e}")
            return {'success': False, 'error': str(e)}

    def get_training_images(self, category=None):
        """Get list of training images for a category"""
        try:
            images = []
            
            if category and category in ['human', 'animal']:
                categories = [category]
            else:
                categories = ['human', 'animal']
            
            for cat in categories:
                folder_path = os.path.join(self.training_data_folder, cat)
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.jpg'):
                            filepath = os.path.join(folder_path, filename)
                            # Create a web-accessible URL
                            web_path = filepath.replace('static/', '/static/')
                            images.append({
                                'category': cat,
                                'filename': filename,
                                'filepath': filepath,
                                'web_path': web_path,
                                'upload_time': os.path.getctime(filepath)
                            })
            
            # Sort by upload time (newest first)
            images.sort(key=lambda x: x['upload_time'], reverse=True)
            return images
            
        except Exception as e:
            print(f"Error getting training images: {e}")
            return []

    def delete_training_image(self, filename, category):
        """Delete a specific training image"""
        try:
            if category not in ['human', 'animal']:
                return {'success': False, 'error': 'Invalid category'}
            
            filepath = os.path.join(self.training_data_folder, category, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                self.training_data_count[category] = max(0, self.training_data_count[category] - 1)
                return {'success': True, 'message': f'Deleted {filename}'}
            else:
                return {'success': False, 'error': 'File not found'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def collect_training_data(self, frame, label):
        """Enhanced training data collection with validation"""
        try:
            if label not in ['human', 'animal']:
                return {'error': 'Invalid label. Only "human" and "animal" are supported.'}
            
            # Process frame to ensure quality
            processed_frame = resize(frame, width=640)
            
            # Check if there's significant motion in the frame
            fg_mask = self.background_subtractor.apply(processed_frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'error': 'No motion detected in frame. Please capture when movement is visible.'}
            
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < self.min_contour_area:
                return {'error': f'Motion area too small ({area:.0f} pixels). Minimum required: {self.min_contour_area} pixels.'}
            
            # Save the training sample
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(self.training_data_folder, label, filename)
            
            cv2.imwrite(filepath, frame)
            self.training_data_count[label] += 1
            
            return {
                'success': True,
                'filepath': filepath,
                'area': area,
                'message': f'Training sample collected for {label} (area: {area:.0f} pixels)'
            }
            
        except Exception as e:
            print(f"Error collecting training data: {e}")
            return {'error': str(e)}
    
    def train_model(self):
        """Enhanced model training with better validation and feature selection"""
        try:
            print("Starting enhanced motion detection training...")
            
            # Collect and prepare training data
            X, y = self.load_training_data()
            
            if len(X) == 0:
                return {'error': 'No training data found. Please collect samples first.'}
            
            if len(X) < 20:
                return {'error': f'Insufficient training data. Need at least 20 samples, but only {len(X)} found.'}
            
            # Check class balance
            unique_labels, counts = np.unique(y, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            
            if len(unique_labels) < 2:
                return {'error': 'Need training data for both Human and Animal categories.'}
            
            min_samples = min(counts)
            if min_samples < 8:
                return {'error': f'Need at least 8 samples per category. Current minimum: {min_samples}'}
            
            print(f"Training with {len(X)} samples: {label_counts}")
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            # Train optimized Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            result = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X),
                'categories': list(unique_labels),
                'category_counts': label_counts,
                'classification_report': report,
                'model_type': 'Enhanced Random Forest (Human/Animal)',
                'feature_importance': dict(zip(
                    [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
                    self.model.feature_importances_
                ))
            }
            
            print(f"Enhanced model trained successfully. Accuracy: {accuracy:.3f}")
            return result
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {'error': str(e)}
    
    def load_training_data(self):
        """Load and preprocess training data"""
        X = []
        y = []
        
        for label in ['human', 'animal']:
            folder_path = os.path.join(self.training_data_folder, label)
            
            if not os.path.exists(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(folder_path, filename)
                    
                    try:
                        # Load and preprocess image
                        frame = cv2.imread(filepath)
                        if frame is None:
                            continue
                        
                        # Resize for consistency
                        frame = resize(frame, width=640)
                        
                        # Apply motion detection to extract features
                        fg_mask = self.background_subtractor.apply(frame)
                        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Use largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            x, y_coord, w, h = cv2.boundingRect(largest_contour)
                            
                            # Extract features only if contour is significant
                            if cv2.contourArea(largest_contour) > self.min_contour_area:
                                features = self.extract_enhanced_features(frame, largest_contour, x, y_coord, w, h)
                                X.append(features)
                                y.append(label)
                                
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        continue
        
        return np.array(X), np.array(y)
    
    def start_detection(self):
        """Start continuous motion detection"""
        if not self.is_detecting:
            self.is_detecting = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            return {"success": True, "message": "Motion detection started successfully"}
        return {"success": False, "message": "Motion detection already running"}
    
    def stop_detection(self):
        """Stop continuous motion detection"""
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        return {"success": True, "message": "Motion detection stopped successfully"}
    
    def _detection_loop(self):
        """Background detection loop with error handling"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Motion detection loop started")
        
        while self.is_detecting:
            ret, frame = cap.read()
            if ret:
                try:
                    # UPDATE THIS LINE - now expecting 4 return values
                    motion_detected, detections, fg_mask, human_detected = self.detect_motion(frame)
                    
                    if motion_detected and detections:
                        for detection in detections:
                            if detection['confidence'] > 0.7:
                                print(f"Confident {detection['type']} detection: {detection['confidence']:.3f}")
                
                except Exception as e:
                    print(f"Error in detection loop: {e}")
            
            time.sleep(0.05)  # 20 FPS
        
        cap.release()
        print("Motion detection loop stopped")
    
    def save_motion_screenshot(self, frame, motion_type):
        """Save screenshot of motion detection"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"motion_{motion_type}_{timestamp}.jpg"
            filepath = os.path.join('static/screenshots', filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Error saving motion screenshot: {e}")
            return None
    
    def get_stats(self):
        """Get comprehensive motion detection statistics"""
        model_info = {}
        if self.model is not None:
            model_info = {
                'model_trained': True,
                'n_features': getattr(self.model, 'n_features_in_', 'Unknown'),
                'classes': getattr(self.model, 'classes_', []).tolist(),
                'n_estimators': getattr(self.model, 'n_estimators', 'Unknown')
            }
        else:
            model_info = {'model_trained': False}
        
        return {
            'detection_count': self.detection_count,
            'training_data_count': self.training_data_count,
            'is_detecting': self.is_detecting,
            'total_detections': sum(self.detection_count.values()),
            'total_training_samples': sum(self.training_data_count.values()),
            'available_categories': ['human', 'animal'],
            'model_info': model_info,
            'detection_settings': {
                'min_contour_area': self.min_contour_area,
                'motion_threshold': self.motion_threshold
            }
        }
    
    def clear_training_data(self, category=None):
        """Clear training data for specified category or all categories"""
        try:
            if category and category in ['human', 'animal']:
                folder_path = os.path.join(self.training_data_folder, category)
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.jpg'):
                            os.remove(os.path.join(folder_path, filename))
                    self.training_data_count[category] = 0
                return {"success": True, "message": f"Cleared {category} training data"}
            else:
                for cat in ['human', 'animal']:
                    folder_path = os.path.join(self.training_data_folder, cat)
                    if os.path.exists(folder_path):
                        for filename in os.listdir(folder_path):
                            if filename.endswith('.jpg'):
                                os.remove(os.path.join(folder_path, filename))
                        self.training_data_count[cat] = 0
                return {"success": True, "message": "Cleared all training data"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}