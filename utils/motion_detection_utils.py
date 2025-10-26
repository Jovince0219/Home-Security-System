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
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        # Lower minimum area to catch more human movements
        self.min_contour_area = 500
        self.motion_threshold = 0.005  # More sensitive
        
        # Human-favoring detection thresholds
        self.human_min_area = 2000     # Lower minimum for humans
        self.human_min_confidence = 0.5  # Lower confidence threshold
        
        self.animal_min_area = 3000    # Higher minimum for animals  
        self.animal_min_confidence = 0.7  # Higher confidence threshold
        
        self.is_detecting = False
        self.detection_thread = None
        
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

    def detect_motion(self, frame):
        """Balanced motion detection without human bias"""
        try:
            # Resize frame for faster processing
            processed_frame = resize(frame, width=640)
            original_height, original_width = frame.shape[:2]
            processed_height, processed_width = processed_frame.shape[:2]
            
            # Calculate scale factors
            scale_x = original_width / processed_width
            scale_y = original_height / processed_height
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(processed_frame, learningRate=0.001)
            
            # Noise removal
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
            fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
            
            # Balanced threshold
            _, fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            detections = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    motion_detected = True
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    x_orig, y_orig = int(x * scale_x), int(y * scale_y)
                    w_orig, h_orig = int(w * scale_x), int(h * scale_y)
                    
                    # Extract features
                    features = self.extract_enhanced_features(processed_frame, contour, x, y, w, h)
                    
                    # Classify motion with human bias
                    if self.model is not None:
                        motion_type, confidence = self.classify_with_ml(features)
                    else:
                        motion_type, confidence = self.classify_motion_advanced(contour, area, w, h, features)
                    
                    # POST-CLASSIFICATION HUMAN BIAS
                    # If classified as animal but has human characteristics, reconsider
                    if motion_type == 'animal' and confidence < 0.8:
                        # Check if this might be human
                        aspect_ratio = w / h if h > 0 else 0
                        scaled_area = area * scale_x * scale_y
                        
                        could_be_human = (
                            scaled_area > self.human_min_area and
                            0.3 <= aspect_ratio <= 1.0 and
                            h > w * 1.1  # Taller than wide
                        )
                        
                        if could_be_human:
                            motion_type = 'human'
                            confidence = max(confidence, 0.6)  # Boost to reasonable confidence
                    
                    # Apply different thresholds
                    scaled_area = area * scale_x * scale_y
                    if motion_type == 'human':
                        if scaled_area >= self.human_min_area and confidence >= self.human_min_confidence:
                            detections.append({
                                'type': motion_type,
                                'bbox': (x_orig, y_orig, w_orig, h_orig),
                                'area': scaled_area,
                                'confidence': confidence,
                                'features': features,
                                'contour': contour
                            })
                    else:  # animal
                        if scaled_area >= self.animal_min_area and confidence >= self.animal_min_confidence:
                            detections.append({
                                'type': motion_type,
                                'bbox': (x_orig, y_orig, w_orig, h_orig),
                                'area': scaled_area,
                                'confidence': confidence,
                                'features': features,
                                'contour': contour
                            })
                    
                    # Log detection
                    if ((motion_type == 'human' and confidence >= self.human_min_confidence) or
                        (motion_type == 'animal' and confidence >= self.animal_min_confidence)):
                        
                        self.detection_count[motion_type] += 1
                        screenshot_path = self.save_motion_screenshot(frame, motion_type)
                        alert_level = 2 if motion_type == 'human' else 1
                        add_detection("motion_detection", motion_type, confidence, screenshot_path, alert_level)
                
            return motion_detected, detections, fg_mask
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return False, [], np.zeros((100, 100), dtype=np.uint8)

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
                    motion_detected, detections, fg_mask = self.detect_motion(frame)
                    
                    if motion_detected and detections:
                        for detection in detections:
                            if detection['confidence'] > 0.7:  # Only log confident detections
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