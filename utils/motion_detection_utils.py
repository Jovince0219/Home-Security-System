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
from sklearn.metrics import accuracy_score
import joblib

class MotionDetector:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.min_contour_area = 1000
        self.motion_threshold = 0.02
        self.is_detecting = False
        self.detection_thread = None
        
        self.training_data_folder = 'static/training_data'
        os.makedirs(self.training_data_folder, exist_ok=True)
        os.makedirs(os.path.join(self.training_data_folder, 'human'), exist_ok=True)
        os.makedirs(os.path.join(self.training_data_folder, 'animal'), exist_ok=True)
        
        # Machine learning model
        self.model = None
        self.model_path = 'motion_classifier.pkl'
        self.load_model()
        
        self.detection_count = {'human': 0, 'animal': 0}
        self.training_data_count = {'human': 0, 'animal': 0}
        
    def detect_motion(self, frame):
        """Detect motion in frame and classify if it's human or animal"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
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
                
                # Extract features for classification
                features = self.extract_features(frame, contour, x, y, w, h)
                
                if self.model is not None:
                    motion_type = self.classify_with_ml(features)
                else:
                    motion_type = self.classify_motion_basic(area, w/h, w, h)
                
                confidence = min(area / 10000, 1.0)  # Normalize confidence
                
                detections.append({
                    'type': motion_type,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'features': features
                })
                
                # Update statistics
                self.detection_count[motion_type] += 1
                
                # Log detection
                screenshot_path = self.save_motion_screenshot(frame, motion_type)
                alert_level = 2 if motion_type == 'human' else 1
                add_detection("motion_detection", motion_type, confidence, screenshot_path, alert_level)
        
        return motion_detected, detections, fg_mask
    
    def extract_features(self, frame, contour, x, y, w, h):
        """Extract features from detected motion for classification"""
        try:
            # Basic geometric features
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Contour features
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Perimeter and circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Motion region analysis
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                # Color features
                mean_color = np.mean(roi, axis=(0, 1))
                std_color = np.std(roi, axis=(0, 1))
                
                # Texture features (using edge density)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_roi, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Gradient features
                grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                mean_gradient = np.mean(gradient_magnitude)
            else:
                mean_color = [0, 0, 0]
                std_color = [0, 0, 0]
                edge_density = 0
                mean_gradient = 0
            
            features = [
                area, aspect_ratio, extent, solidity, circularity,
                w, h, perimeter, edge_density, mean_gradient,
                mean_color[0], mean_color[1], mean_color[2],
                std_color[0], std_color[1], std_color[2]
            ]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return [0] * 16  # Return default features
    
    def classify_with_ml(self, features):
        """Classify motion using trained ML model"""
        try:
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            return prediction
        except Exception as e:
            print(f"Error in ML classification: {e}")
            return self.classify_motion_basic(features[0], features[1], features[5], features[6])
    
    def classify_motion_basic(self, area, aspect_ratio, width, height):
        """Basic classification of detected motion - Human or Animal only"""
        # Human: typically taller than wide, medium to large size, upright posture
        if 0.3 <= aspect_ratio <= 0.8 and area > 5000 and height > width:
            return 'human'
        # Animal: various sizes, typically wider than tall or square-ish, lower to ground
        else:
            return 'animal'
    
    def collect_training_data(self, frame, label):
        """Collect training data for the specified label (human or animal only)"""
        try:
            if label not in ['human', 'animal']:
                return None
                
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(self.training_data_folder, label, filename)
            
            cv2.imwrite(filepath, frame)
            self.training_data_count[label] += 1
            
            return filepath
            
        except Exception as e:
            print(f"Error collecting training data: {e}")
            return None
    
    def train_model(self):
        """Train machine learning model with collected data - simplified for Human/Animal only"""
        try:
            print("Starting simplified motion detection training (Human/Animal only)...")
            
            # Collect all training data
            X, y = self.load_training_data()
            
            if len(X) < 10:
                return {'error': 'Not enough training data. Need at least 10 samples total (5 per category recommended).'}
            
            unique_labels, counts = np.unique(y, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            
            if len(unique_labels) < 2:
                return {'error': 'Need training data for both Human and Animal categories.'}
            
            min_samples = min(counts)
            if min_samples < 3:
                return {'error': f'Need at least 3 samples per category. Current minimum: {min_samples}'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train Random Forest classifier optimized for binary classification
            self.model = RandomForestClassifier(
                n_estimators=50,  # Reduced for faster training
                max_depth=10,     # Prevent overfitting
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            result = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X),
                'categories': ['human', 'animal'],
                'category_counts': label_counts,
                'model_type': 'Binary Classification (Human/Animal)'
            }
            
            print(f"Simplified model trained successfully. Accuracy: {accuracy:.2f}")
            return result
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {'error': str(e)}
    
    def load_training_data(self):
        """Load all training data and extract features"""
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
                        # Load image
                        frame = cv2.imread(filepath)
                        if frame is None:
                            continue
                        
                        # Apply motion detection to extract features
                        fg_mask = self.background_subtractor.apply(frame)
                        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Use largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            
                            # Extract features
                            features = self.extract_features(frame, largest_contour, x, y, w, h)
                            
                            X.append(features)
                            y.append(label)
                            
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        continue
        
        return np.array(X), np.array(y)
    
    def load_model(self):
        """Load trained model if it exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("Simplified motion detection model loaded successfully")
            else:
                print("No trained model found. Using basic Human/Animal classification.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def start_detection(self):
        """Start continuous motion detection in background thread"""
        if not self.is_detecting:
            self.is_detecting = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            return "Motion detection started successfully"
        return "Motion detection already running"
    
    def stop_detection(self):
        """Stop continuous motion detection"""
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join()
        return "Motion detection stopped successfully"
    
    def _detection_loop(self):
        """Background detection loop"""
        cap = cv2.VideoCapture(0)
        
        while self.is_detecting:
            ret, frame = cap.read()
            if ret:
                motion_detected, detections, fg_mask = self.detect_motion(frame)
                
                if motion_detected:
                    print(f"Motion detected: {len(detections)} objects")
                    for detection in detections:
                        print(f"  - {detection['type']}: {detection['confidence']:.2f}")
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        cap.release()
    
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
        """Get motion detection statistics"""
        return {
            'detection_count': self.detection_count,
            'training_data_count': self.training_data_count,
            'model_trained': self.model is not None,
            'is_detecting': self.is_detecting,
            'total_detections': sum(self.detection_count.values()),
            'total_training_samples': sum(self.training_data_count.values()),
            'available_categories': ['human', 'animal'],
            'model_type': 'Binary Classification (Human/Animal)',
            'training_requirements': {
                'minimum_total_samples': 10,
                'recommended_per_category': 5,
                'categories_required': 2
            }
        }
    
    def get_training_interface_data(self):
        """Get data for simplified training interface"""
        stats = self.get_stats()
        
        training_data = {
            'categories': [
                {
                    'name': 'human',
                    'display_name': 'Human',
                    'description': 'People walking, running, or moving',
                    'sample_count': self.training_data_count['human'],
                    'icon': '🚶',
                    'color': '#3B82F6'
                },
                {
                    'name': 'animal',
                    'display_name': 'Animal',
                    'description': 'Pets, wildlife, or any animals',
                    'sample_count': self.training_data_count['animal'],
                    'icon': '🐕',
                    'color': '#10B981'
                }
            ],
            'training_status': {
                'total_samples': stats['total_training_samples'],
                'ready_to_train': stats['total_training_samples'] >= 10,
                'model_exists': stats['model_trained'],
                'requirements_met': all([
                    self.training_data_count['human'] >= 3,
                    self.training_data_count['animal'] >= 3
                ])
            },
            'instructions': [
                "1. Capture at least 5 samples of humans moving (walking, running, etc.)",
                "2. Capture at least 5 samples of animals (pets, wildlife, etc.)",
                "3. Ensure good lighting and clear motion in your samples",
                "4. Click 'Train Model' when you have enough samples",
                "5. Test the trained model with live detection"
            ]
        }
        
        return training_data
    
    def clear_training_data(self, category=None):
        """Clear training data for specified category or all categories"""
        try:
            if category and category in ['human', 'animal']:
                # Clear specific category
                folder_path = os.path.join(self.training_data_folder, category)
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.jpg'):
                            os.remove(os.path.join(folder_path, filename))
                    self.training_data_count[category] = 0
                return f"Cleared {category} training data"
            else:
                # Clear all categories
                for cat in ['human', 'animal']:
                    folder_path = os.path.join(self.training_data_folder, cat)
                    if os.path.exists(folder_path):
                        for filename in os.listdir(folder_path):
                            if filename.endswith('.jpg'):
                                os.remove(os.path.join(folder_path, filename))
                        self.training_data_count[cat] = 0
                return "Cleared all training data"
                
        except Exception as e:
            return f"Error clearing training data: {str(e)}"
