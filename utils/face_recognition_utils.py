import face_recognition
import cv2
import numpy as np
import os
from utils.database import get_all_faces, add_detection, get_face_by_id, delete_face_image_by_index, add_face_image, get_face_images
import datetime
from utils.alert_system import AlertSystem
from utils.distance_estimation import get_distance_estimator
import uuid
from utils.recording_manager import recording_manager
import time
from utils.motion_gated_tracker import MotionGatedFaceTracker


# Global variables for face recognition (Authorized Persons)
known_face_encodings = []
known_face_names = []

# Global variables for known persons (Non-threatening but not authorized)
known_persons_encodings = []
known_persons_names = []

motion_gated_tracker = MotionGatedFaceTracker(motion_timeout=3.0, authorized_cooldown=5.0)

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


def load_known_persons():
    """Load all known persons (non-threatening but not authorized)"""
    global known_persons_encodings, known_persons_names
    known_persons_encodings = []
    known_persons_names = []
    
    try:
        from utils.database import get_all_known_persons
        persons = get_all_known_persons()
        
        for person in persons:
            try:
                encoding = np.array([float(x) for x in person['encoding'].split(',')])
                known_persons_encodings.append(encoding)
                known_persons_names.append(person['name'])
            except Exception as e:
                print(f"Error loading known person encoding for {person['name']}: {e}")
        
        print(f"✅ Loaded {len(known_persons_names)} known persons")
        
    except Exception as e:
        print(f"⚠️ Error loading known persons: {e}")
        # If table doesn't exist yet, just continue with empty lists
        known_persons_encodings = []
        known_persons_names = []
            
def detect_faces_with_motion_gate(frame, motion_detector, scale_factor=1.0):
    """
    UPDATED: Motion-gated face detection with per-face state stabilization.
    """
    global known_face_encodings, known_face_names
    global known_persons_encodings, known_persons_names
    
    print("🎯 DEBUG: Motion-gated face detection called")
    
    # Get motion status - use human motion detection status
    motion_detected = motion_detector.is_human_motion_active() if motion_detector else False
    motion_gated_tracker.update_motion_status(motion_detected)
    
    # Get distance estimator with user settings
    from utils.distance_estimation import get_distance_estimator
    distance_estimator = get_distance_estimator()
    
    # ✅ Get Twilio state from request (if available)
    twilio_enabled = True  # Default to enabled
    try:
        from flask import request
        if hasattr(request, 'form'):
            twilio_enabled = request.form.get('twilio_enabled', 'true').lower() == 'true'
    except:
        pass  # Use default if can't get from request
    
    print(f"🔍 DEBUG: TWILIO STATE IN FACE DETECTION: {'ENABLED' if twilio_enabled else 'DISABLED'}")

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    
    print(f"🔍 DEBUG: Found {len(face_locations)} face locations")
    
    if len(face_locations) == 0:
        return []
    
    # Encode all faces at once
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    print(f"🔍 DEBUG: Encoded {len(face_encodings)} faces")
    
    results = []
    
    for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
        print(f"🔍 DEBUG: Processing face {i+1}/{len(face_locations)}")
        
        # STEP 1: Estimate distance FIRST using user settings
        face_location = [top, right, bottom, left]
        face_width_pixels = right - left
        distance_meters = distance_estimator.estimate_distance(face_width_pixels)
        
        print(f"🔍 DEBUG: Face {i+1} - Distance: {distance_meters:.2f}m")
        
        # STEP 2: Check if within user's configured MAX Detection Distance
        if distance_meters > distance_estimator.MAX_DETECTION_DISTANCE:
            print(f"🔍 DEBUG: Face {i+1} - Beyond max distance, skipping")
            continue  # Skip face recognition if beyond user's max distance
        
        # STEP 3: Perform face recognition to get actual identity
        # Check against KNOWN PERSONS first (non-threatening)
        is_known_person = False
        known_person_matches = face_recognition.compare_faces(
            known_persons_encodings,
            face_encoding,
            tolerance=0.55
        )
        
        name = "Unknown"
        confidence = 0.0
        is_authorized = False
        
        if True in known_person_matches:
            # This is a Known Person - don't trigger alerts
            matched_indices = [i for i, match in enumerate(known_person_matches) if match]
            matched_encodings = [known_persons_encodings[i] for i in matched_indices]
            
            face_distances = face_recognition.face_distance(matched_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            actual_index = matched_indices[best_match_index]
            
            name = known_persons_names[actual_index]
            confidence = float(1.0 - face_distances[best_match_index])
            confidence = max(0.0, min(1.0, confidence))
            is_authorized = False
            is_known_person = True
            
            print(f"DEBUG: Known Person '{name}' detected at {distance_meters:.2f}m")
            
        else:
            # STEP 4: Check against AUTHORIZED PERSONS
            matches = face_recognition.compare_faces(
                known_face_encodings, 
                face_encoding, 
                tolerance=0.55
            )
            
            if True in matches:
                # Authorized person detected
                matched_indices = [i for i, match in enumerate(matches) if match]
                matched_encodings = [known_face_encodings[i] for i in matched_indices]
                
                face_distances = face_recognition.face_distance(matched_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                actual_index = matched_indices[best_match_index]
                
                name = known_face_names[actual_index]
                confidence = float(1.0 - face_distances[best_match_index])
                confidence = max(0.0, min(1.0, confidence))
                is_authorized = True
                
                print(f"DEBUG: Authorized face '{name}' detected at {distance_meters:.2f}m with confidence: {confidence:.3f}")
                    
            else:
                # Unauthorized person detected
                name = "Unauthorized"
                is_authorized = False
                
                # Estimate confidence based on face size/quality
                face_width_pixels = right - left
                face_height_pixels = bottom - top
                face_area = face_width_pixels * face_height_pixels
                
                if face_area > 10000:
                    confidence = 0.85
                elif face_area > 5000:
                    confidence = 0.70
                else:
                    confidence = 0.55
                
                print(f"DEBUG: Unauthorized face detected at {distance_meters:.2f}m with confidence: {confidence:.3f}")
        
        # STEP 5: MOTION-GATED TRACKING - This is the core fix
        # Pass the ACTUAL recognized identity to the tracker
        tracker_info = motion_gated_tracker.register_face(
            face_encoding=face_encoding,
            face_location=face_location,
            is_authorized=is_authorized,
            is_known_person=is_known_person,
            name=name
        )
        
        print(f"🎯 TRACKER: {tracker_info['reason']}")
        print(f"   - Tracker ID: {tracker_info['tracker_id'][:8] if tracker_info['tracker_id'] else 'None'}")
        print(f"   - Name: {tracker_info['name']}")
        print(f"   - State: {'Authorized' if tracker_info['is_authorized'] else 'Known' if tracker_info['is_known_person'] else 'Unauthorized'}")
        print(f"   - Should Process: {tracker_info['should_process']}")
        
        # IMPORTANT: Use the tracker's name and state (which should match the face recognition)
        # The tracker now maintains separate states for each unique face
        final_name = tracker_info['name']
        final_is_authorized = tracker_info['is_authorized']
        final_is_known_person = tracker_info.get('is_known_person', False)
        
        # Use distance estimator with user settings
        distance_analysis = distance_estimator.analyze_face_detection(face_location, final_is_authorized)
        
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
        if final_is_authorized:
            display_name = str(final_name)
            event_type = "authorized_face"
        elif final_is_known_person:
            display_name = f"{final_name} (Known)"
            event_type = "known_face"
        else:
            if distance_analysis['within_detection_range']:
                display_name = 'Unauthorized'
                event_type = "unauthorized_face"
            else:
                display_name = 'Too Far'
                event_type = "face_too_far"
        
        # Create result
        result = {
            'name': str(final_name),
            'confidence': float(confidence),
            'location': [int(adjusted_top), int(adjusted_right), int(adjusted_bottom), int(adjusted_left)],
            'face_covered': bool(face_covered),
            'is_authorized': bool(final_is_authorized),
            'is_known_person': bool(final_is_known_person),
            'display_name': display_name,
            'distance_meters': float(distance_analysis['distance_meters']),
            'distance_feet': float(distance_analysis['distance_feet']),
            'zone': str(distance_analysis['zone']),
            'trigger_alert': bool(distance_analysis['trigger_alert']),
            'alert_level': int(distance_analysis['alert_level']),
            'within_detection_range': bool(distance_analysis['within_detection_range']),
            'event_type': event_type,
            'tracker_id': tracker_info['tracker_id']
        }
        
        results.append(result)
        
        # STEP 6: SAVE TO ALERT_EVENTS ONLY if tracker says we should process
        if distance_analysis['within_detection_range'] and tracker_info['should_process']:
            # Generate unique event ID for this detection
            detection_event_id = f"{event_type}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Start recording for this event
            recording_path = trigger_event_recording(detection_event_id, event_type, frame, confidence, distance_analysis['distance_meters'])
            
            # Save screenshot
            screenshot_path = save_screenshot(
                frame, 
                f"{event_type}_{final_name}_{distance_analysis['distance_meters']}m_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Save to alert_events table for ALL face types
            save_to_alert_events(
                event_type=event_type,
                person_name=final_name,
                confidence=confidence,
                screenshot_path=screenshot_path,
                recording_path=recording_path,
                distance_meters=distance_analysis['distance_meters'],
                is_authorized=final_is_authorized,
                is_known_person=final_is_known_person
            )
            
            print(f"✅ Processed {event_type} event: {final_name} - Recording: {recording_path}")
        
        # STEP 7: Handle ONLY unauthorized faces (not known persons) for alerts
        if not final_is_authorized and not final_is_known_person and distance_analysis['trigger_alert'] and tracker_info['should_process']:
            print(f"🚨 DEBUG: Unauthorized face detected - should trigger alert!")
            
            # ✅ UPDATED: Pass Twilio state to trigger_security_alert
            event_id = trigger_security_alert(
                "Unauthorized Face", 
                "Unauthorized Person", 
                confidence, 
                frame,
                face_encoding,
                twilio_enabled
            )
            
            if event_id:
                status_text = "CALL TRIGGERED" if twilio_enabled else "CALL BLOCKED"
                print(f"🚨 Alert {status_text} for unauthorized face: {event_id}")
            else:
                if not twilio_enabled:
                    print(f"🔕 Alert skipped - Twilio disabled")
                else:
                    print(f"🔄 Alert skipped - face already alerted recently")
            
            # Update detection time for ongoing recording
            recording_manager.update_detection_time()
            
            # Log detection to database (this is separate from alert_events)
            try:
                alert_level = distance_analysis['alert_level']
                
                # Add detection to detections table
                add_detection(
                    "face_detection", 
                    f"Unauthorized ({distance_analysis['distance_meters']}m)", 
                    float(confidence), 
                    screenshot_path, 
                    alert_level
                )
                print(f"📝 DEBUG: Detection logged to database")
                
            except Exception as e:
                print(f"❌ Error logging detection: {e}")
        else:
            print(f"🔍 DEBUG: Face {i+1} - No alert triggered:")
            print(f"  - is_authorized: {final_is_authorized}")
            print(f"  - is_known_person: {final_is_known_person}") 
            print(f"  - trigger_alert: {distance_analysis.get('trigger_alert', False)}")
            print(f"  - should_process: {tracker_info.get('should_process', False)}")
    
    return results

def save_to_alert_events(event_type, person_name, confidence, screenshot_path, recording_path, distance_meters, is_authorized=False, is_known_person=False):
    """Save face detection event to alert_events table for ALL face types with duplicate prevention"""
    try:
        from utils.database import get_db_connection
        import uuid
        import time
        
        conn = get_db_connection()
        
        # Use current datetime from Python
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check for recent duplicate events (within 30 seconds for the same person and type)
        recent_duplicate = conn.execute('''
            SELECT event_id FROM alert_events 
            WHERE person_name = ? AND trigger_type = ? 
            AND timestamp > datetime(?, '-30 seconds')
            LIMIT 1
        ''', (person_name, event_type, current_time)).fetchone()
        
        if recent_duplicate:
            print(f"⏳ Skipping duplicate {event_type} event for {person_name} (within 30s cooldown)")
            conn.close()
            return None
        
        # Generate unique event ID
        event_id = str(uuid.uuid4())
        
        # Determine review status based on face type
        if is_authorized:
            review_status = "authorized"
            call_status = "not_required"
        elif is_known_person:
            review_status = "known_person" 
            call_status = "not_required"
        else:
            review_status = "pending"
            call_status = "pending"
        
        # Ensure recording_path is properly formatted
        if recording_path and isinstance(recording_path, str):
            # Make sure it starts with static/recordings/
            if not recording_path.startswith('static/recordings/'):
                recording_path = f"static/recordings/{os.path.basename(recording_path)}"
        else:
            recording_path = None
        
        # Insert into alert_events table with explicit timestamp
        conn.execute('''
            INSERT INTO alert_events 
            (event_id, timestamp, trigger_type, recording_filepath, review_status, call_status, person_name, confidence, distance_meters, is_authorized, is_known_person)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, 
            current_time,  # Use Python datetime instead of CURRENT_TIMESTAMP
            event_type, 
            recording_path, 
            review_status, 
            call_status,
            person_name,
            float(confidence),
            float(distance_meters),
            int(is_authorized),
            int(is_known_person)
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved to alert_events: {event_type} - {person_name} at {distance_meters}m - Time: {current_time} - Recording: {recording_path}")
        return event_id
        
    except Exception as e:
        print(f"❌ Error saving to alert_events: {e}")
        return None


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
    Detect motion without faces - NO RECORDING for motion only
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
    
def trigger_event_recording(event_id, event_type, frame, confidence=0.0, distance=0.0):
    """Trigger recording for specific event types with cooldown management"""
    try:
        from utils.recording_manager import recording_manager
        
        print(f"🎯 Attempting to trigger recording for {event_type} with confidence {confidence}")
        
        # Start recording
        recording_path = recording_manager.start_recording_for_event(event_id, event_type)
        
        if recording_path:
            print(f"🎥 Recording triggered for {event_type}: {recording_path}")
            
            # Save screenshot
            screenshot_path = save_screenshot(
                frame, 
                f"{event_type}_{distance}m_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Determine alert level and detection type based on event type
            if event_type == 'unauthorized_face':
                alert_level = 3
                person_name = "Unauthorized Person"
                detection_type = "face_detection"
            elif event_type == 'motion_detection':
                alert_level = 2
                person_name = "Motion Detected, Unauthorized Person"
                detection_type = "motion_detection"
            elif event_type == 'authorized_face':
                alert_level = 1
                person_name = "Authorized Person"
                detection_type = "face_detection"
            elif event_type == 'known_face':
                alert_level = 1
                person_name = "Known Person"
                detection_type = "face_detection"
            else:
                alert_level = 1
                person_name = f"{event_type.capitalize()} Person"
                detection_type = "face_detection"
            
            print(f"📝 Creating detection for {person_name} (type: {detection_type})")
            
            # Add to database and get the detection ID
            from utils.database import add_detection, get_db_connection
            detection_id = add_detection(
                detection_type,
                person_name,
                float(confidence), 
                screenshot_path, 
                alert_level
            )
            
            # Link recording to detection
            if detection_id and recording_path:
                conn = get_db_connection()
                # Update the recording with detection_id
                conn.execute(
                    'UPDATE recordings SET detection_id = ? WHERE file_path LIKE ?',
                    (detection_id, f'%{event_id}%')
                )
                conn.commit()
                conn.close()
                print(f"🔗 Linked recording to detection ID: {detection_id}")
            
            print(f"✅ Recording and detection saved for {event_type} (Detection ID: {detection_id})")
            return recording_path
        else:
            print(f"⏳ Recording not started for {event_type} (may be in cooldown)")
            return None
        
    except Exception as e:
        print(f"❌ Error triggering recording for {event_type}: {e}")
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
    
def trigger_security_alert(trigger_type, person_name="Unauthorized", confidence=0.0, frame=None, face_encoding=None, twilio_enabled=True):
    """Trigger security alert with Twilio calls and recording - WITH FACE TRACKING AND TWILIO CONTROL"""
    from utils.twilio_alert_system import twilio_alert_system
    
    print(f"🔍 DEBUG: trigger_security_alert called with:")
    print(f"  - trigger_type: {trigger_type}")
    print(f"  - person_name: {person_name}")
    print(f"  - confidence: {confidence}")
    print(f"  - twilio_enabled: {twilio_enabled}")
    print(f"  - face_encoding: {'Provided' if face_encoding is not None else 'None'}")
    
    # ✅ Check if Twilio is enabled
    if not twilio_enabled:
        print(f"🔕 Twilio alerts disabled - skipping call for {person_name}")
        # Still log the event but don't call
        from utils.database import add_detection
        add_detection(
            trigger_type.lower().replace(' ', '_'),
            person_name,
            confidence,
            None,  # No screenshot for blocked calls
            alert_level=3
        )
        return None
    
    # ✅ Check if we should trigger alert for this face
    if face_encoding is not None:
        should_alert = twilio_alert_system.should_trigger_alert_for_face(face_encoding)
        print(f"🔍 DEBUG: should_trigger_alert_for_face returned: {should_alert}")
        
        if not should_alert:
            print(f"🔄 Skipping alert for face - already alerted recently or answered")
            return None
    else:
        print(f"⚠️ WARNING: No face encoding provided - proceeding without face tracking")
    
    # Generate event ID
    event_id = str(uuid.uuid4())
    print(f"🔍 DEBUG: Generated event_id: {event_id}")
    
    # Start recording
    final_mp4_path = recording_manager.start_recording_for_event(event_id, trigger_type) 
        
    if final_mp4_path is None:
        print("❌ Failed to start recording, aborting alert.")
        return None
    
    print(f"🔍 DEBUG: Recording started: {final_mp4_path}")
    
    # Save screenshot
    screenshot_path = None
    if frame is not None:
        screenshot_path = save_screenshot(frame, f"unauthorized_{event_id}")
        print(f"🔍 DEBUG: Screenshot saved: {screenshot_path}")
    
    # ✅ UPDATED: Trigger escalation WITH face encoding
    print(f"🔍 DEBUG: Calling trigger_alert_escalation...")
    alert_result = twilio_alert_system.trigger_alert_escalation(
        event_id, 
        trigger_type, 
        final_mp4_path,
        face_encoding
    )
    
    print(f"🔍 DEBUG: trigger_alert_escalation returned: {alert_result}")
    
    # Log the result
    if alert_result.get('status') == 'escalation_started':
        print(f"🚨 Alert escalation started for event: {event_id}")
    elif alert_result.get('status') == 'error':
        print(f"❌ Alert failed: {alert_result.get('error')}")
    
    # Log to database
    from utils.database import add_detection
    add_detection(
        trigger_type.lower().replace(' ', '_'),
        person_name,
        confidence,
        screenshot_path,
        alert_level=3
    )
    
    print(f"✅ Detection logged to database for event: {event_id}")
    
    return event_id