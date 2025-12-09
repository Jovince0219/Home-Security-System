import time
import uuid
import numpy as np
import datetime
import threading

class MotionGatedFaceTracker:
    """
    Motion-anchored face tracking system that prevents state flipping.
    Core Logic:
    1. Face recognition triggers a state only once per appearance
    2. When a face is detected, classify it as authorized/unauthorized
    3. Immediately assign a Tracker ID to that detection
    4. Motion detection becomes the "anchor" for state stability
    5. If motion is active, the system must NOT change the person's state
    6. As long as motion continues, the original state sticks
    7. State only resets when motion stops (3 seconds timeout)
    8. After ID removal, system can perform new recognition
    """
    
    def __init__(self, motion_timeout=3.0, authorized_cooldown=5.0):
        self.motion_timeout = motion_timeout
        self.authorized_cooldown = authorized_cooldown  # Seconds after authorized person leaves
        
        # Tracked faces with their states
        self.tracked_faces = {}  # tracker_id -> {state, timestamp, encoding, motion_active}
        
        # For backward compatibility
        self.face_to_tracker = {}  # May not be used in current implementation
        
        # Motion tracking
        self.last_motion_time = None
        self.motion_active = False
        
        # Authorized person tracking
        self.last_authorized_detection_time = None
        self.authorized_person_present = False
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Face recognition threshold
        self.face_recognition_threshold = 0.55
        
    def update_motion_status(self, motion_detected):
        """Update motion detection status."""
        with self.lock:
            current_time = time.time()
            
            if motion_detected:
                self.last_motion_time = current_time
                self.motion_active = True
                
                # Reset motion timeout for all tracked faces
                for tracker_id in list(self.tracked_faces.keys()):
                    if tracker_id in self.tracked_faces:
                        self.tracked_faces[tracker_id]['motion_active'] = True
            else:
                # Check if motion has stopped for timeout period
                if self.last_motion_time and (current_time - self.last_motion_time) > self.motion_timeout:
                    self.motion_active = False
                    
                    # Remove old trackers when motion stops
                    self._cleanup_old_trackers()
    
    def register_face(self, face_encoding, face_location, is_authorized, is_known_person=False, name=None):
        """
        Register a new face detection with motion gating.
        Returns: Dictionary with tracker info and whether to process this face
        """
        with self.lock:
            current_time = time.time()
            
            # Check if we're in authorized cooldown period
            if self._is_in_authorized_cooldown(current_time) and not is_authorized and not is_known_person:
                print(f"⏳ AUTHORIZED COOLDOWN: Blocking unauthorized detection for {self._get_cooldown_remaining(current_time):.1f}s")
                return {
                    'tracker_id': None,
                    'state': None,
                    'is_authorized': False,
                    'is_known_person': False,
                    'name': 'Blocked (Auth Cooldown)',
                    'should_process': False,
                    'reason': 'Authorized cooldown active - blocking unauthorized'
                }
            
            # First, try to match this face with existing tracked faces
            matched_tracker_id = self._find_matching_face(face_encoding)
            
            if matched_tracker_id:
                # We found a matching face
                tracker = self.tracked_faces[matched_tracker_id]
                
                # Update last seen time and location
                tracker['last_seen'] = current_time
                tracker['location'] = face_location
                
                # Update authorized person status
                if is_authorized:
                    self.last_authorized_detection_time = current_time
                    self.authorized_person_present = True
                
                print(f"🔍 FACE MATCH: Found existing tracker {matched_tracker_id[:8]} for {name or 'Unknown'}")
                
                # If motion is active, return the original state for THIS FACE
                if tracker['motion_active']:
                    return {
                        'tracker_id': matched_tracker_id,
                        'state': tracker['state'],
                        'is_authorized': tracker['is_authorized'],
                        'is_known_person': tracker.get('is_known_person', False),
                        'name': tracker.get('name', name or 'Unknown'),
                        'should_process': False,  # Don't re-process, use original state
                        'reason': 'Motion active - maintaining original state for matched face'
                    }
                else:
                    # Motion not active, can update this face's state
                    # Only update if the classification changed
                    state_changed = (
                        tracker['is_authorized'] != is_authorized or 
                        tracker.get('is_known_person') != is_known_person or
                        tracker.get('name', '') != (name or '')
                    )
                    
                    if state_changed:
                        tracker['state'] = self._determine_state(is_authorized, is_known_person)
                        tracker['is_authorized'] = is_authorized
                        tracker['is_known_person'] = is_known_person
                        tracker['name'] = name or tracker.get('name', 'Unknown')
                        tracker['last_updated'] = current_time
                        
                    return {
                        'tracker_id': matched_tracker_id,
                        'state': tracker['state'],
                        'is_authorized': is_authorized,
                        'is_known_person': is_known_person,
                        'name': name or 'Unknown',
                        'should_process': state_changed,  # Only process if state changed
                        'reason': f'Motion inactive - {"updated" if state_changed else "maintained"} state for matched face'
                    }
            
            # New face detection (no match found)
            if self.motion_active:
                # Check if we should block this detection due to authorized presence
                if self._should_block_detection(is_authorized, is_known_person, current_time):
                    return {
                        'tracker_id': None,
                        'state': None,
                        'is_authorized': is_authorized,
                        'is_known_person': is_known_person,
                        'name': 'Blocked (Auth Present)',
                        'should_process': False,
                        'reason': 'Authorized person present or recent - blocking new unauthorized'
                    }
                
                # Motion is active, create new tracker for THIS NEW FACE
                tracker_id = str(uuid.uuid4())
                state = self._determine_state(is_authorized, is_known_person)
                
                # Update authorized person status
                if is_authorized:
                    self.last_authorized_detection_time = current_time
                    self.authorized_person_present = True
                
                # Store the full encoding for future matching
                encoding_str = ','.join([f"{x:.8f}" for x in face_encoding])
                
                self.tracked_faces[tracker_id] = {
                    'encoding': encoding_str,
                    'encoding_array': face_encoding.copy(),  # Store numpy array for matching
                    'state': state,
                    'is_authorized': is_authorized,
                    'is_known_person': is_known_person,
                    'name': name or 'Unknown',
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_updated': current_time,
                    'location': face_location,
                    'motion_active': True
                }
                
                # For backward compatibility
                self.face_to_tracker[encoding_str] = tracker_id
                
                print(f"🎯 NEW TRACKER: Created tracker {tracker_id[:8]} for {name or 'Unknown'}")
                
                return {
                    'tracker_id': tracker_id,
                    'state': state,
                    'is_authorized': is_authorized,
                    'is_known_person': is_known_person,
                    'name': name or 'Unknown',
                    'should_process': True,  # Process this new face
                    'reason': 'New face with motion active'
                }
            else:
                # No motion, don't create tracker for new faces
                return {
                    'tracker_id': None,
                    'state': None,
                    'is_authorized': is_authorized,
                    'is_known_person': is_known_person,
                    'name': name or 'Unknown',
                    'should_process': False,  # Don't process without motion
                    'reason': 'No motion - ignoring new face'
                }
                
    def _should_block_detection(self, is_authorized, is_known_person, current_time):
        """
        Determine if we should block a new detection.
        Blocks unauthorized if authorized person was recently detected.
        """
        # Never block authorized or known persons
        if is_authorized or is_known_person:
            return False
        
        # Check if authorized person is currently tracked
        for tracker_id, tracker in self.tracked_faces.items():
            if tracker.get('is_authorized', False):
                time_since_seen = current_time - tracker['last_seen']
                if time_since_seen < 2.0:  # Authorized person seen within 2 seconds
                    print(f"🚫 BLOCKING: Authorized person {tracker['name']} present (seen {time_since_seen:.1f}s ago)")
                    return True
        
        # Check cooldown period
        if self._is_in_authorized_cooldown(current_time):
            return True
        
        return False
    
    def _is_in_authorized_cooldown(self, current_time):
        """Check if we're in the authorized cooldown period."""
        if not self.last_authorized_detection_time:
            return False
        
        time_since_authorized = current_time - self.last_authorized_detection_time
        return time_since_authorized < self.authorized_cooldown
    
    def _get_cooldown_remaining(self, current_time):
        """Get remaining cooldown time in seconds."""
        if not self.last_authorized_detection_time:
            return 0
        
        time_since_authorized = current_time - self.last_authorized_detection_time
        return max(0, self.authorized_cooldown - time_since_authorized)
                
    def _find_matching_face(self, face_encoding):
        """
        Find if this face matches any existing tracked face.
        Returns tracker_id if match found, None otherwise.
        """
        if len(self.tracked_faces) == 0:
            return None
        
        best_match_id = None
        best_distance = float('inf')
        
        for tracker_id, tracker_data in self.tracked_faces.items():
            if 'encoding_array' in tracker_data:
                try:
                    # Get the stored encoding
                    stored_encoding = tracker_data['encoding_array']
                    
                    # Calculate face distance (lower = more similar)
                    distance = np.linalg.norm(stored_encoding - face_encoding)
                    
                    # If distance is very small, it's likely the same person
                    if distance < best_distance:
                        best_distance = distance
                        best_match_id = tracker_id
                except Exception as e:
                    print(f"Error calculating face distance: {e}")
                    continue
        
        # Only return match if distance is below threshold
        if best_match_id and best_distance < 0.6:
            print(f"🔍 Face match found with distance {best_distance:.4f} (threshold: 0.6)")
            return best_match_id
        
        return None
    
    def _determine_state(self, is_authorized, is_known_person):
        """Determine the state string based on authorization status."""
        if is_authorized:
            return "authorized"
        elif is_known_person:
            return "known"
        else:
            return "unauthorized"
    
    def _get_face_hash(self, face_encoding):
        """Generate a stable hash for face encoding."""
        if len(face_encoding) == 0:
            return "empty"
        
        try:
            # Normalize the encoding for better stability
            normalized = face_encoding / np.linalg.norm(face_encoding)
            
            # Use more values for better discrimination (increase from 5 to 10-15)
            # Take every 3rd value to get a good sample across the encoding
            sample_indices = list(range(0, min(len(normalized), 30), 3))
            encoding_sample = normalized[sample_indices]
            
            # Create a hash string with more precision
            encoding_str = ','.join([f"{x:.8f}" for x in encoding_sample])
            
            # Use Python's built-in hash with a fixed seed for consistency
            return str(hash(encoding_str))
        except Exception as e:
            # Fallback to simpler hash if normalization fails
            encoding_str = ','.join([f"{x:.6f}" for x in face_encoding[:10]])
            return str(hash(encoding_str))
    
    def _cleanup_old_trackers(self):
        """Remove trackers that haven't been seen for a while."""
        current_time = time.time()
        remove_trackers = []
        
        # Check if authorized persons are still present
        authorized_present = False
        for tracker_id, tracker in self.tracked_faces.items():
            if tracker.get('is_authorized', False):
                time_since_seen = current_time - tracker['last_seen']
                if time_since_seen < 2.0:  # Authorized person seen within 2 seconds
                    authorized_present = True
                    break
        
        # Update authorized presence status
        self.authorized_person_present = authorized_present
        if not authorized_present and self.last_authorized_detection_time:
            # Update last authorized detection time if we haven't seen one recently
            time_since_authorized = current_time - self.last_authorized_detection_time
            if time_since_authorized > self.authorized_cooldown:
                self.last_authorized_detection_time = None
        
        for tracker_id, tracker in self.tracked_faces.items():
            # Remove if no motion for timeout period OR if too old
            time_since_last_seen = current_time - tracker['last_seen']
            
            if (not tracker['motion_active'] and time_since_last_seen > self.motion_timeout) or \
               time_since_last_seen > 300:  # Remove if not seen for 5 minutes
                remove_trackers.append(tracker_id)
        
        for tracker_id in remove_trackers:
            if tracker_id in self.tracked_faces:
                print(f"🧹 Cleaned up old tracker: {tracker_id[:8]} ({self.tracked_faces[tracker_id]['name']})")
                del self.tracked_faces[tracker_id]
    
    def get_tracking_stats(self):
        """Get statistics about current tracking."""
        with self.lock:
            current_time = time.time()
            return {
                'total_tracked': len(self.tracked_faces),
                'motion_active': self.motion_active,
                'last_motion_time': self.last_motion_time,
                'authorized_present': self.authorized_person_present,
                'last_authorized_time': self.last_authorized_detection_time,
                'cooldown_remaining': self._get_cooldown_remaining(current_time) if self.last_authorized_detection_time else 0,
                'tracked_faces': [
                    {
                        'tracker_id': tracker_id,
                        'state': data['state'],
                        'name': data['name'],
                        'first_seen': datetime.datetime.fromtimestamp(data['first_seen']).isoformat(),
                        'last_seen': datetime.datetime.fromtimestamp(data['last_seen']).isoformat(),
                        'motion_active': data['motion_active'],
                        'is_authorized': data.get('is_authorized', False)
                    }
                    for tracker_id, data in self.tracked_faces.items()
                ]
            }
    
    def reset_tracking(self):
        """Reset all tracking data - safe version."""
        with self.lock:
            # Clear all data structures
            self.tracked_faces.clear()
            self.face_to_tracker.clear()
            self.last_motion_time = None
            self.motion_active = False
            self.last_authorized_detection_time = None
            self.authorized_person_present = False
            print("✅ Tracking system completely reset")