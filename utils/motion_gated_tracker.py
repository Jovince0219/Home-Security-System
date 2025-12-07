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
    
    def __init__(self, motion_timeout=3.0):
        self.motion_timeout = motion_timeout
        
        # Tracked faces with their states
        self.tracked_faces = {}  # tracker_id -> {state, timestamp, location, motion_active}
        
        # Face encodings to tracker ID mapping
        self.face_to_tracker = {}  # face_hash -> tracker_id
        
        # Motion tracking
        self.last_motion_time = None
        self.motion_active = False
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
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
            # Generate hash for the face encoding
            face_hash = self._get_face_hash(face_encoding)
            current_time = time.time()
            
            # First, check if this face is similar to any tracked face
            # This handles cases where the hash might be slightly different
            tracker_id = None
            closest_tracker = None
            min_distance = 0.6  # Increased tolerance for same person
            
            for existing_tracker_id, tracker_data in self.tracked_faces.items():
                if 'encoding_sample' in tracker_data:
                    # Calculate distance between encodings
                    try:
                        # Get the stored encoding sample
                        stored_encoding = np.array([float(x) for x in tracker_data['encoding_sample'].split(',')])
                        
                        # Get current encoding sample
                        current_encoding = face_encoding[:len(stored_encoding)]
                        
                        # Calculate Euclidean distance
                        distance = np.linalg.norm(stored_encoding - current_encoding)
                        
                        if distance < min_distance:
                            min_distance = distance
                            tracker_id = existing_tracker_id
                            closest_tracker = tracker_data
                    except:
                        continue
            
            # If we found a similar face, use that tracker
            if tracker_id and closest_tracker:
                print(f"🔍 Found similar face with distance {min_distance:.4f}, using tracker {tracker_id[:8]}")
                # Update the hash mapping
                self.face_to_tracker[face_hash] = tracker_id
                
                # Update last seen time and location
                closest_tracker['last_seen'] = current_time
                closest_tracker['location'] = face_location
                
                # If motion is active, return the original state
                if closest_tracker['motion_active']:
                    return {
                        'tracker_id': tracker_id,
                        'state': closest_tracker['state'],
                        'is_authorized': closest_tracker['is_authorized'],
                        'is_known_person': closest_tracker.get('is_known_person', False),
                        'name': closest_tracker.get('name', name or 'Unknown'),
                        'should_process': False,  # Don't re-process, use original state
                        'reason': 'Motion active - maintaining original state (similar face)'
                    }
                else:
                    # Motion not active, can re-evaluate
                    if (closest_tracker['is_authorized'] != is_authorized or 
                        closest_tracker.get('is_known_person') != is_known_person):
                        
                        closest_tracker['state'] = self._determine_state(is_authorized, is_known_person)
                        closest_tracker['is_authorized'] = is_authorized
                        closest_tracker['is_known_person'] = is_known_person
                        closest_tracker['name'] = name or closest_tracker.get('name', 'Unknown')
                        
                    return {
                        'tracker_id': tracker_id,
                        'state': closest_tracker['state'],
                        'is_authorized': is_authorized,
                        'is_known_person': is_known_person,
                        'name': name or 'Unknown',
                        'should_process': True,
                        'reason': 'Motion inactive - updated state (similar face)'
                    }
            
            # Check if we have an exact hash match
            elif face_hash in self.face_to_tracker:
                tracker_id = self.face_to_tracker[face_hash]
                
                if tracker_id in self.tracked_faces:
                    tracker = self.tracked_faces[tracker_id]
                    
                    # Update last seen time and location
                    tracker['last_seen'] = current_time
                    tracker['location'] = face_location
                    
                    # If motion is active, return the original state
                    if tracker['motion_active']:
                        return {
                            'tracker_id': tracker_id,
                            'state': tracker['state'],
                            'is_authorized': tracker['is_authorized'],
                            'is_known_person': tracker.get('is_known_person', False),
                            'name': tracker.get('name', name or 'Unknown'),
                            'should_process': False,  # Don't re-process, use original state
                            'reason': 'Motion active - maintaining original state (exact hash)'
                        }
                    else:
                        # Motion not active, can re-evaluate
                        if (tracker['is_authorized'] != is_authorized or 
                            tracker.get('is_known_person') != is_known_person):
                            
                            tracker['state'] = self._determine_state(is_authorized, is_known_person)
                            tracker['is_authorized'] = is_authorized
                            tracker['is_known_person'] = is_known_person
                            tracker['name'] = name or tracker.get('name', 'Unknown')
                            
                        return {
                            'tracker_id': tracker_id,
                            'state': tracker['state'],
                            'is_authorized': is_authorized,
                            'is_known_person': is_known_person,
                            'name': name or 'Unknown',
                            'should_process': True,
                            'reason': 'Motion inactive - updated state (exact hash)'
                        }
            
            # New face detection
            if self.motion_active:
                # Motion is active, create new tracker
                tracker_id = str(uuid.uuid4())
                state = self._determine_state(is_authorized, is_known_person)
                
                # Store a sample of the encoding for similarity comparison
                encoding_sample = ','.join([f"{x:.8f}" for x in face_encoding[:15]])
                
                self.tracked_faces[tracker_id] = {
                    'face_hash': face_hash,
                    'encoding_sample': encoding_sample,
                    'state': state,
                    'is_authorized': is_authorized,
                    'is_known_person': is_known_person,
                    'name': name or 'Unknown',
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'location': face_location,
                    'motion_active': True
                }
                
                self.face_to_tracker[face_hash] = tracker_id
                
                return {
                    'tracker_id': tracker_id,
                    'state': state,
                    'is_authorized': is_authorized,
                    'is_known_person': is_known_person,
                    'name': name or 'Unknown',
                    'should_process': True,
                    'reason': 'New face with motion active'
                }
            else:
                # No motion, don't create tracker
                return {
                    'tracker_id': None,
                    'state': None,
                    'is_authorized': is_authorized,
                    'is_known_person': is_known_person,
                    'name': name or 'Unknown',
                    'should_process': False,
                    'reason': 'No motion - ignoring face'
                }
    
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
        
        for tracker_id, tracker in self.tracked_faces.items():
            # Remove if no motion for timeout period OR if too old
            time_since_last_seen = current_time - tracker['last_seen']
            
            if (not tracker['motion_active'] and time_since_last_seen > self.motion_timeout) or \
            time_since_last_seen > 300:  # Remove if not seen for 5 minutes
                remove_trackers.append(tracker_id)
        
        for tracker_id in remove_trackers:
            if tracker_id in self.tracked_faces:
                face_hash = self.tracked_faces[tracker_id]['face_hash']
                if face_hash in self.face_to_tracker:
                    del self.face_to_tracker[face_hash]
                del self.tracked_faces[tracker_id]
                print(f"🧹 Cleaned up old tracker: {tracker_id[:8]}")
    
    def get_tracking_stats(self):
        """Get statistics about current tracking."""
        with self.lock:
            return {
                'total_tracked': len(self.tracked_faces),
                'motion_active': self.motion_active,
                'last_motion_time': self.last_motion_time,
                'tracked_faces': [
                    {
                        'tracker_id': tracker_id,
                        'state': data['state'],
                        'name': data['name'],
                        'first_seen': datetime.datetime.fromtimestamp(data['first_seen']).isoformat(),
                        'last_seen': datetime.datetime.fromtimestamp(data['last_seen']).isoformat(),
                        'motion_active': data['motion_active']
                    }
                    for tracker_id, data in self.tracked_faces.items()
                ]
            }
    
    def reset_tracking(self):
        """Reset all tracking data."""
        with self.lock:
            self.tracked_faces.clear()
            self.face_to_tracker.clear()
            self.last_motion_time = None
            self.motion_active = False