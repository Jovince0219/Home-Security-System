import cv2
import numpy as np

class DistanceEstimator:
    """
    Estimate distance of a person from the camera based on face size
    """
    
    def __init__(self):
        # Calibration constants
        # Average human face width in cm (measured from ear to ear)
        self.KNOWN_FACE_WIDTH_CM = 15.0  # Average adult face width
        
        # Focal length (will be calibrated)
        # This needs to be calibrated for your specific camera
        # Default value for a typical webcam at 1280x720
        self.focal_length = 600.0
        
        # Detection thresholds
        self.MAX_DETECTION_DISTANCE = 6.0  # meters (6 meters)
        self.WARNING_DISTANCE = 3.0  # meters (3 meters - close proximity)
        self.CRITICAL_DISTANCE = 1.5  # meters (1.5 meters - very close)
        
        # Camera settings (update these based on your CCTV)
        self.camera_height = 2.0  # meters (height of camera from ground)
        self.camera_angle = 0  # degrees (downward tilt angle)
        
    def calibrate_focal_length(self, known_distance_cm, face_width_pixels):
        """
        Calibrate the focal length for your specific camera
        Call this once with a person standing at a known distance
        
        Args:
            known_distance_cm: Known distance in centimeters
            face_width_pixels: Width of face in pixels at that distance
        """
        self.focal_length = (face_width_pixels * known_distance_cm) / self.KNOWN_FACE_WIDTH_CM
        print(f"Focal length calibrated to: {self.focal_length}")
        return self.focal_length
    
    def estimate_distance(self, face_width_pixels):
        """
        Estimate distance based on face width in pixels
        
        Args:
            face_width_pixels: Width of detected face in pixels
            
        Returns:
            Distance in meters
        """
        if face_width_pixels <= 0:
            return float('inf')
        
        # Distance formula: D = (W x F) / P
        # Where:
        # D = Distance to object
        # W = Known width of object (face width in cm)
        # F = Focal length
        # P = Pixel width of object in image
        
        distance_cm = (self.KNOWN_FACE_WIDTH_CM * self.focal_length) / face_width_pixels
        distance_meters = distance_cm / 100.0
        
        return distance_meters
    
    def should_trigger_alert(self, distance_meters):
        """
        Determine if an alert should be triggered based on distance
        
        Args:
            distance_meters: Estimated distance in meters
            
        Returns:
            bool: True if alert should be triggered
        """
        return distance_meters <= self.MAX_DETECTION_DISTANCE
    
    def get_alert_level(self, distance_meters):
        """
        Get alert level based on distance
        
        Args:
            distance_meters: Estimated distance in meters
            
        Returns:
            int: Alert level (1=low, 2=medium, 3=high)
        """
        if distance_meters <= self.CRITICAL_DISTANCE:
            return 3  # Critical - very close
        elif distance_meters <= self.WARNING_DISTANCE:
            return 2  # Warning - close
        elif distance_meters <= self.MAX_DETECTION_DISTANCE:
            return 1  # Info - within detection range
        else:
            return 0  # No alert - too far
    
    def get_distance_zone(self, distance_meters):
        """
        Get descriptive zone based on distance
        
        Args:
            distance_meters: Estimated distance in meters
            
        Returns:
            str: Zone description
        """
        if distance_meters <= self.CRITICAL_DISTANCE:
            return "CRITICAL ZONE"
        elif distance_meters <= self.WARNING_DISTANCE:
            return "WARNING ZONE"
        elif distance_meters <= self.MAX_DETECTION_DISTANCE:
            return "DETECTION ZONE"
        else:
            return "SAFE ZONE"
    
    def analyze_face_detection(self, face_location, is_authorized):
        """
        Analyze a face detection and determine if alert should be triggered
        
        Args:
            face_location: Face bounding box [top, right, bottom, left]
            is_authorized: Boolean indicating if person is authorized
            
        Returns:
            dict: Analysis results
        """
        top, right, bottom, left = face_location
        
        # Calculate face dimensions
        face_width_pixels = right - left
        face_height_pixels = bottom - top
        
        # Estimate distance
        distance = self.estimate_distance(face_width_pixels)
        
        # Determine if alert should be triggered
        trigger_alert = False
        alert_level = 0
        
        if not is_authorized:
            trigger_alert = self.should_trigger_alert(distance)
            if trigger_alert:
                alert_level = self.get_alert_level(distance)
        
        return {
            'distance_meters': round(distance, 2),
            'distance_feet': round(distance * 3.28084, 2),
            'face_width_pixels': face_width_pixels,
            'face_height_pixels': face_height_pixels,
            'trigger_alert': trigger_alert,
            'alert_level': alert_level,
            'zone': self.get_distance_zone(distance),
            'within_detection_range': distance <= self.MAX_DETECTION_DISTANCE
        }
    
    def update_settings(self, max_distance=None, warning_distance=None, critical_distance=None):
        """
        Update distance thresholds
        
        Args:
            max_distance: Maximum detection distance in meters
            warning_distance: Warning distance in meters
            critical_distance: Critical distance in meters
        """
        if max_distance is not None:
            self.MAX_DETECTION_DISTANCE = max_distance
        if warning_distance is not None:
            self.WARNING_DISTANCE = warning_distance
        if critical_distance is not None:
            self.CRITICAL_DISTANCE = critical_distance
            
    def estimate_distance_from_area(self, motion_area):
        """
        Estimate distance based on motion detection area
        Larger area = closer distance
        """
        # This is a simplified estimation - you may need to calibrate for your camera
        # Typical values: 
        # - Very close (1m): area > 50000 pixels
        # - Close (2m): area ~ 20000-30000 pixels  
        # - Medium (4m): area ~ 8000-15000 pixels
        # - Far (6m+): area < 5000 pixels
        
        if motion_area > 50000:
            return 1.0  # Very close
        elif motion_area > 30000:
            return 1.5  # Close to critical distance
        elif motion_area > 15000:
            return 2.5  # Within warning distance
        elif motion_area > 8000:
            return 4.0  # Within detection range
        else:
            return 6.0  # Beyond detection range


# Global instance
distance_estimator = DistanceEstimator()


def get_distance_estimator():
    """Get the global distance estimator instance"""
    return distance_estimator
