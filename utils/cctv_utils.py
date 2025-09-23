import json
import os
import time
import threading
from datetime import datetime

class CCTVController:
    def __init__(self):
        self.presets_file = 'data/camera_presets.json'
        self.settings_file = 'data/camera_settings.json'
        self.current_position = {'pan': 0, 'tilt': 0}  # Simulated position
        self.is_moving = False
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Load existing presets and settings
        self.presets = self.load_presets_from_file()
        self.settings = self.load_settings_from_file()
    
    def move_camera(self, action, speed=5):
        """Move camera in specified direction"""
        if self.is_moving:
            return {'status': 'busy', 'message': 'Camera is already moving'}
        
        self.is_moving = True
        
        try:
            # Simulate camera movement (in real implementation, this would control actual hardware)
            movement_amount = speed * 2  # Degrees per speed unit
            
            if action == 'pan_left':
                self.current_position['pan'] = max(-180, self.current_position['pan'] - movement_amount)
            elif action == 'pan_right':
                self.current_position['pan'] = min(180, self.current_position['pan'] + movement_amount)
            elif action == 'tilt_up':
                self.current_position['tilt'] = min(90, self.current_position['tilt'] + movement_amount)
            elif action == 'tilt_down':
                self.current_position['tilt'] = max(-90, self.current_position['tilt'] - movement_amount)
            elif action == 'center':
                self.current_position = {'pan': 0, 'tilt': 0}
            else:
                return {'status': 'error', 'message': 'Invalid action'}
            
            # Simulate movement time
            time.sleep(0.5)
            
            return {
                'status': 'success',
                'message': f'Camera moved {action}',
                'position': self.current_position.copy()
            }
        
        finally:
            self.is_moving = False
    
    def save_preset(self, preset_name):
        """Save current camera position as preset"""
        try:
            self.presets[preset_name] = {
                'position': self.current_position.copy(),
                'created_at': datetime.now().isoformat(),
                'settings': self.settings.copy()
            }
            
            self.save_presets_to_file()
            
            return {
                'status': 'success',
                'message': f'Preset "{preset_name}" saved successfully'
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def load_preset(self, preset_name):
        """Load camera position from preset"""
        try:
            if preset_name not in self.presets:
                return {'status': 'error', 'message': 'Preset not found'}
            
            preset = self.presets[preset_name]
            self.current_position = preset['position'].copy()
            
            # Also restore camera settings if available
            if 'settings' in preset:
                self.settings.update(preset['settings'])
                self.save_settings_to_file()
            
            return {
                'status': 'success',
                'message': f'Preset "{preset_name}" loaded successfully',
                'position': self.current_position.copy()
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_presets(self):
        """Get all saved presets"""
        presets_list = []
        for name, data in self.presets.items():
            presets_list.append({
                'name': name,
                'position': data['position'],
                'created_at': data['created_at']
            })
        
        return presets_list
    
    def delete_preset(self, preset_name):
        """Delete a preset"""
        try:
            if preset_name not in self.presets:
                return {'status': 'error', 'message': 'Preset not found'}
            
            del self.presets[preset_name]
            self.save_presets_to_file()
            
            return {
                'status': 'success',
                'message': f'Preset "{preset_name}" deleted successfully'
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_camera_settings(self):
        """Get current camera settings"""
        return {
            'position': self.current_position.copy(),
            'settings': self.settings.copy(),
            'is_moving': self.is_moving
        }
    
    def update_camera_settings(self, new_settings):
        """Update camera settings"""
        try:
            self.settings.update(new_settings)
            self.save_settings_to_file()
            
            # In real implementation, apply settings to actual camera hardware
            
            return {
                'status': 'success',
                'message': 'Camera settings updated successfully',
                'settings': self.settings.copy()
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def load_presets_from_file(self):
        """Load presets from file"""
        try:
            if os.path.exists(self.presets_file):
                with open(self.presets_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading presets: {e}")
        
        return {}
    
    def save_presets_to_file(self):
        """Save presets to file"""
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")
    
    def load_settings_from_file(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        # Default settings
        return {
            'brightness': 50,
            'contrast': 50,
            'saturation': 50,
            'zoom': 100,
            'night_vision': False,
            'auto_tracking': False
        }
    
    def save_settings_to_file(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def auto_track_object(self, object_position):
        """Automatically track detected object"""
        if not self.settings.get('auto_tracking', False):
            return {'status': 'disabled', 'message': 'Auto-tracking is disabled'}
        
        try:
            # Calculate required camera movement to center object
            # This is a simplified implementation
            center_x, center_y = 320, 240  # Assuming 640x480 resolution
            obj_x, obj_y = object_position
            
            pan_adjustment = (obj_x - center_x) * 0.1  # Adjust sensitivity
            tilt_adjustment = (center_y - obj_y) * 0.1
            
            self.current_position['pan'] += pan_adjustment
            self.current_position['tilt'] += tilt_adjustment
            
            # Keep within bounds
            self.current_position['pan'] = max(-180, min(180, self.current_position['pan']))
            self.current_position['tilt'] = max(-90, min(90, self.current_position['tilt']))
            
            return {
                'status': 'success',
                'message': 'Camera adjusted to track object',
                'position': self.current_position.copy()
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
