import cv2
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from collections import deque
import logging
import imageio.v3 as iio

class RecordingManager:
    def __init__(self):
        # Setup logging FIRST
        self.setup_logging()
        
        self.recording = False
        self.current_event_id = None
        self.video_writer = None
        self.pre_buffer = deque(maxlen=75)  # 5 seconds of pre-buffer at 15fps
        self.recording_thread = None
        self.stop_recording_flag = False
        self.last_detection_time = None
        self.recording_timeout = 10  # Stop 10s after last detection
        self.start_time = time.time()
        self.current_filepath = None  # Tracks the raw .avi filepath
        
        self.fps = 15
        self.resolution = (640, 480)
        
        # ==========================================================
        # ✅ STEP 1: Record in the MOST RELIABLE format (XVID/AVI)
        # This fixes the "Cannot open file - codec issue" error.
        # ==========================================================
        self.codec = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID'
        self.file_extension = '.avi'                 # Use '.avi' container
        
        self.recordings_folder = 'static/recordings'
        os.makedirs(self.recordings_folder, exist_ok=True)
        
        # Removed emoji to prevent UnicodeEncodeError
        self.logger.info("RecordingManager initialized with XVID codec and AVI format")
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # Added encoding='utf-8' to file handler
                logging.FileHandler('recording_manager.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start_recording_for_event(self, event_id, trigger_type):
        """Start recording for a new event"""
        if self.recording:
            if self.current_event_id == event_id:
                self.last_detection_time = time.time()
                # Return the *final* path the browser will look for
                if self.current_filepath:
                    return self.current_filepath.replace(self.file_extension, '.mp4')
                return None
            else:
                # Stop the previous recording if a new event comes in
                self.stop_recording()
                
        self.current_event_id = event_id
        self.recording = True
        self.stop_recording_flag = False
        self.last_detection_time = time.time()
        self.start_time = time.time()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # We use event_id in the name to make it unique
        filename = f"recording_{event_id}_{timestamp}{self.file_extension}"
        filepath = os.path.join(self.recordings_folder, filename)
        
        filepath = filepath.replace('\\', '/')
        
        self.current_filepath = filepath  # Store the raw .avi filepath
        
        self.video_writer = cv2.VideoWriter(
            filepath, 
            self.codec, 
            self.fps, 
            self.resolution,
            isColor=True
        )
        
        if not self.video_writer.isOpened():
            self.logger.error(f"❌ Failed to initialize video writer for {filepath} (Codec: XVID)")
            self.recording = False
            return None
        
        self._write_pre_buffer()
        
        self.recording_thread = threading.Thread(target=self._recording_monitor)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.logger.info(f"🎥 Started raw recording for event {event_id}: {filepath}")
        
        # Return the *final* .mp4 path that will be created
        return filepath.replace(self.file_extension, '.mp4')
        
    def add_frame(self, frame):
        """Add frame to recording or pre-buffer"""
        if frame is None:
            return
            
        try:
            # Resize frame to match resolution
            frame_resized = cv2.resize(frame, self.resolution)
            
            if self.recording and self.video_writer and self.video_writer.isOpened():
                # Write frame in BGR format (OpenCV expectation)
                self.video_writer.write(frame_resized)
            else:
                # Store in pre-buffer
                self.pre_buffer.append(frame_resized.copy())
                
        except Exception as e:
            self.logger.error(f"Error adding frame to recording: {e}")
            
    def _write_pre_buffer(self):
        """Write pre-buffer frames to current recording"""
        for frame in self.pre_buffer:
            if self.video_writer and self.video_writer.isOpened():
                try:
                    self.video_writer.write(frame)
                except Exception as e:
                    self.logger.error(f"Error writing pre-buffer frame: {e}")
                    
    def _recording_monitor(self):
        """Monitor recording and stop when no detections for timeout period"""
        
        MINIMUM_RECORDING_DURATION = 10
        
        while self.recording and not self.stop_recording_flag:
                current_time = time.time()

                if self.last_detection_time is None: # Safety check
                    self.last_detection_time = current_time
                if self.start_time is None: # Safety check
                    self.start_time = current_time

                time_since_last_detection = current_time - self.last_detection_time
                time_since_start = current_time - self.start_time

                if (time_since_last_detection > self.recording_timeout and
                    time_since_start > MINIMUM_RECORDING_DURATION):

                    self.stop_recording()
                    break

                time.sleep(1)
            
    def stop_recording(self):
        """Stop current recording, release the file, and convert it to MP4."""
        if self.recording and self.video_writer:
            self.recording = False
            self.stop_recording_flag = True
            
            # Keep a copy of the path before clearing
            filepath_to_convert = self.current_filepath 
            
            try:
                self.video_writer.release()
                self.logger.info(f"🛑 Stopped raw recording for event {self.current_event_id} at {filepath_to_convert}")
            except Exception as e:
                self.logger.error(f"Error releasing video writer: {e}")
            
            self.video_writer = None
            self.pre_buffer.clear()
            self.current_event_id = None
            self.current_filepath = None # Clear this
            
            # ==========================================================
            # ✅ STEP 2: Automatically convert the .avi file to .mp4
            # ==========================================================
            if filepath_to_convert:
                # Run conversion in a separate thread to not block the app
                convert_thread = threading.Thread(
                    target=self.convert_to_web_format, 
                    args=(filepath_to_convert,)
                )
                convert_thread.daemon = True
                convert_thread.start()

    def convert_to_web_format(self, avi_filepath):
        """
        Converts the raw .avi file to a web-friendly .mp4 using imageio-ffmpeg.
        """
        if not os.path.exists(avi_filepath):
            self.logger.error(f"Cannot convert: {avi_filepath} does not exist.")
            return None
            
        # Check if file has size, otherwise conversion will fail
        try:
            if os.path.getsize(avi_filepath) < 1024:
                 self.logger.warning(f"Skipping conversion: {avi_filepath} is empty (0 frames).")
                 try:
                     os.remove(avi_filepath) # Clean up empty file
                 except Exception as e:
                     self.logger.warning(f"Could not delete empty file: {e}")
                 return None
        except OSError as e:
            self.logger.error(f"Error checking file size for {avi_filepath}: {e}")
            return None

        # Create the new .mp4 filepath
        base_name = os.path.splitext(avi_filepath)[0]
        mp4_filepath = base_name + ".mp4"
        
        self.logger.info(f"🔄 Converting {avi_filepath} to {mp4_filepath}...")
        
        try:
            # Use imageio to read the AVI and write an MP4
            # 'libx264' is the universal H.264 codec for web.
            # 'yuv420p' pixel format is required for max browser compatibility
            iio.imwrite(
                iio.imread(avi_filepath, plugin="FFMPEG"), 
                mp4_filepath, 
                plugin="FFMPEG", 
                codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
            
            self.logger.info(f"✅ Successfully converted to {mp4_filepath}")
            
            # Clean up the old .avi file
            try:
                os.remove(avi_filepath)
                self.logger.info(f"🗑️ Cleaned up raw file: {avi_filepath}")
            except Exception as e:
                self.logger.warning(f"Could not delete raw file: {e}")
                
            return mp4_filepath
            
        except Exception as e:
            self.logger.error(f"❌ FFMPEG conversion failed: {e}")
            return None
            
    def get_current_recording_path(self):
        """Get path of current recording"""
        if self.current_filepath:
            # Return the path of the file *being written*
            return self.current_filepath
        return None
        
    def update_detection_time(self):
        """Update last detection time to keep recording active"""
        if self.recording:
            self.last_detection_time = time.time()
            
    def cleanup_old_recordings(self, days=3):
        """Remove recordings older than specified days (both .mp4 and .avi)"""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            deleted_count = 0
            
            for filename in os.listdir(self.recordings_folder):
                # Check for both formats
                if filename.endswith(('.mp4', '.avi')):
                    filepath = os.path.join(self.recordings_folder, filename)
                    file_time = os.path.getctime(filepath)
                    
                    if file_time < cutoff_time:
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                            self.logger.info(f"🗑️ Deleted old recording: {filename}")
                        except Exception as e:
                             self.logger.warning(f"Could not delete old file: {e}")
                            
            self.logger.info(f"🧹 Cleaned up {deleted_count} old recordings")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old recordings: {e}")
            return 0
            
    def get_recording_stats(self):
        """Get recording statistics (looks for .mp4 files)"""
        try:
            total_size = 0
            file_count = 0
            
            for filename in os.listdir(self.recordings_folder):
                # We only care about the final .mp4 files for stats
                if filename.endswith('.mp4'):
                    filepath = os.path.join(self.recordings_folder, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                        file_count += 1
                        
            return {
                'total_recordings': file_count,
                'total_size_bytes': total_size,
                'total_size_gb': round(total_size / (1024**3), 2),
                'total_size_mb': round(total_size / (1024**2), 2),
                'is_recording': self.recording,
                'current_event': self.current_event_id,
                'file_format': 'MP4 (from AVI)',
                'resolution': f"{self.resolution[0]}x{self.resolution[1]}",
                'fps': self.fps,
                'codec': 'H.264 (libx264)'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recording stats: {e}")
            return {}
            
    def verify_recording_playback(self, filepath):
        """Verify that a recording can be played back"""
        try:
            if not os.path.exists(filepath):
                # If we are checking an .mp4, the .avi might still be converting
                if filepath.endswith('.mp4'):
                    avi_path = filepath.replace('.mp4', '.avi')
                    if os.path.exists(avi_path):
                        return {'playable': False, 'error': 'Video is still converting...'}
                return {'playable': False, 'error': 'File not found'}
            
            # Try with OpenCV
            cap = cv2.VideoCapture(filepath)
            
            if not cap.isOpened():
                return {'playable': False, 'error': 'Cannot open file - codec issue'}
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {
                    'playable': False, 
                    'error': 'Cannot read frames - file corrupted',
                    'file_size': os.path.getsize(filepath),
                    'file_size_mb': round(os.path.getsize(filepath) / (1024**2), 2)
                }
            
            return {
                'playable': True,
                'fps': fps,
                'frame_count': frame_count,
                'duration_seconds': round(duration, 2),
                'file_size': os.path.getsize(filepath),
                'file_size_mb': round(os.path.getsize(filepath) / (1024**2), 2)
            }
            
        except Exception as e:
            return {'playable': False, 'error': f'Verification error: {str(e)}'}

# Global instance
recording_manager = RecordingManager()