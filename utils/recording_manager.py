import cv2
import os
import threading
import time
import datetime
from utils.database import add_recording
import logging
import queue

class RecordingManager:
    def __init__(self):
        self.is_recording = False
        self.current_recording_path = None
        self.video_writer = None
        self.recording_start_time = None
        self.recording_thread = None
        self.frame_processing_thread = None
        self.should_stop_recording = False
        self.recording_cooldowns = {}
        self.min_recording_duration = 3  # Minimum seconds to prevent spam
        self.recording_duration = 20  # Default recording duration
        
        # Real-time frame management
        self.frame_queue = queue.Queue(maxsize=30)  # Limit queue size to prevent memory issues
        self.target_fps = 15  # Normal FPS for security footage
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = None
        self.frame_count = 0
        
        # Create recordings directory if it doesn't exist
        os.makedirs('static/recordings', exist_ok=True)

    def start_recording_for_event(self, event_id, event_type):
        """Start recording for a specific event with improved cooldown management"""
        try:
            current_time = time.time()
            
            # Check cooldown for this specific event (prevent exact duplicates)
            last_event_recording = self.recording_cooldowns.get(f"event_{event_id}", 0)
            if current_time - last_event_recording < 30:  # 30 second cooldown per event
                print(f"⏳ Recording cooldown active for event {event_id}, skipping...")
                return None
            
            # Check cooldown for event type (prevent too many of same type)
            last_type_recording = self.recording_cooldowns.get(f"type_{event_type}", 0)
            if current_time - last_type_recording < 10:  # 10 second cooldown per type
                print(f"⏳ Recording cooldown active for {event_type}, skipping...")
                return None
            
            # Stop any existing recording
            if self.is_recording:
                self.stop_recording()
                time.sleep(1.0)  # Brief pause between recordings
            
            # Generate filename using the EXACT event_id to ensure consistency
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Use consistent naming pattern for all event types
            if event_type == "authorized_face":
                clean_type = "Authorized_Face"
            elif event_type == "known_face":
                clean_type = "Known_Face" 
            elif event_type == "unauthorized_face":
                clean_type = "Unauthorized_Face"
            else:
                clean_type = event_type.replace(' ', '_').title()
            
            # CRITICAL FIX: Use the event_id for filename to match database
            filename = f"recording_{clean_type}_{event_id.replace('-', '_')}.mp4"
            filepath = os.path.join('static', 'recordings', filename)
            
            # Try different codecs
            fourcc = None
            codecs_to_try = [
                ('avc1', '.mp4'),  # H.264
                ('mp4v', '.mp4'),  # MPEG-4
                ('XVID', '.avi'),  # XVID
                ('MJPG', '.avi')   # Motion JPEG
            ]
            
            self.video_writer = None
            for codec, extension in codecs_to_try:
                test_path = filepath.replace('.mp4', extension) if extension != '.mp4' else filepath
                fourcc_code = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(test_path, fourcc_code, self.target_fps, (640, 480))
                
                if test_writer.isOpened():
                    self.video_writer = test_writer
                    filepath = test_path
                    fourcc = codec
                    print(f"✅ Using codec: {codec}")
                    print(f"📁 Recording filepath: {filepath}")
                    break
                else:
                    test_writer.release()
                    print(f"❌ Codec {codec} failed")
            
            if self.video_writer is None:
                print(f"❌ All codecs failed for recording")
                return None
            
            # Start recording
            self.is_recording = True
            self.current_recording_path = filepath
            self.recording_start_time = current_time
            self.should_stop_recording = False
            self.last_frame_time = current_time
            self.frame_count = 0
            
            # Update cooldowns
            self.recording_cooldowns[f"event_{event_id}"] = current_time
            self.recording_cooldowns[f"type_{event_type}"] = current_time
            
            # Clear the frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start frame processing thread
            self.frame_processing_thread = threading.Thread(target=self._process_frames)
            self.frame_processing_thread.daemon = True
            self.frame_processing_thread.start()
            
            print(f"🎥 Started recording for {event_type}: {filename}")
            print(f"🎯 Event ID: {event_id}")
            
            # Start background thread to stop recording after duration
            self.recording_thread = threading.Thread(
                target=self._recording_timer,
                args=(self.recording_duration,)
            )
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Return the web-accessible path (consistent format)
            web_path = f"static/recordings/{os.path.basename(filepath)}"
            print(f"📁 Final web path: {web_path}")
            return web_path
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            import traceback
            traceback.print_exc()
            return None

    def add_frame(self, frame):
        """Add a frame to the recording queue with timestamp"""
        try:
            if self.is_recording and frame is not None:
                # Resize frame if necessary
                if frame.shape[1] != 640 or frame.shape[0] != 480:
                    frame = cv2.resize(frame, (640, 480))
                
                # Add timestamp to frame
                current_time = time.time()
                elapsed = current_time - self.recording_start_time
                
                # Create frame copy to avoid modifying original
                frame_copy = frame.copy()
                
                # Add detailed timestamp
                timestamp_text = f"Time: {elapsed:.1f}s | Frame: {self.frame_count}"
                cv2.putText(frame_copy, timestamp_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add actual datetime
                datetime_text = datetime.datetime.now().strftime('%H:%M:%S')
                cv2.putText(frame_copy, datetime_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Try to add frame to queue (non-blocking)
                try:
                    self.frame_queue.put((frame_copy, current_time), block=False)
                    self.frame_count += 1
                    return True
                except queue.Full:
                    # If queue is full, drop the frame to maintain real-time
                    print("⚠️ Frame queue full, dropping frame")
                    return False
                    
            return False
        except Exception as e:
            print(f"❌ Error adding frame to queue: {e}")
            return False

    def _process_frames(self):
        """Process frames from queue with real-time timing"""
        print("🔄 Starting frame processing thread")
        
        frames_processed = 0
        start_time = time.time()
        
        while self.is_recording or not self.frame_queue.empty():
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, frame_time = frame_data
                
                # Calculate when this frame should be written based on real time
                expected_frame_time = self.recording_start_time + (frames_processed * self.frame_interval)
                current_time = time.time()
                
                # If we're behind schedule, write immediately
                # If we're ahead of schedule, wait to maintain proper timing
                if current_time < expected_frame_time:
                    time_to_wait = expected_frame_time - current_time
                    time.sleep(time_to_wait)
                
                # Write frame to video
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    frames_processed += 1
                
                # Log progress every 30 frames
                if frames_processed % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frames_processed / elapsed if elapsed > 0 else 0
                    print(f"📊 Processed {frames_processed} frames, Actual FPS: {actual_fps:.1f}")
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frames in queue, continue checking if we should stop
                continue
            except Exception as e:
                print(f"❌ Error in frame processing: {e}")
                break
        
        print(f"🛑 Frame processing stopped. Processed {frames_processed} frames total")

    def _recording_timer(self, duration):
        """Background thread to stop recording after specified duration"""
        try:
            print(f"⏰ Recording timer started: {duration} seconds")
            start_time = time.time()
            
            while time.time() - start_time < duration and not self.should_stop_recording:
                time.sleep(0.5)  # Check every 500ms
            
            if self.is_recording:
                actual_duration = time.time() - self.recording_start_time
                print(f"🛑 Recording timer expired after {actual_duration:.1f} seconds")
                self.stop_recording()
                
        except Exception as e:
            print(f"❌ Error in recording timer: {e}")

    def stop_recording(self):
        """Stop the current recording and save properly"""
        try:
            if self.is_recording:
                print("🛑 Beginning recording stop process...")
                self.should_stop_recording = True
                self.is_recording = False
                
                # Wait for frame queue to empty (with timeout)
                print("⏳ Waiting for frame queue to empty...")
                queue_empty = self.frame_queue.empty()
                wait_start = time.time()
                while not queue_empty and (time.time() - wait_start) < 5.0:  # 5 second timeout
                    time.sleep(0.1)
                    queue_empty = self.frame_queue.empty()
                
                if not queue_empty:
                    print("⚠️ Frame queue not empty after timeout, proceeding...")
                
                # Calculate actual recording duration
                current_time = time.time()
                recording_length = current_time - self.recording_start_time
                
                print(f"📊 Recording stats: {recording_length:.1f}s, {self.frame_count} frames")
                
                # Properly close the video writer
                if self.video_writer is not None:
                    self.video_writer.release()
                    print("💾 Video writer released")
                
                # Wait for processing thread to finish
                if self.frame_processing_thread and self.frame_processing_thread.is_alive():
                    self.frame_processing_thread.join(timeout=3.0)
                    print("✅ Frame processing thread stopped")
                
                # Validate the recording file
                if self.current_recording_path and os.path.exists(self.current_recording_path):
                    file_size = os.path.getsize(self.current_recording_path)
                    
                    # Calculate actual FPS
                    actual_fps = self.frame_count / recording_length if recording_length > 0 else 0
                    print(f"📊 Final stats: {recording_length:.1f}s, {self.frame_count} frames, {actual_fps:.1f} FPS, {file_size} bytes")
                    
                    # Only save to database if recording meets minimum duration
                    if recording_length >= self.min_recording_duration:
                        # Use Python datetime for accurate timestamps
                        from datetime import datetime
                        start_dt = datetime.fromtimestamp(self.recording_start_time)
                        end_dt = datetime.fromtimestamp(current_time)
                        
                        add_recording(
                            self.current_recording_path,
                            start_dt,
                            end_dt,
                            recording_length,
                            file_size
                        )
                        print(f"💾 Saved recording: {os.path.basename(self.current_recording_path)} - Start: {start_dt}, End: {end_dt}")
                    else:
                        print(f"🗑️ Discarding short recording ({recording_length:.1f}s)")
                        os.remove(self.current_recording_path)
                else:
                    print(f"❌ Recording file not found: {self.current_recording_path}")
                
                # Clean up
                self.current_recording_path = None
                self.video_writer = None
                self.recording_start_time = None
                self.last_frame_time = None
                self.frame_count = 0
                
                # Clear queue
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if self.recording_thread and self.recording_thread.is_alive():
                    self.recording_thread.join(timeout=2.0)
                
                print("✅ Recording stopped completely")
                return True
            return False
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
            # Force cleanup on error
            self.is_recording = False
            self.current_recording_path = None
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            return False

    def update_detection_time(self):
        """Update detection time to extend recording if needed"""
        pass

    def get_recording_stats(self):
        """Get recording statistics"""
        try:
            recordings_dir = 'static/recordings'
            total_size = 0
            file_count = 0
            
            if os.path.exists(recordings_dir):
                for filename in os.listdir(recordings_dir):
                    if filename.endswith(('.mp4', '.avi')):
                        filepath = os.path.join(recordings_dir, filename)
                        total_size += os.path.getsize(filepath)
                        file_count += 1
            
            return {
                'is_recording': self.is_recording,
                'total_recordings': file_count,
                'total_size_bytes': total_size,
                'total_size_gb': round(total_size / (1024**3), 2),
                'cooldown_status': self.recording_cooldowns,
                'target_fps': self.target_fps,
                'current_frame_count': self.frame_count,
                'queue_size': self.frame_queue.qsize()
            }
        except Exception as e:
            print(f"Error getting recording stats: {e}")
            return {}

    def cleanup_old_recordings(self, days=3):
        """Clean up recordings older than specified days"""
        try:
            recordings_dir = 'static/recordings'
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            deleted_count = 0
            
            if os.path.exists(recordings_dir):
                for filename in os.listdir(recordings_dir):
                    if filename.endswith(('.mp4', '.avi')):
                        filepath = os.path.join(recordings_dir, filename)
                        if os.path.getctime(filepath) < cutoff_time:
                            os.remove(filepath)
                            deleted_count += 1
                            print(f"🧹 Deleted old recording: {filename}")
            
            return deleted_count
        except Exception as e:
            print(f"Error cleaning up old recordings: {e}")
            return 0

    def verify_recording_playback(self, filepath):
        """Verify if a recording file is playable and check timing"""
        try:
            if not os.path.exists(filepath):
                return {'playable': False, 'error': 'File not found'}
            
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return {'playable': False, 'error': 'Cannot open file'}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            
            # Read first few frames to check timing
            frames_checked = 0
            timestamps_found = []
            
            for i in range(min(10, int(frame_count))):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for text detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_checked += 1
            
            cap.release()
            
            file_size = os.path.getsize(filepath)
            
            result = {
                'playable': True,
                'file_size_mb': round(file_size / (1024*1024), 2),
                'duration_seconds': round(duration, 2),
                'frame_count': int(frame_count),
                'fps': round(fps, 1),
                'frames_checked': frames_checked,
                'expected_duration': round(frame_count / self.target_fps, 2) if self.target_fps > 0 else 0
            }
            
            # Check if duration makes sense
            expected_duration = frame_count / self.target_fps
            actual_duration = duration
            duration_diff = abs(expected_duration - actual_duration)
            
            if duration_diff > 2.0:  # More than 2 seconds difference
                result['timing_warning'] = f"Duration mismatch: expected {expected_duration:.1f}s, got {actual_duration:.1f}s"
                result['speed_issue'] = True
            else:
                result['timing_warning'] = "Timing appears normal"
                result['speed_issue'] = False
            
            return result
                
        except Exception as e:
            return {'playable': False, 'error': str(e)}

# Global instance
recording_manager = RecordingManager()