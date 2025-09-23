import cv2
import os
import threading
import datetime
import time
from utils.database import add_recording, add_screenshot

class RecordingManager:
    def __init__(self):
        self.is_recording = False
        self.recording_thread = None
        self.video_writer = None
        self.current_recording_path = None
        self.recording_start_time = None
        
        # Ensure directories exist
        os.makedirs('static/recordings', exist_ok=True)
        os.makedirs('static/screenshots', exist_ok=True)
        
        # Recording settings
        self.fps = 20
        self.frame_width = 640
        self.frame_height = 480
        self.codec = cv2.VideoWriter_fourcc(*'XVID')
    
    def start_recording(self):
        """Start video recording"""
        try:
            if self.is_recording:
                return {'error': 'Recording already in progress'}
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"recording_{timestamp}.avi"
            self.current_recording_path = os.path.join('static/recordings', filename)
            
            # Initialize video writer
            self.video_writer = cv2.VideoWriter(
                self.current_recording_path,
                self.codec,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            
            if not self.video_writer.isOpened():
                return {'error': 'Failed to initialize video writer'}
            
            # Start recording thread
            self.is_recording = True
            self.recording_start_time = datetime.datetime.now()
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return {
                'message': 'Recording started successfully',
                'filename': filename,
                'start_time': self.recording_start_time.isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to start recording: {str(e)}'}
    
    def stop_recording(self):
        """Stop video recording with comprehensive cleanup and database integration"""
        try:
            if not self.is_recording:
                return {'error': 'No recording in progress'}
        
            self.is_recording = False
        
            # Wait for recording thread with extended timeout
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=20)  # Extended timeout for safety
        
                # Force cleanup if thread doesn't finish
                if self.recording_thread.is_alive():
                    print("Warning: Recording thread did not finish cleanly")
        
            # Properly release video writer
            if self.video_writer:
                try:
                    self.video_writer.release()
                    time.sleep(0.5)  # Allow time for file system to sync
                except Exception as e:
                    print(f"Error releasing video writer: {e}")
                finally:
                    self.video_writer = None
        
            end_time = datetime.datetime.now()
            duration = int((end_time - self.recording_start_time).total_seconds()) if self.recording_start_time else 0
        
            file_size = 0
            file_exists = False
            database_saved = False
        
            # Check if file exists and has content
            if self.current_recording_path:
                # Wait a moment for file system to sync
                time.sleep(1.0)
        
                if os.path.exists(self.current_recording_path):
                    file_size = os.path.getsize(self.current_recording_path)
                    file_exists = True
        
                    # Only save to database if file has meaningful content (>1KB)
                    if file_size > 1024:
                        try:
                            add_recording(
                                self.current_recording_path,
                                self.recording_start_time,
                                end_time,
                                duration,
                                file_size
                            )
                            database_saved = True
                            print(f"Recording saved to database: {self.current_recording_path} ({file_size} bytes)")
                        except Exception as e:
                            print(f"Error saving recording to database: {e}")
                    else:
                        print(f"Recording file too small ({file_size} bytes), not saving to database")
        
            result = {
                'message': 'Recording stopped successfully' if database_saved else 'Recording stopped but file may be empty or corrupted',
                'filename': os.path.basename(self.current_recording_path) if self.current_recording_path else 'unknown',
                'duration': duration,
                'file_size': file_size,
                'file_path': self.current_recording_path if file_exists else None,
                'saved_to_database': database_saved,
                'file_exists': file_exists
            }
        
            # Reset state
            self.current_recording_path = None
            self.recording_start_time = None
        
            return result
        
        except Exception as e:
            # Ensure cleanup even on error
            self.is_recording = False
            if self.video_writer:
                try:
                    self.video_writer.release()
                except:
                    pass
                finally:
                    self.video_writer = None
        
            self.current_recording_path = None
            self.recording_start_time = None
        
            return {'error': f'Failed to stop recording: {str(e)}'}
    
    def _recording_loop(self):
        """Main recording loop with improved error handling"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera for recording")
            self.is_recording = False
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        frame_count = 0
        
        try:
            while self.is_recording:
                ret, frame = cap.read()
                
                if ret and self.video_writer and self.video_writer.isOpened():
                    # Resize frame if necessary
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    
                    # Add timestamp overlay
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Write frame
                    try:
                        self.video_writer.write(frame)
                        frame_count += 1
                    except Exception as e:
                        print(f"Error writing frame: {e}")
                        break
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
        
        except Exception as e:
            print(f"Error in recording loop: {e}")
        finally:
            cap.release()
            print(f"Recording finished. Total frames: {frame_count}")
    
    def capture_screenshot(self, description="Manual screenshot"):
        """Capture a screenshot from the camera with database integration"""
        try:
            cap = cv2.VideoCapture(0)
        
            if not cap.isOpened():
                return {'error': 'Could not access camera'}
        
            ret, frame = cap.read()
            cap.release()
        
            if not ret:
                return {'error': 'Failed to capture frame'}
        
            # Generate filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"screenshot_{timestamp}.jpg"
            filepath = os.path.join('static/screenshots', filename)
        
            # Add timestamp overlay
            timestamp_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
            # Save screenshot
            cv2.imwrite(filepath, frame)
        
            # Get file size
            file_size = os.path.getsize(filepath)
        
            try:
                add_screenshot(filepath, description, file_size, "manual_capture")
                database_saved = True
            except Exception as e:
                print(f"Error saving screenshot to database: {e}")
                database_saved = False
        
            return {
                'message': 'Screenshot captured successfully',
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'saved_to_database': database_saved
            }
        
        except Exception as e:
            return {'error': f'Failed to capture screenshot: {str(e)}'}
    
    def get_status(self):
        """Get current recording status"""
        status = {
            'is_recording': self.is_recording,
            'current_file': os.path.basename(self.current_recording_path) if self.current_recording_path else None,
            'start_time': self.recording_start_time.isoformat() if self.recording_start_time else None,
            'duration': 0
        }
        
        if self.is_recording and self.recording_start_time:
            duration = (datetime.datetime.now() - self.recording_start_time).total_seconds()
            status['duration'] = int(duration)
        
        return status
    
    def get_storage_stats(self):
        """Get storage statistics"""
        try:
            recordings_dir = 'static/recordings'
            screenshots_dir = 'static/screenshots'
            
            # Calculate recordings storage
            recordings_size = 0
            recordings_count = 0
            if os.path.exists(recordings_dir):
                for filename in os.listdir(recordings_dir):
                    filepath = os.path.join(recordings_dir, filename)
                    if os.path.isfile(filepath):
                        recordings_size += os.path.getsize(filepath)
                        recordings_count += 1
            
            # Calculate screenshots storage
            screenshots_size = 0
            screenshots_count = 0
            if os.path.exists(screenshots_dir):
                for filename in os.listdir(screenshots_dir):
                    filepath = os.path.join(screenshots_dir, filename)
                    if os.path.isfile(filepath):
                        screenshots_size += os.path.getsize(filepath)
                        screenshots_count += 1
            
            return {
                'recordings': {
                    'count': recordings_count,
                    'size_bytes': recordings_size,
                    'size_mb': round(recordings_size / (1024 * 1024), 2)
                },
                'screenshots': {
                    'count': screenshots_count,
                    'size_bytes': screenshots_size,
                    'size_mb': round(screenshots_size / (1024 * 1024), 2)
                },
                'total': {
                    'size_bytes': recordings_size + screenshots_size,
                    'size_mb': round((recordings_size + screenshots_size) / (1024 * 1024), 2)
                }
            }
            
        except Exception as e:
            return {'error': f'Failed to get storage stats: {str(e)}'}
    
    def cleanup_old_files(self, days=30):
        """Clean up files older than specified days"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            deleted_files = []
            
            # Clean up recordings
            recordings_dir = 'static/recordings'
            if os.path.exists(recordings_dir):
                for filename in os.listdir(recordings_dir):
                    filepath = os.path.join(recordings_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            deleted_files.append(filepath)
            
            # Clean up screenshots
            screenshots_dir = 'static/screenshots'
            if os.path.exists(screenshots_dir):
                for filename in os.listdir(screenshots_dir):
                    filepath = os.path.join(screenshots_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            deleted_files.append(filepath)
            
            # Clean up database entries
            from utils.database import cleanup_old_recordings, cleanup_old_screenshots
            cleanup_old_recordings(days)
            cleanup_old_screenshots(days)
            
            return {
                'message': f'Cleanup completed. Deleted {len(deleted_files)} files older than {days} days.',
                'deleted_files': deleted_files
            }
            
        except Exception as e:
            return {'error': f'Cleanup failed: {str(e)}'}

    def create_video_clip(self, recording_id, start_time, end_time, clip_name):
        """Create a video clip from a recording"""
        try:
            from utils.database import get_db_connection
            
            # Get recording info
            conn = get_db_connection()
            recording = conn.execute('SELECT * FROM recordings WHERE id = ?', (recording_id,)).fetchone()
            conn.close()
            
            if not recording or not os.path.exists(recording['file_path']):
                return {'error': 'Recording not found'}
            
            # Generate clip filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            clip_filename = f"clip_{clip_name}_{timestamp}.avi"
            clip_path = os.path.join('static/recordings', clip_filename)
            
            # Use OpenCV to create clip
            cap = cv2.VideoCapture(recording['file_path'])
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(clip_path, fourcc, fps, (frame_width, frame_height))
            
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                current_frame += 1
            
            cap.release()
            out.release()
            
            # Get clip file size
            clip_size = os.path.getsize(clip_path)
            clip_duration = end_time - start_time
            
            # Save clip info to database
            from utils.database import add_recording
            add_recording(
                clip_path,
                datetime.datetime.now(),
                datetime.datetime.now() + datetime.timedelta(seconds=clip_duration),
                clip_duration,
                clip_size
            )
            
            return {
                'message': 'Video clip created successfully',
                'clip_path': clip_path,
                'clip_name': clip_filename,
                'duration': clip_duration,
                'file_size': clip_size
            }
            
        except Exception as e:
            return {'error': f'Failed to create clip: {str(e)}'}

    def export_media(self, media_ids, media_type, export_format):
        """Export selected recordings or screenshots"""
        try:
            import zipfile
            import shutil
            
            # Create export directory
            export_dir = 'static/exports'
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if export_format == 'zip':
                # Create ZIP file
                zip_filename = f"security_export_{media_type}_{timestamp}.zip"
                zip_path = os.path.join(export_dir, zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    from utils.database import get_db_connection
                    conn = get_db_connection()
                    
                    for media_id in media_ids:
                        if media_type == 'recordings':
                            media = conn.execute('SELECT * FROM recordings WHERE id = ?', (media_id,)).fetchone()
                        else:
                            media = conn.execute('SELECT * FROM screenshots WHERE id = ?', (media_id,)).fetchone()
                        
                        if media and os.path.exists(media['file_path']):
                            # Add file to ZIP with descriptive name
                            filename = os.path.basename(media['file_path'])
                            zipf.write(media['file_path'], filename)
                    
                    conn.close()
                
                return {
                    'message': f'Exported {len(media_ids)} {media_type} to ZIP file',
                    'export_path': zip_path,
                    'export_filename': zip_filename,
                    'download_url': f'/static/exports/{zip_filename}'
                }
            
            else:
                # Create folder export
                folder_name = f"security_export_{media_type}_{timestamp}"
                folder_path = os.path.join(export_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                
                from utils.database import get_db_connection
                conn = get_db_connection()
                
                exported_count = 0
                for media_id in media_ids:
                    if media_type == 'recordings':
                        media = conn.execute('SELECT * FROM recordings WHERE id = ?', (media_id,)).fetchone()
                    else:
                        media = conn.execute('SELECT * FROM screenshots WHERE id = ?', (media_id,)).fetchone()
                    
                    if media and os.path.exists(media['file_path']):
                        # Copy file to export folder
                        filename = os.path.basename(media['file_path'])
                        dest_path = os.path.join(folder_path, filename)
                        shutil.copy2(media['file_path'], dest_path)
                        exported_count += 1
                
                conn.close()
                
                return {
                    'message': f'Exported {exported_count} {media_type} to folder',
                    'export_path': folder_path,
                    'export_folder': folder_name,
                    'exported_count': exported_count
                }
            
        except Exception as e:
            return {'error': f'Export failed: {str(e)}'}
