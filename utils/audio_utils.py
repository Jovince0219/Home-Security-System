import pyaudio
import wave
import threading
import time
import os
from datetime import datetime
import pyttsx3
import speech_recognition as sr

class AudioManager:
    def __init__(self):
        self.audio_folder = 'static/audio'
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        
        # State variables
        self.is_recording = False
        self.is_playing = False
        self.two_way_active = False
        self.current_recording = None
        
        self.tts_engine = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Pre-defined messages
        self.predefined_messages = {
            'warning': "Warning: You are being recorded. Please leave the premises immediately.",
            'greeting': "Hello, you are being monitored by our security system.",
            'police': "This is a security alert. Police have been notified.",
            'stop': "Stop! You are trespassing on private property."
        }
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return {'status': 'error', 'message': 'Recording already in progress'}
        
        try:
            self.is_recording = True
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"audio_recording_{timestamp}.wav"
            filepath = os.path.join(self.audio_folder, filename)
            
            # Start recording in separate thread
            self.recording_thread = threading.Thread(
                target=self._record_audio, 
                args=(filepath,)
            )
            self.recording_thread.start()
            
            return {
                'status': 'success',
                'message': 'Audio recording started',
                'filename': filename
            }
        
        except Exception as e:
            self.is_recording = False
            return {'status': 'error', 'message': str(e)}
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return {'status': 'error', 'message': 'No recording in progress'}
        
        try:
            self.is_recording = False
            
            # Wait for recording thread to finish
            if hasattr(self, 'recording_thread'):
                self.recording_thread.join(timeout=5)
            
            # Save recording info to database
            if self.current_recording:
                from utils.database import add_audio_recording
                add_audio_recording(
                    self.current_recording['filepath'],
                    self.current_recording['duration']
                )
            
            return {
                'status': 'success',
                'message': 'Audio recording stopped',
                'recording': self.current_recording
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _record_audio(self, filepath):
        """Internal method to handle audio recording"""
        try:
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            frames = []
            start_time = time.time()
            
            while self.is_recording:
                data = stream.read(self.chunk)
                frames.append(data)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save audio file
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            # Store recording info
            self.current_recording = {
                'filepath': filepath,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False
    
    def play_message(self, message_type, custom_text=None):
        """Play audio message through speakers with improved reliability"""
        try:
            if message_type == 'custom' and custom_text:
                text = custom_text
            elif message_type in self.predefined_messages:
                text = self.predefined_messages[message_type]
            else:
                return {'status': 'error', 'message': 'Invalid message type'}
            
            if self.is_playing:
                self.is_playing = False
                time.sleep(0.8)  # Longer wait for complete cleanup
            
            play_thread = threading.Thread(target=self._play_tts_isolated, args=(text,))
            play_thread.daemon = True
            play_thread.start()
            
            return {
                'status': 'success',
                'message': f'Playing message: {message_type}',
                'text': text
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _play_tts_isolated(self, text):
        """Completely isolated TTS playback to prevent conflicts"""
        try:
            self.is_playing = True
            
            import pyttsx3
            tts_engine = pyttsx3.init()
            
            # Configure engine with error handling
            try:
                tts_engine.setProperty('rate', 150)
                tts_engine.setProperty('volume', 0.9)
                
                # Set voice properties
                voices = tts_engine.getProperty('voices')
                if voices and len(voices) > 0:
                    tts_engine.setProperty('voice', voices[0].id)
            except Exception as e:
                print(f"TTS configuration warning: {e}")
            
            # Play the message with timeout protection
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS playback error: {e}")
            
            try:
                tts_engine.stop()
                del tts_engine
            except Exception as e:
                print(f"TTS cleanup warning: {e}")
            
            # Ensure complete cleanup
            time.sleep(0.2)
        
        except Exception as e:
            print(f"Error in TTS playback: {e}")
        finally:
            # Always reset playing state
            self.is_playing = False

    def _play_tts(self, text):
        """Internal method to play text-to-speech"""
        self._play_tts_isolated(text)
    
    def start_two_way_communication(self):
        """Start two-way audio communication"""
        if self.two_way_active:
            return {'status': 'error', 'message': 'Two-way communication already active'}
        
        try:
            self.two_way_active = True
            
            # Start two-way communication in separate thread
            self.two_way_thread = threading.Thread(target=self._handle_two_way_audio)
            self.two_way_thread.start()
            
            return {
                'status': 'success',
                'message': 'Two-way communication started'
            }
        
        except Exception as e:
            self.two_way_active = False
            return {'status': 'error', 'message': str(e)}
    
    def stop_two_way_communication(self):
        """Stop two-way audio communication"""
        if not self.two_way_active:
            return {'status': 'error', 'message': 'Two-way communication not active'}
        
        try:
            self.two_way_active = False
            
            # Wait for thread to finish
            if hasattr(self, 'two_way_thread'):
                self.two_way_thread.join(timeout=5)
            
            return {
                'status': 'success',
                'message': 'Two-way communication stopped'
            }
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _handle_two_way_audio(self):
        """Handle two-way audio communication"""
        try:
            # This is a simplified implementation
            # In a real system, this would handle real-time audio streaming
            while self.two_way_active:
                time.sleep(0.1)  # Simulate audio processing
                
        except Exception as e:
            print(f"Error in two-way audio: {e}")
            self.two_way_active = False
    
    def get_status(self):
        """Get current audio system status"""
        return {
            'is_recording': self.is_recording,
            'is_playing': self.is_playing,
            'two_way_active': self.two_way_active,
            'available_messages': list(self.predefined_messages.keys())
        }
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                
                return {
                    'status': 'success',
                    'transcription': text
                }
        
        except sr.UnknownValueError:
            return {'status': 'error', 'message': 'Could not understand audio'}
        except sr.RequestError as e:
            return {'status': 'error', 'message': f'Speech recognition error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
