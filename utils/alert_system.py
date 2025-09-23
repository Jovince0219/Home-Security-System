import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import datetime
import json
from utils.database import add_alert, save_alert_settings, get_alert_settings

class AlertSystem:
    def __init__(self):
        # Load settings from database
        self.load_settings()
        
        # Default alert messages
        self.alert_messages = {
            'face_detection': {
                'authorized': 'Authorized person detected: {name}',
                'unauthorized': 'SECURITY ALERT: Unauthorized person detected on your property',
                'face_covered': 'SECURITY ALERT: Person with covered face detected'
            },
            'motion_detection': {
                'human': 'SECURITY ALERT: Human motion detected on your property',
                'animal': 'Animal motion detected on your property',
                'object': 'Object motion detected on your property'
            },
            'emergency': 'EMERGENCY ALERT: Emergency protocol has been activated'
        }
    
    def load_settings(self):
        """Load alert settings from database with enhanced error handling and defaults"""
        try:
            settings = get_alert_settings()
            
            if settings:
                self.smtp_server = settings['smtp_server'] or "smtp.gmail.com"
                self.smtp_port = settings['smtp_port'] or 587
                self.email_user = settings['email_user'] or "your-email@gmail.com"
                self.email_password = settings['email_password'] or "your-app-password"
                self.owner_email = settings['owner_email'] or "owner@example.com"
                self.police_email = settings['police_email'] or "police@example.com"
                self.enable_email = bool(settings['enable_email']) if settings['enable_email'] is not None else True
                self.enable_police_alerts = bool(settings['enable_police_alerts']) if settings['enable_police_alerts'] is not None else False
            else:
                # Default settings
                self.smtp_server = "smtp.gmail.com"
                self.smtp_port = 587
                self.email_user = "your-email@gmail.com"
                self.email_password = "your-app-password"
                self.owner_email = "owner@example.com"
                self.police_email = "police@example.com"
                self.enable_email = True
                self.enable_police_alerts = False
                
                self._save_default_settings()
                
        except Exception as e:
            print(f"Error loading alert settings: {e}")
            # Use defaults if loading fails
            self.smtp_server = "smtp.gmail.com"
            self.smtp_port = 587
            self.email_user = "your-email@gmail.com"
            self.email_password = "your-app-password"
            self.owner_email = "owner@example.com"
            self.police_email = "police@example.com"
            self.enable_email = True
            self.enable_police_alerts = False
    
    def _save_default_settings(self):
        """Save default settings to database"""
        try:
            default_settings = {
                'owner_email': self.owner_email,
                'police_email': self.police_email,
                'smtp_server': self.smtp_server,
                'smtp_port': self.smtp_port,
                'email_user': self.email_user,
                'email_password': self.email_password,
                'enable_email': self.enable_email,
                'enable_police_alerts': self.enable_police_alerts
            }
            save_alert_settings(default_settings)
        except Exception as e:
            print(f"Error saving default settings: {e}")
    
    def send_alert(self, alert_level, detection_type, person_name=None, custom_message=None, screenshot_path=None):
        """Send alert based on alert level with enhanced unauthorized face detection"""
        try:
            # Generate appropriate message
            if custom_message:
                message = custom_message
            else:
                message = self.generate_alert_message(detection_type, person_name)
            
            result = {'level': alert_level, 'actions': []}
            
            if detection_type == 'face_detection' and (person_name == 'Unknown' or person_name is None):
                # Unauthorized face detected - send immediate alert
                if self.enable_email:
                    owner_result = self.notify_owner(detection_type, message, screenshot_path)
                    result['actions'].append(f"Owner notification: {owner_result}")
                else:
                    result['actions'].append("Owner notification: Email disabled")
                
                # Also trigger police alert for unauthorized faces
                if self.enable_police_alerts:
                    police_result = self.notify_police(detection_type, message, screenshot_path)
                    result['actions'].append(f"Police notification: {police_result}")
                else:
                    police_result = self.mock_police_notification(detection_type, message)
                    result['actions'].append(f"Mock police notification: {police_result}")
            
            elif alert_level == 1:
                # Level 1: Notify owner only
                if self.enable_email:
                    owner_result = self.notify_owner(detection_type, message, screenshot_path)
                    result['actions'].append(f"Owner notification: {owner_result}")
                else:
                    result['actions'].append("Owner notification: Email disabled")
                    
            elif alert_level == 2:
                # Level 2: Notify owner and police
                if self.enable_email:
                    owner_result = self.notify_owner(detection_type, message, screenshot_path)
                    result['actions'].append(f"Owner notification: {owner_result}")
                else:
                    result['actions'].append("Owner notification: Email disabled")
                
                if self.enable_police_alerts:
                    police_result = self.notify_police(detection_type, message, screenshot_path)
                    result['actions'].append(f"Police notification: {police_result}")
                else:
                    police_result = self.mock_police_notification(detection_type, message)
                    result['actions'].append(f"Mock police notification: {police_result}")
                    
            elif alert_level == 3:
                # Level 3: Full emergency protocol
                if self.enable_email:
                    owner_result = self.notify_owner(detection_type, message, screenshot_path)
                    result['actions'].append(f"Owner notification: {owner_result}")
                else:
                    result['actions'].append("Owner notification: Email disabled")
                
                if self.enable_police_alerts:
                    police_result = self.notify_police(detection_type, message, screenshot_path)
                    result['actions'].append(f"Police notification: {police_result}")
                else:
                    police_result = self.mock_police_notification(detection_type, message)
                    result['actions'].append(f"Mock police notification: {police_result}")
                
                emergency_result = self.trigger_emergency_protocol(message)
                result['actions'].append(f"Emergency protocol: {emergency_result}")
            
            # Log alert to database
            add_alert(f"Level {alert_level}", message, self.owner_email)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_alert_message(self, detection_type, person_name=None):
        """Generate appropriate alert message"""
        if detection_type == 'face_detection':
            if person_name and person_name != "Unknown":
                return self.alert_messages['face_detection']['authorized'].format(name=person_name)
            else:
                return self.alert_messages['face_detection']['unauthorized']
        elif detection_type == 'motion_detection':
            return self.alert_messages['motion_detection'].get(person_name, 'Motion detected')
        else:
            return f"Security alert: {detection_type}"
    
    def notify_owner(self, detection_type, message, screenshot_path=None):
        """Send email notification to owner"""
        try:
            subject = f"Security Alert: {detection_type.replace('_', ' ').title()}"
            return self.send_email(self.owner_email, subject, message, screenshot_path)
        except Exception as e:
            return f"Failed: {str(e)}"
    
    def notify_police(self, detection_type, message, screenshot_path=None):
        """Send notification to police (or mock if not configured)"""
        try:
            subject = f"Security Breach Alert: {detection_type.replace('_', ' ').title()}"
            emergency_message = f"URGENT: {message}\n\nImmediate attention required at security location."
            return self.send_email(self.police_email, subject, emergency_message, screenshot_path)
        except Exception as e:
            return f"Failed: {str(e)}"
    
    def mock_police_notification(self, detection_type, message):
        """Mock police notification for demonstration"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mock_message = f"[MOCK POLICE ALERT - {timestamp}] {detection_type}: {message}"
        
        # Log to console and file
        print(mock_message)
        
        try:
            with open('mock_police_alerts.log', 'a') as f:
                f.write(f"{mock_message}\n")
        except:
            pass
        
        # Log to database
        add_alert("Mock Police Alert", message, "police@mock.local")
        
        return "Mock alert logged successfully"
    
    def trigger_emergency_protocol(self, reason="Emergency protocol activated"):
        """Trigger emergency response (Level 3)"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            emergency_actions = [
                "🚨 EMERGENCY PROTOCOL ACTIVATED 🚨",
                f"Timestamp: {timestamp}",
                f"Reason: {reason}",
                "Actions taken:",
                "- All security cameras activated",
                "- Emergency lighting enabled", 
                "- Alarm system triggered",
                "- Authorities notified",
                "- Building lockdown initiated"
            ]
            
            emergency_log = "\n".join(emergency_actions)
            
            # Log to console
            print(emergency_log)
            
            # Log to file
            try:
                with open('emergency_protocol.log', 'a') as f:
                    f.write(f"\n{emergency_log}\n{'='*50}\n")
            except:
                pass
            
            # Log to database
            add_alert("Emergency Protocol", reason, "system")
            
            return "Emergency protocol activated successfully"
            
        except Exception as e:
            return f"Emergency protocol failed: {str(e)}"
    
    def send_email(self, recipient, subject, body, attachment_path=None):
        """Send email with optional image attachment and improved error handling"""
        try:
            if not self.enable_email:
                return "Email notifications disabled"
            
            if not self.email_user or self.email_user == "your-email@gmail.com":
                return "Email not configured - please update email settings"
            
            if not self.email_password or self.email_password == "your-app-password":
                return "Email password not configured - please update email settings"
            
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add timestamp to body
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            full_body = f"{body}\n\nTimestamp: {timestamp}\nSecurity System Alert"
            
            msg.attach(MIMEText(full_body, 'plain'))
            
            # Add screenshot if provided
            if attachment_path and os.path.exists(attachment_path):
                try:
                    with open(attachment_path, 'rb') as f:
                        img_data = f.read()
                        image = MIMEImage(img_data)
                        image.add_header('Content-Disposition', 'attachment', 
                                       filename=os.path.basename(attachment_path))
                        msg.attach(image)
                except Exception as e:
                    print(f"Failed to attach image: {e}")
            
            # Send email with improved error handling
            try:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.email_user, self.email_password)
                text = msg.as_string()
                server.sendmail(self.email_user, recipient, text)
                server.quit()
                
                return "Email sent successfully"
            except smtplib.SMTPAuthenticationError:
                return "Email authentication failed - check username/password"
            except smtplib.SMTPRecipientsRefused:
                return "Email recipient refused - check recipient address"
            except smtplib.SMTPServerDisconnected:
                return "SMTP server disconnected - check server settings"
            except Exception as e:
                return f"Email failed: {str(e)}"
            
        except Exception as e:
            return f"Email setup failed: {str(e)}"
    
    def test_email_config(self, test_recipient):
        """Test email configuration with detailed feedback"""
        try:
            if not self.enable_email:
                return "Email notifications are disabled - enable them first"
            
            if not self.email_user or self.email_user == "your-email@gmail.com":
                return "Email user not configured - please set your email address"
            
            if not self.email_password or self.email_password == "your-app-password":
                return "Email password not configured - please set your app password"
            
            if not test_recipient:
                return "Test recipient email address is required"
            
            subject = "Security System - Email Test"
            body = "This is a test email from your security system. If you receive this, email notifications are working correctly."
            
            result = self.send_email(test_recipient, subject, body)
            return result
            
        except Exception as e:
            return f"Test failed: {str(e)}"
    
    def update_settings(self, settings):
        """Update alert system settings with enhanced persistence and validation"""
        try:
            required_fields = ['owner_email', 'smtp_server', 'smtp_port', 'email_user']
            for field in required_fields:
                if not settings.get(field):
                    return f"Settings update failed: {field} is required"
            
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if not re.match(email_pattern, settings['owner_email']):
                return "Settings update failed: Invalid owner email format"
            
            if settings.get('police_email') and not re.match(email_pattern, settings['police_email']):
                return "Settings update failed: Invalid police email format"
            
            if not re.match(email_pattern, settings['email_user']):
                return "Settings update failed: Invalid email user format"
            
            try:
                smtp_port = int(settings['smtp_port'])
                if smtp_port < 1 or smtp_port > 65535:
                    return "Settings update failed: SMTP port must be between 1 and 65535"
                settings['smtp_port'] = smtp_port
            except ValueError:
                return "Settings update failed: SMTP port must be a valid number"
            
            # Save to database with explicit commit and verification
            save_alert_settings(settings)
            
            # Force reload settings from database to verify persistence
            self.load_settings()
            
            # Verify settings were actually saved by checking critical fields
            saved_settings = get_alert_settings()
            if not saved_settings:
                return "Settings update failed: Could not verify settings were saved"
            
            # Double-check critical settings
            if saved_settings['enable_email'] != settings['enable_email']:
                return "Settings update failed: Email enable setting not persisted"
            
            if saved_settings['owner_email'] != settings['owner_email']:
                return "Settings update failed: Owner email not persisted"
            
            return "Settings updated and verified successfully"
            
        except Exception as e:
            return f"Settings update failed: {str(e)}"
    
    def get_settings(self):
        """Get current alert settings (without sensitive data)"""
        return {
            'owner_email': self.owner_email,
            'police_email': self.police_email,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'email_user': self.email_user,
            'enable_email': self.enable_email,
            'enable_police_alerts': self.enable_police_alerts,
            'email_configured': (
                self.email_user != "your-email@gmail.com" and 
                self.email_password != "your-app-password" and
                self.owner_email != "owner@example.com"
            ),
            'police_configured': (
                self.police_email != "police@example.com" and
                self.enable_police_alerts
            )
        }
