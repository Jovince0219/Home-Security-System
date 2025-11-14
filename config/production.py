import os
from datetime import timedelta

class ProductionConfig:
    # Basic Flask Config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-production-secret-key-change-this')
    DEBUG = False
    TESTING = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/dbname')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    
    # AWS S3
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.environ.get('S3_BUCKET', 'your-security-system-recordings')
    S3_RECORDINGS_PREFIX = 'recordings/'
    S3_FACES_PREFIX = 'faces/'
    S3_SCREENSHOTS_PREFIX = 'screenshots/'
    
    # File Upload
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    UPLOAD_EXTENSIONS = ['.jpg', '.png', '.jpeg', '.mp4', '.avi']
    
    # Security
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    
    # Application Specific
    MOTION_DETECTION_SENSITIVITY = float(os.environ.get('MOTION_SENSITIVITY', '2.0'))
    FACE_RECOGNITION_CONFIDENCE = float(os.environ.get('FACE_CONFIDENCE', '0.6'))
    MAX_RECORDING_DURATION = 600  # 10 minutes
    
    # Twilio
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')