#!/usr/bin/env python3
"""
Security System Setup Script
This script initializes the complete security system with all required dependencies.
"""

import os
import sys
import subprocess
import sqlite3

def install_requirements():
    """Install all required Python packages"""
    requirements = [
        'flask',
        'opencv-python',
        'face-recognition',
        'numpy',
        'pillow',
        'scikit-learn',
        'pyaudio',
        'pyttsx3',
        'speechrecognition',
        # 'smtplib-ssl'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'static/uploads',
        'static/faces',
        'static/screenshots',
        'static/recordings',
        'static/audio',
        'data',
        'training_data/human',
        'training_data/animal',
        'training_data/object'
    ]
    
    print("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}")
        
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def initialize_database():
    """Initialize the security system database"""
    print("Initializing database...")
    from utils.database import init_db
    init_db()
    print("✓ Database initialized")

def create_sample_data():
    """Create sample configuration files"""
    
    # Sample camera presets
    sample_presets = {
        "Front Door": {
            "position": {"pan": 0, "tilt": 0},
            "created_at": "2024-01-01T00:00:00",
            "settings": {"brightness": 50, "contrast": 50}
        },
        "Parking Area": {
            "position": {"pan": 45, "tilt": -15},
            "created_at": "2024-01-01T00:00:00",
            "settings": {"brightness": 60, "contrast": 55}
        }
    }
    
    # Sample camera settings
    sample_settings = {
        "brightness": 50,
        "contrast": 50,
        "saturation": 50,
        "zoom": 100,
        "night_vision": False,
        "auto_tracking": False
    }
    
    import json
    
    # Save sample presets
    with open('data/camera_presets.json', 'w') as f:
        json.dump(sample_presets, f, indent=2)
    
    # Save sample settings
    with open('data/camera_settings.json', 'w') as f:
        json.dump(sample_settings, f, indent=2)
    
    print("✓ Created sample configuration files")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Security System Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return False
    
    # Create directories
    create_directories()
    
    # Initialize database
    initialize_database()
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nTo start the security system:")
    print("python app.py")
    print("\nThen open your browser to: http://localhost:5000")
    print("\nFeatures available:")
    print("- Face Recognition & Registration")
    print("- Motion Detection with AI Classification")
    print("- 3-Level Alert System with Email Notifications")
    print("- CCTV Controls with Pan/Tilt/Zoom")
    print("- Audio Recording & Two-Way Communication")
    print("- Screenshot & Video Recording")
    print("- Comprehensive Management Dashboard")
    
    return True

if __name__ == "__main__":
    main()
