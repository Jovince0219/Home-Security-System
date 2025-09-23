#!/usr/bin/env python3
"""
Security System Test Script
This script tests all major components of the security system.
"""

import os
import sys
import time
import requests
import json

def test_flask_server():
    """Test if Flask server is running"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        return response.status_code == 200
    except:
        return False

def test_database_connection():
    """Test database connectivity"""
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        conn.execute('SELECT 1')
        conn.close()
        return True
    except:
        return False

def test_face_recognition():
    """Test face recognition system"""
    try:
        from utils.face_recognition_utils import load_known_faces
        load_known_faces()
        return True
    except:
        return False

def test_motion_detection():
    """Test motion detection system"""
    try:
        from utils.motion_detection_utils import MotionDetector
        detector = MotionDetector()
        return True
    except:
        return False

def test_alert_system():
    """Test alert system"""
    try:
        from utils.alert_system import AlertSystem
        alert_system = AlertSystem()
        return True
    except:
        return False

def test_cctv_controls():
    """Test CCTV control system"""
    try:
        from utils.cctv_utils import CCTVController
        controller = CCTVController()
        return True
    except:
        return False

def test_audio_system():
    """Test audio system"""
    try:
        from utils.audio_utils import AudioManager
        audio_manager = AudioManager()
        return True
    except:
        return False

def test_recording_system():
    """Test recording system"""
    try:
        from utils.recording_utils import RecordingManager
        recording_manager = RecordingManager()
        return True
    except:
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Security System Component Tests")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Face Recognition System", test_face_recognition),
        ("Motion Detection System", test_motion_detection),
        ("Alert System", test_alert_system),
        ("CCTV Controls", test_cctv_controls),
        ("Audio System", test_audio_system),
        ("Recording System", test_recording_system),
        ("Flask Server", test_flask_server)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...", end=" ")
        try:
            result = test_func()
            if result:
                print("✓ PASS")
                results.append((test_name, True))
            else:
                print("✗ FAIL")
                results.append((test_name, False))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some systems need attention.")
    
    return passed == total

if __name__ == "__main__":
    main()
