#!/usr/bin/env python3
"""
Test script for the Face Recognition API
"""

import requests
import json
import time
import os
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"
API_HEALTH_URL = f"{API_BASE_URL}/api/health"
API_REGISTER_URL = f"{API_BASE_URL}/api/faces/register"
API_VERIFY_URL = f"{API_BASE_URL}/api/faces/verify"
API_LIST_URL = f"{API_BASE_URL}/api/faces/list"
API_STATS_URL = f"{API_BASE_URL}/api/stats"


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(API_HEALTH_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_system_stats():
    """Test the system stats endpoint"""
    print("\n📊 Testing system stats...")
    
    try:
        response = requests.get(API_STATS_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System stats: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ System stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System stats error: {e}")
        return False


def test_list_faces():
    """Test listing faces"""
    print("\n👥 Testing list faces...")
    
    try:
        response = requests.get(API_LIST_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ List faces: {data['total_count']} faces found")
            return True
        else:
            print(f"❌ List faces failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ List faces error: {e}")
        return False


def create_test_image():
    """Create a simple test image for testing"""
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple test image
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face-like shape
        draw.ellipse([50, 50, 150, 150], outline='black', width=2)
        draw.ellipse([70, 80, 85, 95], fill='black')  # Left eye
        draw.ellipse([115, 80, 130, 95], fill='black')  # Right eye
        draw.arc([80, 100, 120, 130], 0, 180, fill='black', width=2)  # Smile
        
        # Save to bytes
        import io
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    except ImportError:
        print("⚠️  PIL not available, skipping image creation")
        return None


def test_face_registration():
    """Test face registration"""
    print("\n📝 Testing face registration...")
    
    # Create test image
    image_data = create_test_image()
    if not image_data:
        print("⚠️  Skipping face registration test (no test image)")
        return False
    
    try:
        files = {
            'image': ('test_face.jpg', image_data, 'image/jpeg')
        }
        data = {
            'person_name': 'Test Person',
            'description': 'Test face for API testing'
        }
        
        response = requests.post(API_REGISTER_URL, files=files, data=data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Face registered: {data}")
            return data.get('face_id')
        else:
            print(f"❌ Face registration failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Face registration error: {e}")
        return None


def test_face_verification(face_id=None):
    """Test face verification"""
    print("\n🔍 Testing face verification...")
    
    # Create test image
    image_data = create_test_image()
    if not image_data:
        print("⚠️  Skipping face verification test (no test image)")
        return False
    
    try:
        files = {
            'image': ('test_face.jpg', image_data, 'image/jpeg')
        }
        data = {
            'threshold': 0.6
        }
        
        response = requests.post(API_VERIFY_URL, files=files, data=data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Face verification: {data}")
            return True
        else:
            print(f"❌ Face verification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Face verification error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Face Recognition API Tests")
    print("=" * 50)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("System Stats", test_system_stats),
        ("List Faces", test_list_faces),
        ("Face Registration", test_face_registration),
        ("Face Verification", test_face_verification),
    ]
    
    results = []
    face_id = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Face Registration":
                face_id = test_func()
                results.append((test_name, face_id is not None))
            elif test_name == "Face Verification":
                results.append((test_name, test_func(face_id)))
            else:
                results.append((test_name, test_func()))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")


if __name__ == "__main__":
    main() 