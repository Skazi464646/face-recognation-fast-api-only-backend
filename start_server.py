#!/usr/bin/env python3
"""
Startup script for the Face Recognition API
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'insightface',
        'qdrant-client',
        'opencv-python',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True


def check_qdrant():
    """Check if Qdrant is running"""
    print("\n🔍 Checking Qdrant connection...")
    
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
        else:
            print(f"❌ Qdrant returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Qdrant at localhost:6333")
        print("Please start Qdrant first:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return False
    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")
        return False


def start_qdrant():
    """Start Qdrant using Docker if not running"""
    print("\n🚀 Starting Qdrant...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not available. Please install Docker or start Qdrant manually.")
            return False
        
        # Check if Qdrant container is already running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=qdrant', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        
        if 'qdrant' in result.stdout:
            print("✅ Qdrant container is already running")
            return True
        
        # Start Qdrant container
        print("Starting Qdrant container...")
        result = subprocess.run([
            'docker', 'run', '-d',
            '--name', 'qdrant',
            '-p', '6333:6333',
            'qdrant/qdrant'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Qdrant container started successfully")
            # Wait for Qdrant to be ready
            print("⏳ Waiting for Qdrant to be ready...")
            time.sleep(5)
            return True
        else:
            print(f"❌ Failed to start Qdrant: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Qdrant: {e}")
        return False


def start_api_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Face Recognition API...")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['QDRANT_HOST'] = 'localhost'
        env['QDRANT_PORT'] = '6333'
        env['DEBUG'] = 'True'
        
        # Start the server
        subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'app.main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


def main():
    """Main startup function"""
    print("🎯 Face Recognition API Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Qdrant
    if not check_qdrant():
        print("\n🔄 Attempting to start Qdrant...")
        if not start_qdrant():
            print("\n❌ Failed to start Qdrant. Please start it manually:")
            print("  docker run -p 6333:6333 qdrant/qdrant")
            sys.exit(1)
    
    # Wait a moment for Qdrant to be fully ready
    print("\n⏳ Waiting for services to be ready...")
    time.sleep(3)
    
    # Start the API server
    start_api_server()


if __name__ == "__main__":
    main() 