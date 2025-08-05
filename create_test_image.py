#!/usr/bin/env python3
"""
Create a test image for the Face Recognition API
"""

from PIL import Image, ImageDraw
import os

def create_test_face_image():
    """Create a simple test face image"""
    
    # Create a 200x200 image with a white background
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    # Head outline
    draw.ellipse([40, 40, 160, 160], outline='black', width=3)
    
    # Eyes
    draw.ellipse([70, 80, 85, 95], fill='black')  # Left eye
    draw.ellipse([115, 80, 130, 95], fill='black')  # Right eye
    
    # Nose
    draw.line([100, 95, 100, 115], fill='black', width=2)
    
    # Mouth
    draw.arc([80, 110, 120, 140], 0, 180, fill='black', width=2)
    
    # Save the image
    test_image_path = "test_face.jpg"
    img.save(test_image_path, "JPEG", quality=95)
    
    print(f"✅ Test image created: {test_image_path}")
    print(f"📁 Full path: {os.path.abspath(test_image_path)}")
    
    return test_image_path

if __name__ == "__main__":
    create_test_face_image() 