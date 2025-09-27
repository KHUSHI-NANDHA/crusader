#!/usr/bin/env python3
"""
Test script for YOLO human detection integration.
This script tests the YOLO detector independently before full integration.
"""

import cv2
import numpy as np
import time
from yolo_human_detector import YOLOHumanDetector

def test_yolo_detection():
    """Test YOLO human detection on a test image or video."""
    print("Testing YOLO Human Detection")
    print("=" * 40)
    
    # Initialize YOLO detector
    try:
        detector = YOLOHumanDetector(model_size='n', confidence_threshold=0.5)
        print("✓ YOLO detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize YOLO detector: {e}")
        return False
    
    # Test with a simple synthetic image
    print("\nTesting with synthetic image...")
    test_image = create_test_image()
    
    try:
        detections, detection_time = detector.detect_humans(test_image)
        print(f"✓ Detection completed in {detection_time:.3f} seconds")
        print(f"✓ Found {len(detections)} detections")
        
        for i, det in enumerate(detections):
            print(f"  Detection {i+1}: bbox={det['bbox']}, confidence={det['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False

def create_test_image():
    """Create a simple test image with some shapes."""
    # Create a 640x480 image with some basic shapes
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some rectangles to simulate people
    cv2.rectangle(img, (100, 100), (200, 300), (255, 255, 255), -1)  # White rectangle
    cv2.rectangle(img, (300, 150), (400, 350), (255, 255, 255), -1)  # White rectangle
    cv2.rectangle(img, (500, 120), (600, 320), (255, 255, 255), -1)  # White rectangle
    
    # Add some text
    cv2.putText(img, "Test Image for YOLO Detection", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img

def test_video_processing():
    """Test YOLO detection on video processing."""
    print("\nTesting video processing with YOLO...")
    
    # Create a simple video processor
    from simple_mood_detector import SimpleStudentMoodAnalyzer
    
    try:
        analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)
        print("✓ Mood analyzer with YOLO initialized")
        
        # Test with synthetic frame
        test_frame = create_test_image()
        analysis = analyzer.analyze_frame(test_frame)
        
        print(f"✓ Frame analysis completed")
        print(f"  People count: {analysis['people_count']}")
        print(f"  Dominant mood: {analysis['dominant_mood']}")
        print(f"  Chaos level: {analysis['overall_chaos']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Video processing test failed: {e}")
        return False

def main():
    """Main test function."""
    print("YOLO Human Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Basic YOLO detection
    test1_passed = test_yolo_detection()
    
    # Test 2: Video processing integration
    test2_passed = test_video_processing()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"YOLO Detection: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Video Processing: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! YOLO integration is working correctly.")
        return True
    else:
        print("\n✗ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
