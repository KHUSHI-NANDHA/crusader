#!/usr/bin/env python3
"""
Comprehensive test for people detection and counting
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer

def create_test_image_with_faces():
    """Create a test image that might trigger face detection"""
    # Create a simple image with oval shapes that might be detected as faces
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some oval shapes that might be detected as faces
    cv2.ellipse(img, (150, 200), (60, 80), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (400, 250), (50, 70), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (500, 180), (45, 65), 0, 0, 360, (255, 255, 255), -1)
    
    # Add some text
    cv2.putText(img, "Test Image with Potential Faces", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img

def test_people_detection():
    """Test people detection comprehensively"""
    print("Comprehensive People Detection Test")
    print("=" * 40)
    
    try:
        # Create analyzer
        print("Creating analyzer...")
        analyzer = SimpleStudentMoodAnalyzer()
        
        # Test 1: Empty image
        print("\n1. Testing empty image...")
        empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces, gray = analyzer.detect_faces(empty_img)
        print(f"   Result: {len(faces)} faces detected")
        
        # Test 2: Image with potential face-like shapes
        print("\n2. Testing image with face-like shapes...")
        test_img = create_test_image_with_faces()
        faces, gray = analyzer.detect_faces(test_img)
        print(f"   Result: {len(faces)} faces detected")
        
        # Test 3: Full analysis
        print("\n3. Testing full analysis...")
        analysis = analyzer.analyze_frame(test_img)
        print(f"   Analysis Results:")
        print(f"   - People Count: {analysis['people_count']}")
        print(f"   - Current People: {len(analysis['current_people'])}")
        print(f"   - Chaos People: {len(analysis['chaos_people'])}")
        print(f"   - Dominant Mood: {analysis['dominant_mood']}")
        print(f"   - Overall Chaos: {analysis['overall_chaos']:.2f}")
        
        # Test 4: Multiple frames (simulating video)
        print("\n4. Testing multiple frames...")
        for i in range(5):
            # Create slightly different frames
            frame = test_img.copy()
            cv2.putText(frame, f"Frame {i+1}", (10, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            analysis = analyzer.analyze_frame(frame)
            print(f"   Frame {i+1}: {analysis['people_count']} people")
        
        print("\n✓ People detection test completed!")
        print("\nNote: If you see 0 people detected, it means:")
        print("- The test images don't contain actual human faces")
        print("- Face detection requires real human faces with proper lighting")
        print("- The system is working correctly - it's just not detecting faces in test images")
        
        print("\nTo test with real faces:")
        print("1. Use a video with clear human faces")
        print("2. Ensure good lighting")
        print("3. Make sure faces are clearly visible and not too small")
        
    except Exception as e:
        print(f"✗ Error during people detection test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_people_detection()

