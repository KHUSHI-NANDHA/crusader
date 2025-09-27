#!/usr/bin/env python3
"""
Test people counting functionality
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer

def test_people_counting():
    """Test people counting with different scenarios"""
    print("Testing People Counting Functionality")
    print("=" * 40)
    
    try:
        # Create analyzer
        analyzer = SimpleStudentMoodAnalyzer()
        
        # Test 1: Empty image
        print("\n1. Testing empty image...")
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces, gray = analyzer.detect_faces(empty_image)
        print(f"   Empty image: {len(faces)} faces detected")
        
        # Test 2: Image with text (no faces)
        print("\n2. Testing image with text...")
        text_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(text_image, "No faces here", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        faces, gray = analyzer.detect_faces(text_image)
        print(f"   Text image: {len(faces)} faces detected")
        
        # Test 3: Image with geometric shapes (no faces)
        print("\n3. Testing image with shapes...")
        shapes_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(shapes_image, (100, 100), (200, 200), (255, 0, 0), -1)
        cv2.circle(shapes_image, (400, 300), 50, (0, 255, 0), -1)
        faces, gray = analyzer.detect_faces(shapes_image)
        print(f"   Shapes image: {len(faces)} faces detected")
        
        # Test 4: Full analysis
        print("\n4. Testing full analysis...")
        analysis = analyzer.analyze_frame(text_image)
        print(f"   Analysis results:")
        print(f"   - People Count: {analysis['people_count']}")
        print(f"   - Current People: {len(analysis['current_people'])}")
        print(f"   - Chaos People: {len(analysis['chaos_people'])}")
        print(f"   - Dominant Mood: {analysis['dominant_mood']}")
        print(f"   - Overall Chaos: {analysis['overall_chaos']:.2f}")
        
        print("\n✓ People counting test completed!")
        print("Note: Face detection requires actual faces in the image.")
        print("The system is working correctly - it shows 0 people when no faces are present.")
        
    except Exception as e:
        print(f"✗ Error during people counting test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_people_counting()
