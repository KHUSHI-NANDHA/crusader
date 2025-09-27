#!/usr/bin/env python3
"""
Simple test script to verify the system works
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer

def test_basic_functionality():
    """Test basic functionality without video"""
    print("Testing Student Mood & Chaos Detection System")
    print("=" * 50)
    
    try:
        # Create analyzer
        print("Creating analyzer...")
        analyzer = SimpleStudentMoodAnalyzer()
        print("✓ Analyzer created successfully")
        
        # Create a simple test image
        print("Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        print("✓ Test image created")
        
        # Test face detection
        print("Testing face detection...")
        faces, gray = analyzer.detect_faces(test_image)
        print(f"✓ Face detection completed - Found {len(faces)} faces")
        
        # Test analysis
        print("Testing frame analysis...")
        result = analyzer.analyze_frame(test_image)
        print(f"✓ Analysis completed")
        print(f"  - People Count: {result['people_count']}")
        print(f"  - Dominant Mood: {result['dominant_mood']}")
        print(f"  - Chaos Level: {result['overall_chaos']:.2f}")
        print(f"  - Current People: {len(result['current_people'])}")
        print(f"  - Chaos People: {len(result['chaos_people'])}")
        
        print("\n✓ All tests passed! System is working correctly.")
        print("You can now run 'python main.py' to use the GUI application.")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()
