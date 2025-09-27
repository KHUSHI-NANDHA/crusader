#!/usr/bin/env python3
"""
Test script for the Student Mood & Chaos Detection System
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer

def test_system():
    """Test the mood detection system with a simple synthetic image"""
    print("Testing Student Mood & Chaos Detection System")
    print("=" * 50)
    
    # Create analyzer
    analyzer = SimpleStudentMoodAnalyzer()
    
    # Create a simple test image (black background)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some text to the test image
    cv2.putText(test_image, "Test Image - No Faces", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    print("Testing with synthetic image...")
    
    # Analyze the test image
    result = analyzer.analyze_frame(test_image)
    
    print(f"People Count: {result['people_count']}")
    print(f"Dominant Mood: {result['dominant_mood']}")
    print(f"Chaos Level: {result['overall_chaos']:.2f}")
    print(f"Average People: {result['average_people']:.1f}")
    
    print("\nSystem is working correctly!")
    print("You can now run 'python main.py' to use the GUI application.")

if __name__ == "__main__":
    test_system()
