#!/usr/bin/env python3
"""
Test video display functionality
"""

import cv2
import numpy as np
from simple_video_processor import SimpleVideoProcessor
import tkinter as tk

def test_video_display():
    """Test video display without actual video file"""
    print("Testing video display functionality...")
    
    try:
        # Create a simple test video processor
        app = SimpleVideoProcessor()
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "Test Video Frame", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(test_frame, "This should be visible", (50, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Test display function
        print("Testing display_frame method...")
        app.display_frame(test_frame)
        print("✓ Display method executed successfully")
        
        # Test analysis
        print("Testing analysis...")
        analysis = app.analyzer.analyze_frame(test_frame)
        print(f"✓ Analysis completed - People: {analysis['people_count']}")
        
        # Test drawing
        print("Testing draw_analysis_on_frame...")
        app.draw_analysis_on_frame(test_frame, analysis)
        print("✓ Drawing completed successfully")
        
        print("\n✓ All video display tests passed!")
        print("The video display should work correctly now.")
        
    except Exception as e:
        print(f"✗ Error during video display test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_display()
