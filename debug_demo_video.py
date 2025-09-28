#!/usr/bin/env python3
"""
Debug script to test YOLO detection on demo video with different settings
"""

import cv2
import numpy as np
from yolo_human_detector import YOLOHumanDetector
import os

def test_demo_video_detection():
    """Test YOLO detection on demo video with different settings"""
    print("🔍 Testing YOLO Detection on Demo Video")
    print("=" * 50)
    
    demo_video = "demo_people_counting.mp4"
    if not os.path.exists(demo_video):
        print(f"❌ Demo video not found: {demo_video}")
        return
    
    # Test different confidence thresholds
    confidence_levels = [0.05, 0.1, 0.15, 0.25, 0.5]
    
    for conf in confidence_levels:
        print(f"\n🎯 Testing with confidence threshold: {conf}")
        
        # Initialize YOLO detector
        detector = YOLOHumanDetector(model_size='s', confidence_threshold=conf)
        
        # Open video
        cap = cv2.VideoCapture(demo_video)
        if not cap.isOpened():
            print(f"❌ Could not open video: {demo_video}")
            continue
        
        # Test first 50 frames
        frame_count = 0
        total_detections = 0
        max_detections = 0
        
        while frame_count < 50:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect humans
            detections, _ = detector.detect_humans(frame)
            people_count = len(detections)
            total_detections += people_count
            max_detections = max(max_detections, people_count)
            
            # Print frame info every 10 frames
            if frame_count % 10 == 0:
                print(f"   Frame {frame_count}: {people_count} people detected")
                if people_count > 0:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    print(f"     Average confidence: {avg_conf:.3f}")
            
            frame_count += 1
        
        cap.release()
        
        avg_detections = total_detections / frame_count if frame_count > 0 else 0
        print(f"📊 Results for confidence {conf}:")
        print(f"   Average detections per frame: {avg_detections:.2f}")
        print(f"   Maximum detections in a frame: {max_detections}")
        print(f"   Total detections: {total_detections}")
    
    # Test with a single frame to see what YOLO detects
    print(f"\n🖼️ Testing single frame detection...")
    cap = cv2.VideoCapture(demo_video)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Test with very low confidence
        detector = YOLOHumanDetector(model_size='s', confidence_threshold=0.01)
        detections, _ = detector.detect_humans(frame)
        
        print(f"📊 Single frame test (confidence 0.01):")
        print(f"   Total detections: {len(detections)}")
        
        if len(detections) > 0:
            for i, det in enumerate(detections):
                print(f"   Detection {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
        else:
            print("   No detections found even with very low confidence")
            
            # Try to save the frame for manual inspection
            cv2.imwrite("debug_frame.jpg", frame)
            print("   Saved frame as 'debug_frame.jpg' for manual inspection")
    
    print("\n🔍 Demo video detection test complete!")

if __name__ == "__main__":
    test_demo_video_detection()




