#!/usr/bin/env python3
"""
Demo script showing YOLO human detection in action.
This script demonstrates the YOLO integration with a simple video display.
"""

import cv2
import numpy as np
import time
from yolo_human_detector import YOLOHumanDetector

def create_demo_video():
    """Create a simple demo video with moving rectangles to simulate people."""
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_people_counting.mp4', fourcc, 30.0, (640, 480))
    
    # Create frames with moving rectangles
    for frame_num in range(300):  # 10 seconds at 30 FPS
        # Create black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some moving rectangles to simulate people
        t = frame_num / 30.0  # Time in seconds
        
        # Person 1 - moving left to right
        x1 = int(50 + 200 * np.sin(t * 0.5))
        y1 = 150
        cv2.rectangle(frame, (x1, y1), (x1 + 80, y1 + 120), (255, 255, 255), -1)
        
        # Person 2 - moving up and down
        x2 = 300
        y2 = int(100 + 100 * np.cos(t * 0.3))
        cv2.rectangle(frame, (x2, y2), (x2 + 80, y2 + 120), (255, 255, 255), -1)
        
        # Person 3 - stationary
        x3 = 500
        y3 = 200
        cv2.rectangle(frame, (x3, y3), (x3 + 80, y3 + 120), (255, 255, 255), -1)
        
        # Add some text
        cv2.putText(frame, f"Demo Video - Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Simulated People Movement", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print("Demo video created: demo_people_counting.mp4")

def demo_yolo_detection():
    """Demonstrate YOLO detection on the demo video."""
    print("YOLO Human Detection Demo")
    print("=" * 40)
    
    # Initialize YOLO detector
    try:
        detector = YOLOHumanDetector(model_size='n', confidence_threshold=0.3)
        print("✓ YOLO detector initialized")
    except Exception as e:
        print(f"✗ Failed to initialize YOLO detector: {e}")
        return
    
    # Open the demo video
    cap = cv2.VideoCapture('demo_people_counting.mp4')
    if not cap.isOpened():
        print("✗ Could not open demo video")
        return
    
    print("✓ Demo video opened successfully")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    total_detection_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        # Detect humans
        start_time = time.time()
        detections, detection_time = detector.detect_humans(frame)
        total_detection_time += detection_time
        
        # Track people and get chaos levels
        current_people, chaos_people = detector.track_humans(detections, frame)
        
        # Draw detections on frame
        for i, person in enumerate(current_people):
            x, y, w, h = person['rect']
            confidence = person['confidence']
            chaos_level = person['chaos_level']
            
            # Choose color based on chaos level
            if person['is_chaotic']:
                color = (0, 0, 255)  # Red for chaotic people
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for normal people
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw labels
            label = f"Person {person['id']}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw chaos level
            chaos_text = f"Chaos: {chaos_level:.2f}"
            cv2.putText(frame, chaos_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw statistics
        stats_text = f"People: {len(current_people)} | Chaotic: {len(chaos_people)} | FPS: {1/detection_time:.1f}"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('YOLO Human Detection Demo', frame)
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'demo_frame_{frame_count}.jpg', frame)
            print(f"Frame {frame_count} saved")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print performance stats
    avg_detection_time = total_detection_time / frame_count if frame_count > 0 else 0
    avg_fps = 1 / avg_detection_time if avg_detection_time > 0 else 0
    
    print(f"\nPerformance Statistics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average detection time: {avg_detection_time:.3f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")

def main():
    """Main demo function."""
    print("Creating demo video...")
    create_demo_video()
    
    print("\nStarting YOLO detection demo...")
    demo_yolo_detection()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
