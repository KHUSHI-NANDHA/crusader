#!/usr/bin/env python3
"""
Debug script to test YOLO detection on a single frame.
This helps diagnose why detection counts are low.
"""

import cv2
import numpy as np
from yolo_human_detector import YOLOHumanDetector

def debug_yolo_detection(video_path, frame_number=100):
    """Debug YOLO detection on a specific frame."""
    print("YOLO Detection Debug Tool")
    print("=" * 40)
    
    # Initialize YOLO detector with different confidence thresholds
    confidence_thresholds = [0.1, 0.25, 0.5, 0.7]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Seek to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_number}")
        cap.release()
        return
    
    print(f"Testing frame {frame_number} from {video_path}")
    print(f"Frame size: {frame.shape}")
    
    # Test different confidence thresholds
    for conf_thresh in confidence_thresholds:
        print(f"\n--- Testing confidence threshold: {conf_thresh} ---")
        
        try:
            detector = YOLOHumanDetector(model_size='s', confidence_threshold=conf_thresh)
            detections, detection_time = detector.detect_humans(frame)
            
            print(f"Detection time: {detection_time:.3f}s")
            print(f"People detected: {len(detections)}")
            
            if len(detections) > 0:
                confidences = [d['confidence'] for d in detections]
                print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
                print(f"Average confidence: {np.mean(confidences):.3f}")
                
                # Show first few detections
                for i, det in enumerate(detections[:5]):  # Show first 5
                    print(f"  Detection {i+1}: bbox={det['bbox']}, conf={det['confidence']:.3f}")
                
                if len(detections) > 5:
                    print(f"  ... and {len(detections) - 5} more")
            
            # Save frame with detections for visual inspection
            debug_frame = frame.copy()
            for det in detections:
                x, y, w, h = det['bbox']
                confidence = det['confidence']
                
                # Draw bounding box
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence
                cv2.putText(debug_frame, f"{confidence:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save debug image
            debug_filename = f"debug_frame_{frame_number}_conf_{conf_thresh}.jpg"
            cv2.imwrite(debug_filename, debug_frame)
            print(f"Debug image saved: {debug_filename}")
            
        except Exception as e:
            print(f"Error with confidence {conf_thresh}: {e}")
    
    cap.release()
    print("\nDebug complete!")

def test_raw_yolo_detection(video_path, frame_number=100):
    """Test raw YOLO detection without our wrapper."""
    print("\n" + "=" * 40)
    print("Testing Raw YOLO Detection")
    print("=" * 40)
    
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('yolov8s.pt')
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Could not read frame")
            return
        
        # Test different confidence thresholds
        for conf in [0.1, 0.25, 0.5]:
            print(f"\nTesting raw YOLO with confidence {conf}:")
            results = model(frame, conf=conf, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    # Count person detections
                    person_count = sum(1 for class_id in class_ids if int(class_id) == 0)
                    print(f"  Total detections: {len(class_ids)}")
                    print(f"  Person detections: {person_count}")
                    
                    if person_count > 0:
                        person_confidences = [confidences[i] for i, class_id in enumerate(class_ids) if int(class_id) == 0]
                        print(f"  Person confidence range: {min(person_confidences):.3f} - {max(person_confidences):.3f}")
                else:
                    print("  No detections found")
            else:
                print("  No results")
                
    except Exception as e:
        print(f"Error in raw YOLO test: {e}")

def main():
    """Main debug function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_yolo_detection.py <video_path> [frame_number]")
        print("Example: python debug_yolo_detection.py video.mp4 100")
        return
    
    video_path = sys.argv[1]
    frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    debug_yolo_detection(video_path, frame_number)
    test_raw_yolo_detection(video_path, frame_number)

if __name__ == "__main__":
    main()


