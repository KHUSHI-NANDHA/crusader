#!/usr/bin/env python3
"""
Test the system functionality with a more realistic scenario
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer
from attendance_tracker import AttendanceTracker
import time
import os

def create_realistic_test_frame():
    """Create a more realistic test frame that might be detected as faces"""
    # Create a frame with better contrast
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(100)  # Gray background
    
    # Create face-like patterns using OpenCV drawing functions
    # These should be more likely to be detected by face detection
    
    # Face 1 - Simple oval with eyes and mouth
    cv2.ellipse(frame, (150, 150), (40, 60), 0, 0, 360, (200, 180, 160), -1)  # Face
    cv2.circle(frame, (140, 130), 5, (0, 0, 0), -1)  # Left eye
    cv2.circle(frame, (160, 130), 5, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(frame, (150, 170), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Face 2 - Another face
    cv2.ellipse(frame, (300, 200), (35, 55), 0, 0, 360, (220, 190, 170), -1)  # Face
    cv2.circle(frame, (290, 185), 4, (0, 0, 0), -1)  # Left eye
    cv2.circle(frame, (310, 185), 4, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(frame, (300, 215), (12, 6), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Face 3 - Third face
    cv2.ellipse(frame, (450, 180), (30, 50), 0, 0, 360, (180, 160, 140), -1)  # Face
    cv2.circle(frame, (440, 165), 4, (0, 0, 0), -1)  # Left eye
    cv2.circle(frame, (460, 165), 4, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(frame, (450, 195), (10, 5), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    return frame

def test_system_with_realistic_faces():
    """Test the system with more realistic face-like patterns"""
    print("🧪 Testing System with Realistic Face Patterns")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    analyzer = SimpleStudentMoodAnalyzer(use_yolo=False)  # Use OpenCV face detection
    attendance_tracker = AttendanceTracker()
    
    # Start a new lecture
    print("\n2. Starting new lecture...")
    attendance_tracker.start_new_lecture("realistic_test.mp4")
    
    # Create frames with movement
    print("\n3. Creating frames with movement...")
    frames = []
    for i in range(30):
        frame = create_realistic_test_frame()
        
        # Add some movement to simulate chaos
        if i > 10 and i < 20:  # Chaos period
            # Add noise to simulate movement
            noise = np.random.randint(-15, 15, frame.shape)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        frames.append(frame)
    
    # Process frames
    print("\n4. Processing frames...")
    prev_frame = None
    people_detected = []
    chaos_detected = []
    
    for i, frame in enumerate(frames):
        try:
            # Analyze frame
            analysis = analyzer.analyze_frame(frame, prev_frame, i)
            
            # Track people detection
            people_count = analysis['people_count']
            people_detected.append(people_count)
            
            # Track chaos detection
            chaos_status = analysis.get('overall_chaos_status', 'UNKNOWN')
            chaos_detected.append(chaos_status)
            
            # Register people with attendance tracker
            if attendance_tracker.current_lecture_id:
                for person in analysis.get('current_people', []):
                    person_id = person['id']
                    bbox = person['rect']
                    dummy_name = attendance_tracker.register_person(person_id, bbox)
                    if dummy_name:
                        person['dummy_name'] = dummy_name
            
            # Print progress every 5 frames
            if i % 5 == 0:
                print(f"   Frame {i}: {people_count} people, Chaos: {chaos_status}")
                print(f"   Current people: {len(analysis.get('current_people', []))}")
                print(f"   Chaos people: {len(analysis.get('chaos_people', []))}")
                print(f"   Overall chaos: {analysis.get('overall_chaos', 0):.3f}")
            
            prev_frame = frame.copy()
            
        except Exception as e:
            print(f"❌ Error processing frame {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Analyze results
    print("\n5. Analyzing results...")
    if people_detected:
        max_people = max(people_detected)
        avg_people = sum(people_detected) / len(people_detected)
        print(f"📊 People Detection:")
        print(f"   Max people detected: {max_people}")
        print(f"   Average people detected: {avg_people:.1f}")
        print(f"   Frames with people: {sum(1 for p in people_detected if p > 0)}/{len(people_detected)}")
    
    if chaos_detected:
        chaos_count = sum(1 for c in chaos_detected if c == 'CHAOS')
        calm_count = sum(1 for c in chaos_detected if c == 'CALM')
        analyzing_count = sum(1 for c in chaos_detected if c == 'ANALYZING')
        print(f"📊 Chaos Detection:")
        print(f"   Chaos frames: {chaos_count}")
        print(f"   Calm frames: {calm_count}")
        print(f"   Analyzing frames: {analyzing_count}")
    
    # Check attendance tracking
    print("\n6. Checking attendance tracking...")
    lecture_info = attendance_tracker.get_current_lecture_info()
    if lecture_info:
        print(f"📚 Current lecture: {lecture_info['lecture_name']}")
        print(f"👥 People in lecture: {lecture_info['people_present']}")
        print(f"👤 People list: {lecture_info['people_list']}")
    else:
        print("❌ No current lecture info")
    
    # End lecture and check report
    print("\n7. Ending lecture and checking report...")
    attendance_tracker.end_lecture()
    
    # Get attendance summary
    summary = attendance_tracker.get_attendance_summary()
    print(f"📊 Attendance Summary:")
    print(f"   Total lectures: {summary['total_lectures']}")
    print(f"   Total people: {summary['total_people']}")
    
    # Export report
    print("\n8. Exporting attendance report...")
    try:
        attendance_tracker.export_attendance_report("realistic_test_report.txt")
        print("✅ Report exported successfully")
    except Exception as e:
        print(f"❌ Error exporting report: {e}")
    
    print("\n✅ Realistic test complete!")

def test_webcam_if_available():
    """Test with webcam if available"""
    print("\n🎥 Testing with webcam (if available)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No webcam available")
        return False
    
    print("✅ Webcam detected! You can test with real people.")
    print("Press 'q' to quit, 's' to start processing")
    
    # Initialize components
    analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)  # Use YOLO for real people
    attendance_tracker = AttendanceTracker()
    
    processing = False
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for better performance
        frame = cv2.resize(frame, (640, 480))
        
        if processing:
            try:
                # Analyze frame
                analysis = analyzer.analyze_frame(frame, prev_frame, frame_count)
                
                # Draw results on frame
                people_count = analysis['people_count']
                chaos_status = analysis.get('overall_chaos_status', 'UNKNOWN')
                
                cv2.putText(frame, f"People: {people_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Status: {chaos_status}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Register people
                if attendance_tracker.current_lecture_id:
                    for person in analysis.get('current_people', []):
                        person_id = person['id']
                        bbox = person['rect']
                        dummy_name = attendance_tracker.register_person(person_id, bbox)
                
                prev_frame = frame.copy()
                frame_count += 1
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Show frame
        cv2.imshow('Test System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not processing:
            print("🎬 Starting processing...")
            attendance_tracker.start_new_lecture("webcam_test.mp4")
            processing = True
        elif key == ord('e') and processing:
            print("⏹️ Ending processing...")
            attendance_tracker.end_lecture()
            processing = False
    
    cap.release()
    cv2.destroyAllWindows()
    
    if processing:
        attendance_tracker.end_lecture()
    
    return True

if __name__ == "__main__":
    # Test with realistic faces first
    test_system_with_realistic_faces()
    
    # Then test with webcam if available
    test_webcam_if_available()




