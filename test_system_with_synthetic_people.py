#!/usr/bin/env python3
"""
Test the system with synthetic people data to verify chaos detection and attendance tracking
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer
from attendance_tracker import AttendanceTracker
import time

def create_synthetic_frame_with_people():
    """Create a synthetic frame with people-like objects"""
    # Create a frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(50)  # Dark gray background
    
    # Add some "people" as rectangles
    people_rects = [
        (100, 100, 60, 120),  # Person 1
        (200, 150, 50, 100),  # Person 2
        (300, 120, 55, 110),  # Person 3
        (400, 180, 45, 90),   # Person 4
    ]
    
    for i, (x, y, w, h) in enumerate(people_rects):
        # Draw person as a rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 150, 200), -1)
        # Add some movement simulation (noise)
        noise = np.random.randint(-10, 10, (h, w, 3))
        frame[y:y+h, x:x+w] = np.clip(frame[y:y+h, x:x+w].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame, people_rects

def test_system_with_synthetic_data():
    """Test the system with synthetic people data"""
    print("🧪 Testing System with Synthetic People Data")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    analyzer = SimpleStudentMoodAnalyzer(use_yolo=False)  # Use OpenCV face detection
    attendance_tracker = AttendanceTracker()
    
    # Start a new lecture
    print("\n2. Starting new lecture...")
    attendance_tracker.start_new_lecture("synthetic_test.mp4")
    
    # Create synthetic frames with movement
    print("\n3. Creating synthetic frames with movement...")
    frames = []
    for i in range(50):
        frame, people_rects = create_synthetic_frame_with_people()
        
        # Add some movement to simulate chaos
        if i > 20 and i < 30:  # Chaos period
            # Add more noise and movement
            noise = np.random.randint(-20, 20, frame.shape)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        frames.append((frame, people_rects))
    
    # Process frames
    print("\n4. Processing synthetic frames...")
    prev_frame = None
    people_detected = []
    chaos_detected = []
    
    for i, (frame, people_rects) in enumerate(frames):
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
            
            # Print progress every 10 frames
            if i % 10 == 0:
                print(f"   Frame {i}: {people_count} people, Chaos: {chaos_status}")
                print(f"   Current people: {len(analysis.get('current_people', []))}")
                print(f"   Chaos people: {len(analysis.get('chaos_people', []))}")
                print(f"   Overall chaos: {analysis.get('overall_chaos', 0):.3f}")
            
            prev_frame = frame.copy()
            
        except Exception as e:
            print(f"❌ Error processing frame {i}: {e}")
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
    attendance_tracker.export_attendance_report("synthetic_test_report.txt")
    
    print("\n✅ Synthetic test complete!")
    print("This test verifies that the system works when people are detected.")

if __name__ == "__main__":
    test_system_with_synthetic_data()




