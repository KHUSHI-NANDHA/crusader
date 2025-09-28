#!/usr/bin/env python3
"""
Debug script to test single file processing issues
"""

import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer
from attendance_tracker import AttendanceTracker
import os
import time

def test_single_file_processing():
    """Test single file processing to identify issues"""
    print("🔍 Debugging Single File Processing Issues")
    print("=" * 50)
    
    # Check if demo video exists
    demo_video = "demo_people_counting.mp4"
    if not os.path.exists(demo_video):
        print(f"❌ Demo video not found: {demo_video}")
        return
    
    print(f"📹 Testing with video: {demo_video}")
    
    # Initialize components
    print("\n1. Initializing components...")
    analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)
    attendance_tracker = AttendanceTracker()
    
    # Start a new lecture
    print("\n2. Starting new lecture...")
    attendance_tracker.start_new_lecture(demo_video)
    
    # Open video
    print("\n3. Opening video...")
    cap = cv2.VideoCapture(demo_video)
    if not cap.isOpened():
        print(f"❌ Could not open video: {demo_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 Video FPS: {fps}, Total frames: {total_frames}")
    
    # Process first 100 frames
    print("\n4. Processing first 100 frames...")
    frame_count = 0
    prev_frame = None
    people_detected = []
    chaos_detected = []
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            print("❌ End of video reached")
            break
        
        try:
            # Analyze frame
            analysis = analyzer.analyze_frame(frame, prev_frame, frame_count)
            
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
            
            # Print progress every 20 frames
            if frame_count % 20 == 0:
                print(f"   Frame {frame_count}: {people_count} people, Chaos: {chaos_status}")
                print(f"   Current people: {len(analysis.get('current_people', []))}")
                print(f"   Chaos people: {len(analysis.get('chaos_people', []))}")
            
            prev_frame = frame.copy()
            frame_count += 1
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_count}: {e}")
            continue
    
    cap.release()
    
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
    
    # Check if report file was created
    if os.path.exists("attendance_records.json"):
        print("✅ Attendance records file exists")
        with open("attendance_records.json", 'r') as f:
            content = f.read()
            print(f"📄 File size: {len(content)} characters")
            if len(content) > 100:
                print("✅ File has content")
            else:
                print("❌ File is too small or empty")
    else:
        print("❌ Attendance records file not found")
    
    print("\n🔍 Debug complete!")

if __name__ == "__main__":
    test_single_file_processing()




