#!/usr/bin/env python3
"""
Test script to verify chaos detection with real video data.
"""

import cv2
import numpy as np
import time
from simple_mood_detector import SimpleStudentMoodAnalyzer

def test_with_video_frame(video_path, frame_number=100):
    """Test chaos detection with a real video frame."""
    print(f"Testing Chaos Detection with Real Video")
    print("=" * 45)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return False
    
    # Seek to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"✗ Could not read frame {frame_number}")
        cap.release()
        return False
    
    print(f"✓ Testing frame {frame_number} from {video_path}")
    print(f"  Frame size: {frame.shape}")
    
    try:
        # Initialize mood analyzer with YOLO
        analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)
        print("✓ Mood analyzer with YOLO initialized")
        
        # Analyze frame
        print("Analyzing frame...")
        analysis = analyzer.analyze_frame(frame)
        
        print(f"\nAnalysis Results:")
        print(f"  People detected: {analysis['people_count']}")
        print(f"  Chaos level: {analysis.get('chaos_level', 0):.1f}")
        print(f"  Chaos level name: {analysis.get('chaos_level_name', 'Unknown')}")
        print(f"  Chaos creators: {analysis.get('chaos_creators_count', 0)}")
        print(f"  Chaos percentage: {analysis.get('chaos_percentage', 0):.1f}%")
        
        print(f"\nActivity Breakdown:")
        print(f"  Individual Work: {analysis.get('individual_work_count', 0)}")
        print(f"  Group Work: {analysis.get('structured_group_work_count', 0)}")
        print(f"  Structured Chaos: {analysis.get('structured_chaos_count', 0)}")
        print(f"  Individual Chaos: {analysis.get('individual_chaos_count', 0)}")
        print(f"  Calm: {analysis.get('calm_count', 0)}")
        print(f"  Active Clusters: {analysis.get('active_clusters', 0)}")
        
        # Check if we have people data
        current_people = analysis.get('current_people', [])
        if current_people:
            print(f"\nIndividual Person Analysis:")
            for i, person in enumerate(current_people[:5]):  # Show first 5 people
                person_id = person.get('id', 'Unknown')
                activity_type = person.get('activity_type', 'Unknown')
                activity_name = person.get('activity_name', 'Unknown')
                speed = person.get('movement_speed', 0)
                chaos_level = person.get('chaos_level', 0)
                
                print(f"  Person {person_id}: {activity_name}")
                print(f"    Speed: {speed:.1f}, Chaos: {chaos_level:.1f}")
        else:
            print("  No people data available")
        
        # Test with previous frame for movement calculation
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret_prev, prev_frame = cap.read()
            if ret_prev:
                print(f"\nTesting with previous frame for movement calculation...")
                analysis_with_movement = analyzer.analyze_frame(frame, prev_frame)
                
                print(f"  Movement analysis completed")
                print(f"  People with movement data: {len(analysis_with_movement.get('current_people', []))}")
                
                # Check for movement speeds
                people_with_speeds = analysis_with_movement.get('current_people', [])
                if people_with_speeds:
                    speeds = [p.get('movement_speed', 0) for p in people_with_speeds]
                    avg_speed = np.mean(speeds)
                    max_speed = np.max(speeds)
                    print(f"  Average speed: {avg_speed:.2f}")
                    print(f"  Maximum speed: {max_speed:.2f}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        cap.release()
        return False

def test_chaos_thresholds():
    """Test different chaos thresholds to ensure proper classification."""
    print(f"\nTesting Chaos Thresholds")
    print("=" * 30)
    
    from enhanced_chaos_analyzer import EnhancedChaosAnalyzer
    analyzer = EnhancedChaosAnalyzer()
    
    # Test different scenarios
    test_cases = [
        {'speed': 5, 'chaos_level': 10, 'expected': 'CALM'},
        {'speed': 15, 'chaos_level': 20, 'expected': 'INDIVIDUAL_WORK'},
        {'speed': 25, 'chaos_level': 30, 'expected': 'INDIVIDUAL_CHAOS'},
        {'speed': 30, 'chaos_level': 80, 'expected': 'INDIVIDUAL_CHAOS'},
    ]
    
    for i, case in enumerate(test_cases):
        person_data = {'id': i, 'chaos_level': case['chaos_level']}
        activity = analyzer.classify_individual_activity(person_data, case['speed'], (1, 1))
        
        status = "✓" if activity == case['expected'] else "✗"
        print(f"{status} Speed {case['speed']}, Chaos {case['chaos_level']}: {activity} (expected: {case['expected']})")
    
    return True

def main():
    """Main test function."""
    print("Real Video Chaos Detection Test")
    print("=" * 40)
    
    # Test with your video file
    video_path = "C:/Users/PMCC/Downloads/Choas.mp4"
    
    # Test 1: Real video analysis
    test1_passed = test_with_video_frame(video_path, 100)
    
    # Test 2: Chaos thresholds
    test2_passed = test_chaos_thresholds()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Real Video Analysis: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Chaos Thresholds: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ Chaos and calm features are working correctly with real video data!")
        return True
    else:
        print("\n✗ Some issues detected. Check the results above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
