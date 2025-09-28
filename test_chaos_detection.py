#!/usr/bin/env python3
"""
Test script to verify chaos and calm detection features are working properly.
"""

import cv2
import numpy as np
import time
from enhanced_chaos_analyzer import EnhancedChaosAnalyzer
from simple_mood_detector import SimpleStudentMoodAnalyzer

def test_chaos_analyzer():
    """Test the enhanced chaos analyzer with synthetic data."""
    print("Testing Enhanced Chaos Analyzer")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = EnhancedChaosAnalyzer()
    print("✓ Enhanced chaos analyzer initialized")
    
    # Create test people data
    test_people = [
        {
            'id': 1,
            'rect': (100, 100, 80, 120),
            'center': (140, 160),
            'chaos_level': 30,  # Moderate chaos
            'movement_speed': 15,  # Work speed
            'movement_vector': (5, 3)
        },
        {
            'id': 2,
            'rect': (200, 200, 80, 120),
            'center': (240, 260),
            'chaos_level': 70,  # High chaos
            'movement_speed': 25,  # Chaos speed
            'movement_vector': (10, 8)
        },
        {
            'id': 3,
            'rect': (300, 300, 80, 120),
            'center': (340, 360),
            'chaos_level': 10,  # Low chaos
            'movement_speed': 5,  # Calm speed
            'movement_vector': (1, 1)
        }
    ]
    
    # Create test frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some movement to frame2
    cv2.rectangle(frame2, (105, 105), (185, 225), (255, 255, 255), -1)
    cv2.rectangle(frame2, (205, 205), (285, 325), (255, 255, 255), -1)
    cv2.rectangle(frame2, (305, 305), (385, 425), (255, 255, 255), -1)
    
    try:
        # Test enhanced chaos analysis
        people_with_activities, individual_activities, cluster_activities, clusters = analyzer.analyze_enhanced_chaos(
            test_people, frame1, frame2
        )
        
        print(f"✓ Analyzed {len(people_with_activities)} people")
        print(f"✓ Found {len(individual_activities)} individual activities")
        print(f"✓ Found {len(cluster_activities)} cluster activities")
        print(f"✓ Detected {len(clusters)} clusters")
        
        # Check individual activities
        for person_id, activity in individual_activities.items():
            print(f"  Person {person_id}: {activity['activity_name']} (Speed: {activity['speed']:.1f})")
        
        # Check cluster activities
        for cluster_id, activity in cluster_activities.items():
            print(f"  Cluster {cluster_id}: {activity['activity_name']} (Size: {activity['cluster_size']})")
        
        # Get activity summary
        summary = analyzer.get_activity_summary()
        print(f"\nActivity Summary:")
        print(f"  Individual Work: {summary['individual_work']}")
        print(f"  Structured Group Work: {summary['structured_group_work']}")
        print(f"  Structured Chaos: {summary['structured_chaos']}")
        print(f"  Individual Chaos: {summary['individual_chaos']}")
        print(f"  Calm: {summary['calm']}")
        print(f"  Active Clusters: {summary['active_clusters']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in chaos analysis: {e}")
        return False

def test_mood_analyzer():
    """Test the mood analyzer with enhanced chaos detection."""
    print("\nTesting Mood Analyzer with Enhanced Chaos Detection")
    print("=" * 50)
    
    try:
        # Initialize mood analyzer
        analyzer = SimpleStudentMoodAnalyzer(use_yolo=False)  # Use OpenCV for testing
        print("✓ Mood analyzer initialized")
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some test faces
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (300, 150), (400, 250), (255, 255, 255), -1)
        
        # Test frame analysis
        analysis = analyzer.analyze_frame(frame)
        
        print(f"✓ Frame analysis completed")
        print(f"  People count: {analysis['people_count']}")
        print(f"  Chaos level: {analysis.get('chaos_level', 0):.1f}")
        print(f"  Chaos level name: {analysis.get('chaos_level_name', 'Unknown')}")
        print(f"  Individual work: {analysis.get('individual_work_count', 0)}")
        print(f"  Group work: {analysis.get('structured_group_work_count', 0)}")
        print(f"  Structured chaos: {analysis.get('structured_chaos_count', 0)}")
        print(f"  Individual chaos: {analysis.get('individual_chaos_count', 0)}")
        print(f"  Calm: {analysis.get('calm_count', 0)}")
        print(f"  Active clusters: {analysis.get('active_clusters', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in mood analyzer: {e}")
        return False

def test_activity_classification():
    """Test activity classification with different scenarios."""
    print("\nTesting Activity Classification")
    print("=" * 35)
    
    analyzer = EnhancedChaosAnalyzer()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Individual Work',
            'speed': 15,
            'chaos_level': 20,
            'expected': 'INDIVIDUAL_WORK'
        },
        {
            'name': 'Individual Chaos',
            'speed': 25,
            'chaos_level': 70,
            'expected': 'INDIVIDUAL_CHAOS'
        },
        {
            'name': 'Calm Activity',
            'speed': 5,
            'chaos_level': 10,
            'expected': 'CALM'
        }
    ]
    
    for scenario in test_scenarios:
        person_data = {
            'id': 1,
            'chaos_level': scenario['chaos_level']
        }
        
        activity_type = analyzer.classify_individual_activity(
            person_data, scenario['speed'], (1, 1)
        )
        
        status = "✓" if activity_type == scenario['expected'] else "✗"
        print(f"{status} {scenario['name']}: {activity_type} (expected: {scenario['expected']})")
    
    return True

def main():
    """Main test function."""
    print("Enhanced Chaos Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Enhanced chaos analyzer
    test1_passed = test_chaos_analyzer()
    
    # Test 2: Mood analyzer integration
    test2_passed = test_mood_analyzer()
    
    # Test 3: Activity classification
    test3_passed = test_activity_classification()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Enhanced Chaos Analyzer: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Mood Analyzer Integration: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Activity Classification: {'PASS' if test3_passed else 'FAIL'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✓ All tests passed! Chaos and calm features are working correctly.")
        return True
    else:
        print("\n✗ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)









