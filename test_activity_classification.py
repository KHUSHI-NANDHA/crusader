#!/usr/bin/env python3
"""
Test script to verify the new activity classification functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer

def test_activity_classification():
    """Test the new activity classification features"""
    print("Testing Activity Classification System")
    print("=" * 50)
    
    # Create analyzer
    analyzer = SimpleStudentMoodAnalyzer(use_yolo=False)
    
    # Test 1: Structured Group Work
    print("\n1. Testing Structured Group Work Detection...")
    people_data = [
        {'rect': [100, 100, 50, 80], 'last_rect': [100, 100, 50, 80], 'last_seen': 0},
        {'rect': [120, 110, 50, 80], 'last_rect': [120, 110, 50, 80], 'last_seen': 0},
        {'rect': [140, 105, 50, 80], 'last_rect': [140, 105, 50, 80], 'last_seen': 0}
    ]
    chaos_level = 0.2  # Low chaos
    activity_type = analyzer.classify_activity_type(people_data, chaos_level)
    print(f"   Activity Type: {activity_type}")
    print(f"   Expected: structured_group_work")
    print(f"   ✓ PASS" if activity_type == "structured_group_work" else "   ✗ FAIL")
    
    # Test 2: Distractive Group Chaos
    print("\n2. Testing Distractive Group Chaos Detection...")
    people_data = [
        {'rect': [100, 100, 50, 80], 'last_rect': [90, 90, 50, 80], 'last_seen': 0},
        {'rect': [120, 110, 50, 80], 'last_rect': [130, 120, 50, 80], 'last_seen': 0},
        {'rect': [140, 105, 50, 80], 'last_rect': [150, 115, 50, 80], 'last_seen': 0}
    ]
    chaos_level = 0.9  # High chaos
    activity_type = analyzer.classify_activity_type(people_data, chaos_level)
    print(f"   Activity Type: {activity_type}")
    print(f"   Expected: distractive_group_chaos")
    print(f"   ✓ PASS" if activity_type == "distractive_group_chaos" else "   ✗ FAIL")
    
    # Test 3: Individual Work
    print("\n3. Testing Individual Work Detection...")
    people_data = [
        {'rect': [100, 100, 50, 80], 'last_rect': [100, 100, 50, 80], 'last_seen': 0},
        {'rect': [400, 300, 50, 80], 'last_rect': [400, 300, 50, 80], 'last_seen': 0},
        {'rect': [600, 200, 50, 80], 'last_rect': [600, 200, 50, 80], 'last_seen': 0}
    ]
    chaos_level = 0.3  # Low chaos
    activity_type = analyzer.classify_activity_type(people_data, chaos_level)
    print(f"   Activity Type: {activity_type}")
    print(f"   Expected: structured_individual_work")
    print(f"   ✓ PASS" if activity_type == "structured_individual_work" else "   ✗ FAIL")
    
    # Test 4: Activity Summary
    print("\n4. Testing Activity Summary...")
    
    # Simulate some activity history
    analyzer.activity_history.extend([
        "structured_group_work",
        "structured_group_work", 
        "unstructured_individual_work",
        "distractive_group_chaos",
        "structured_individual_work"
    ])
    
    summary = analyzer.get_activity_summary()
    print(f"   Total Periods: {summary['total_periods']}")
    print(f"   Structured Group Work: {summary['structured_group_work']} ({summary['structured_group_work_percentage']:.1f}%)")
    print(f"   Distractive Group Chaos: {summary['distractive_group_chaos']} ({summary['distractive_group_chaos_percentage']:.1f}%)")
    
    expected_periods = 5
    print(f"   ✓ PASS" if summary['total_periods'] == expected_periods else "   ✗ FAIL")
    
    # Test 5: Group Work Score Calculation
    print("\n5. Testing Group Work Score Calculation...")
    close_people = [
        {'rect': [100, 100, 50, 80]},
        {'rect': [120, 110, 50, 80]},  # Close to first person
        {'rect': [500, 500, 50, 80]}   # Far from others
    ]
    
    group_score = analyzer._calculate_group_work_score(close_people)
    print(f"   Group Work Score: {group_score:.3f}")
    print(f"   Expected: > 0.3 (some group work detected)")
    print(f"   ✓ PASS" if group_score > 0.3 else "   ✗ FAIL")
    
    # Test 6: Structured Score Calculation
    print("\n6. Testing Structured Score Calculation...")
    stable_people = [
        {'rect': [100, 100, 50, 80], 'last_rect': [100, 100, 50, 80], 'last_seen': 0},
        {'rect': [200, 200, 50, 80], 'last_rect': [200, 200, 50, 80], 'last_seen': 0}
    ]
    low_chaos = 0.1
    
    structured_score = analyzer._calculate_structured_score(stable_people, low_chaos)
    print(f"   Structured Score: {structured_score:.3f}")
    print(f"   Expected: > 0.6 (structured activity)")
    print(f"   ✓ PASS" if structured_score > 0.6 else "   ✗ FAIL")
    
    print("\n" + "=" * 50)
    print("Activity Classification Test Complete!")
    print("The system can now distinguish between:")
    print("• Structured vs Unstructured activities")
    print("• Group work vs Individual work")
    print("• Productive vs Distractive chaos")
    print("• Real-time activity monitoring and reporting")

if __name__ == "__main__":
    test_activity_classification()
