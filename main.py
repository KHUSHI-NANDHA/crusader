#!/usr/bin/env python3
"""
Student Mood & Chaos Detection System
=====================================

A real-time video analysis system that detects:
- Student moods (happy, excited, calm, sad, neutral)
- People count in the video
- Chaos level based on movement and activity
- Overall classroom atmosphere

Usage:
    python main.py

Features:
- Real-time video processing
- Multiple mood detection
- People counting
- Chaos level analysis
- Simple GUI interface
- Support for MP4 videos
"""

import sys
import os
from simple_video_processor import SimpleVideoProcessor

def main():
    """Main entry point"""
    print("Student Mood & Chaos Detection System")
    print("=====================================")
    print()
    print("This system will analyze video files to detect:")
    print("- Student moods and emotions")
    print("- Number of people in the video")
    print("- Chaos level and activity")
    print("- Overall classroom atmosphere")
    print()
    print("Starting GUI application...")
    print()
    
    try:
        # Create and run the video processor
        app = SimpleVideoProcessor()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
