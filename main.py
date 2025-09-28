
import sys
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
