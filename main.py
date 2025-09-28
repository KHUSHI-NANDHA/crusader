
import sys
import tkinter as tk
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
    
    # Global fail-safe to prevent hard exits
    def excepthook(exc_type, exc_value, exc_traceback):
        print(f"Global exception: {exc_type.__name__}: {exc_value}")
        try:
            tk.messagebox.showerror("Unexpected Error", f"{exc_type.__name__}: {exc_value}\nThe application will remain open.")
        except Exception:
            pass

    sys.excepthook = excepthook

    # Create and run the video processor
    app = SimpleVideoProcessor()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    # Never exit the process due to other exceptions

if __name__ == "__main__":
    main()
