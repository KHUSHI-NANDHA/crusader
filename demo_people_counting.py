import cv2
import numpy as np
from simple_mood_detector import SimpleStudentMoodAnalyzer

def create_demo_video():
    """Create a simple demo video with face-like objects"""
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_people_counting.mp4', fourcc, 10.0, (640, 480))
    
    # Create analyzer
    analyzer = SimpleStudentMoodAnalyzer()
    
    print("Creating demo video with people counting...")
    
    for frame_num in range(100):  # 100 frames = 10 seconds at 10 FPS
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some moving face-like objects
        for i in range(3):
            x = 100 + i * 150 + int(20 * np.sin(frame_num * 0.1 + i))
            y = 200 + int(10 * np.cos(frame_num * 0.1 + i))
            cv2.ellipse(frame, (x, y), (40, 60), 0, 0, 360, (255, 255, 255), -1)
        
        # Analyze frame
        analysis = analyzer.analyze_frame(frame)
        
        # Draw results on frame
        cv2.putText(frame, f"People Detected: {analysis['people_count']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_num}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw face rectangles
        faces, gray = analyzer.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Write frame
        out.write(frame)
        
        if frame_num % 20 == 0:
            print(f"Processed frame {frame_num}/100")
    
    # Release everything
    out.release()
    print("Demo video created: demo_people_counting.mp4")
    print("You can now load this video in the main application to see people counting in action!")

if __name__ == "__main__":
    create_demo_video()

