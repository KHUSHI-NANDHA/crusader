import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from simple_mood_detector import SimpleStudentMoodAnalyzer
import os
import numpy as np
import time
from PIL import Image, ImageTk

class SimpleVideoProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Student Mood & Chaos Detection System (YOLO Version)")
        self.root.geometry("1200x800")
        
        self.analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)
        self.is_processing = False
        self.current_video = None
        self.cap = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video selection frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding="10")
        video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.video_path_var = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(video_frame, text="Browse", command=self.browse_video).grid(row=0, column=1)
        ttk.Button(video_frame, text="Start Processing", command=self.start_processing).grid(row=0, column=2, padx=(10, 0))
        ttk.Button(video_frame, text="Stop", command=self.stop_processing).grid(row=0, column=3, padx=(10, 0))
        
        # Video display frame
        video_display_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.video_label = ttk.Label(video_display_frame, text="No video loaded", background="black", foreground="white")
        self.video_label.grid(row=0, column=0)
        
        # Analysis results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # People count
        ttk.Label(results_frame, text="People Count:").grid(row=0, column=0, sticky=tk.W)
        self.people_count_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.people_count_var, font=("Arial", 14, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Dominant mood
        ttk.Label(results_frame, text="Dominant Mood:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.mood_var = tk.StringVar(value="Unknown")
        mood_label = ttk.Label(results_frame, textvariable=self.mood_var, font=("Arial", 14, "bold"))
        mood_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Chaos level
        ttk.Label(results_frame, text="Chaos Level:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_var = tk.StringVar(value="0%")
        ttk.Label(results_frame, textvariable=self.chaos_var, font=("Arial", 14, "bold")).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Chaos creators
        ttk.Label(results_frame, text="Chaos Creators:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_creators_var = tk.StringVar(value="None")
        ttk.Label(results_frame, textvariable=self.chaos_creators_var, font=("Arial", 14, "bold"), foreground="red").grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Mood distribution
        ttk.Label(results_frame, text="Mood Distribution:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.mood_dist_var = tk.StringVar(value="No data")
        ttk.Label(results_frame, textvariable=self.mood_dist_var, font=("Arial", 10)).grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Status
        ttk.Label(results_frame, text="Status:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(results_frame, textvariable=self.status_var, font=("Arial", 10)).grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        video_display_frame.columnconfigure(0, weight=1)
        video_display_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        
    def browse_video(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            
    def start_processing(self):
        """Start video processing"""
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already running")
            return
            
        self.is_processing = True
        self.status_var.set("Processing...")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video, args=(video_path,))
        thread.daemon = True
        thread.start()
        
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        if self.cap:
            self.cap.release()
        self.status_var.set("Stopped")
        
    def process_video(self, video_path):
        """Process video file"""
        try:
            print(f"Opening video: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                error_msg = "Could not open video file. Please check if the file exists and is a valid video format."
                print(error_msg)
                messagebox.showerror("Error", error_msg)
                return
                
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video FPS: {fps}, Total frames: {total_frames}")
                
            prev_frame = None
            frame_count = 0
            
            while self.is_processing:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    # End of video, restart or stop
                    if self.is_processing:
                        print("Restarting video...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                        prev_frame = None  # Reset previous frame
                        continue
                    else:
                        break
                
                try:
                    # Analyze frame
                    analysis = self.analyzer.analyze_frame(frame, prev_frame, frame_count)
                    
                    # Draw analysis results on frame
                    self.draw_analysis_on_frame(frame, analysis)
                    
                    # Update UI and display frame in main thread
                    # Only update every 5th frame for maximum speed
                    if frame_count % 5 == 0:  # Update every 5th frame
                        def update_display():
                            self.update_ui(analysis)
                            self.display_frame(frame)
                        
                        self.root.after_idle(update_display)
                    else:
                        # Just display frame without UI update for maximum performance
                        def display_only():
                            self.display_frame(frame)
                        
                        self.root.after_idle(display_only)
                    
                    prev_frame = frame.copy()
                    frame_count += 1
                    
                    # Print progress every 500 frames for speed
                    if frame_count % 500 == 0:
                        print(f"Processed {frame_count} frames")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
                
                # No delay for maximum speed
                # cv2.waitKey(1)  # Commented out for maximum speed
                
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
        finally:
            if self.cap:
                self.cap.release()
            self.is_processing = False
            self.status_var.set("Ready")
            print("Video processing stopped")
            
    def draw_analysis_on_frame(self, frame, analysis):
        """Draw analysis results on the frame"""
        height, width = frame.shape[:2]
        
        # Draw people count with better visibility
        people_count = analysis['people_count']
        cv2.putText(frame, f"People Detected: {people_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Add background rectangle for better visibility
        cv2.rectangle(frame, (5, 5), (300, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"People Detected: {people_count}", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw dominant mood
        mood_color = self.get_mood_color(analysis['dominant_mood'])
        cv2.putText(frame, f"Mood: {analysis['dominant_mood'].upper()}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, mood_color, 2)
        
        # Draw chaos level
        chaos_percent = int(analysis['overall_chaos'] * 100)
        chaos_color = (0, 255 - chaos_percent * 2, chaos_percent * 2)
        cv2.putText(frame, f"Chaos: {chaos_percent}%", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, chaos_color, 2)
        
        # Draw chaos creators count
        chaos_count = len(analysis.get('chaos_people', []))
        cv2.putText(frame, f"Chaos Creators: {chaos_count}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw mood distribution
        mood_text = f"Avg People: {int(analysis['average_people'])}"
        cv2.putText(frame, mood_text, 
                   (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw individual face rectangles with different colors for chaos creators
        current_people = analysis.get('current_people', [])
        chaos_people = analysis.get('chaos_people', [])
        
        # Draw debug info
        debug_text = f"Threshold: 10% | People: {len(current_people)}"
        cv2.putText(frame, debug_text, 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Create a set of chaos person IDs for quick lookup
        chaos_ids = {person['id'] for person in chaos_people}
        
        for person in current_people:
            x, y, w, h = person['rect']
            person_id = person['id']
            chaos_level = person['chaos_level']
            
            # Show chaos level for all people (debug info)
            cv2.putText(frame, f"C:{int(chaos_level*100)}%", 
                       (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if person_id in chaos_ids:
                # Highlight chaos creators with red rectangle and thick border
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)  # Red, thick
                cv2.putText(frame, f"CHAOS {int(chaos_level*100)}%", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add pulsing effect for chaos creators
                pulse = int(50 * (1 + np.sin(time.time() * 10) / 2))
                cv2.rectangle(frame, (x-pulse//10, y-pulse//10), (x+w+pulse//10, y+h+pulse//10), (0, 0, 255), 2)
            else:
                # Normal people with blue rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue, normal
                cv2.putText(frame, f"ID:{person_id}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def get_mood_color(self, mood):
        """Get color for mood display"""
        colors = {
            'happy': (0, 255, 0),      # Green
            'excited': (0, 255, 255),  # Yellow
            'calm': (255, 255, 0),     # Cyan
            'sad': (255, 0, 0),        # Red
            'neutral': (255, 255, 255), # White
            'unknown': (128, 128, 128)  # Gray
        }
        return colors.get(mood, (128, 128, 128))
        
    def update_ui(self, analysis):
        """Update UI with analysis results"""
        self.people_count_var.set(str(analysis['people_count']))
        self.mood_var.set(analysis['dominant_mood'].upper())
        self.chaos_var.set(f"{int(analysis['overall_chaos'] * 100)}%")
        
        # Update chaos creators display
        chaos_people = analysis.get('chaos_people', [])
        if chaos_people:
            chaos_text = f"{len(chaos_people)} people creating chaos"
            self.chaos_creators_var.set(chaos_text)
        else:
            self.chaos_creators_var.set("None")
        
        self.mood_dist_var.set(f"Avg: {int(analysis['average_people'])} people")
        
    def display_frame(self, frame):
        """Display frame in the UI"""
        try:
            # Resize frame to fit in the display area
            display_height = 400
            display_width = int(frame.shape[1] * display_height / frame.shape[0])
            
            resized_frame = cv2.resize(frame, (display_width, display_height))
            
            # Convert BGR to RGB for tkinter
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error displaying frame: {e}")
            # Show error message
            self.video_label.configure(text=f"Error: {str(e)}", image="")
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleVideoProcessor()
    app.run()
