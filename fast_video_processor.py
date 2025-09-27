import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from simple_mood_detector import SimpleStudentMoodAnalyzer
import os
import numpy as np
import time
from PIL import Image, ImageTk

class FastVideoProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Student Mood & Chaos Detection System (High Speed)")
        self.root.geometry("1200x800")
        
        self.analyzer = SimpleStudentMoodAnalyzer()
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
        video_display_frame = ttk.LabelFrame(main_frame, text="Video Feed (High Speed)", padding="10")
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
        
        # Status
        ttk.Label(results_frame, text="Status:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(results_frame, textvariable=self.status_var, font=("Arial", 10)).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
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
        self.status_var.set("Processing at high speed...")
        
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
        """Process video file at maximum speed"""
        try:
            print(f"Opening video: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
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
                    # End of video, restart
                    if self.is_processing:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        prev_frame = None
                        continue
                    else:
                        break
                
                try:
                    # Analyze frame
                    analysis = self.analyzer.analyze_frame(frame, prev_frame, frame_count)
                    
                    # Draw analysis results on frame
                    self.draw_analysis_on_frame(frame, analysis)
                    
                    # Update display (minimal updates for speed)
                    if frame_count % 10 == 0:  # Update every 10th frame only
                        def update_display():
                            self.update_ui(analysis)
                            self.display_frame(frame)
                        
                        self.root.after_idle(update_display)
                    else:
                        # Just display frame
                        def display_only():
                            self.display_frame(frame)
                        
                        self.root.after_idle(display_only)
                    
                    prev_frame = frame.copy()
                    frame_count += 1
                    
                    # Minimal progress reporting
                    if frame_count % 1000 == 0:
                        print(f"Processed {frame_count} frames")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
                
                # No delay for maximum speed
                
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
        # Draw people count
        people_count = analysis['people_count']
        cv2.putText(frame, f"People: {people_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw face rectangles
        current_people = analysis.get('current_people', [])
        for person in current_people:
            x, y, w, h = person['rect']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    def update_ui(self, analysis):
        """Update UI with analysis results"""
        self.people_count_var.set(str(analysis['people_count']))
        
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
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FastVideoProcessor()
    app.run()

