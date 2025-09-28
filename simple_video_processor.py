import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from simple_mood_detector import SimpleStudentMoodAnalyzer
from attendance_tracker import AttendanceTracker
import os
import numpy as np
import time
from PIL import Image, ImageTk

class SimpleVideoProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Student Mood & Chaos Detection System - Window stays open until YOU close it")
        self.root.geometry("1200x800")
        # Reduce OpenCV/FFmpeg threading contention
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        
        # Prevent window from closing accidentally
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Keep window alive on callback errors
        self.root.report_callback_exception = self.tk_error_handler
        
        self.analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)
        self.attendance_tracker = AttendanceTracker()
        self.is_processing = False
        self.current_video = None
        self.cap = None
        self.video_list = []  # List of selected videos
        self.current_video_index = 0
        self.is_processing_all = False
        
        # Minimal UI mode hides status and advanced chaos/activity widgets
        self.minimal_ui = True
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video selection frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding="10")
        video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Single video selection
        single_frame = ttk.Frame(video_frame)
        single_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.video_path_var = tk.StringVar()
        ttk.Entry(single_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(single_frame, text="Browse Single", command=self.browse_video).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(single_frame, text="Start Processing", command=self.start_processing).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(single_frame, text="Stop", command=self.stop_processing).grid(row=0, column=3)
        
        # Multiple video selection
        multi_frame = ttk.Frame(video_frame)
        multi_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(multi_frame, text="Select Multiple Videos", command=self.browse_multiple_videos).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(multi_frame, text="Process All Videos", command=self.process_all_videos).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(multi_frame, text="Next Video", command=self.next_video).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(multi_frame, text="Previous Video", command=self.previous_video).grid(row=0, column=3, padx=(0, 10))
        ttk.Button(multi_frame, text="Clear List", command=self.clear_video_list).grid(row=0, column=4)
        
        # Video list display
        self.video_list_var = tk.StringVar(value="No videos selected")
        ttk.Label(multi_frame, text="Selected Videos:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.video_list_label = ttk.Label(multi_frame, textvariable=self.video_list_var, font=("Arial", 9))
        self.video_list_label.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
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
        
        # Advanced/chaos/activity widgets (created but optionally hidden)
        chaos_status_label_text = ttk.Label(results_frame, text="Chaos Status:")
        chaos_status_label_text.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_status_var = tk.StringVar(value="ANALYZING...")
        chaos_status_label = ttk.Label(results_frame, textvariable=self.chaos_status_var, font=("Arial", 14, "bold"))
        chaos_status_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        chaos_clusters_label_text = ttk.Label(results_frame, text="Chaos Clusters:")
        chaos_clusters_label_text.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_clusters_var = tk.StringVar(value="0")
        chaos_clusters_value = ttk.Label(results_frame, textvariable=self.chaos_clusters_var, font=("Arial", 14, "bold"), foreground="red")
        chaos_clusters_value.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        chaos_people_label_text = ttk.Label(results_frame, text="Chaos People:")
        chaos_people_label_text.grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_people_var = tk.StringVar(value="0")
        chaos_people_value = ttk.Label(results_frame, textvariable=self.chaos_people_var, font=("Arial", 14, "bold"), foreground="red")
        chaos_people_value.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        analysis_status_label_text = ttk.Label(results_frame, text="Analysis Status:")
        analysis_status_label_text.grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.analysis_status_var = tk.StringVar(value="Collecting data...")
        analysis_status_value = ttk.Label(results_frame, textvariable=self.analysis_status_var, font=("Arial", 10))
        analysis_status_value.grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        individual_work_label_text = ttk.Label(results_frame, text="Individual Work:")
        individual_work_label_text.grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        self.individual_work_var = tk.StringVar(value="0")
        individual_work_value = ttk.Label(results_frame, textvariable=self.individual_work_var, font=("Arial", 10))
        individual_work_value.grid(row=6, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        group_work_label_text = ttk.Label(results_frame, text="Group Work:")
        group_work_label_text.grid(row=7, column=0, sticky=tk.W, pady=(5, 0))
        self.group_work_var = tk.StringVar(value="0")
        group_work_value = ttk.Label(results_frame, textvariable=self.group_work_var, font=("Arial", 10))
        group_work_value.grid(row=7, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))

        structured_chaos_label_text = ttk.Label(results_frame, text="Structured Chaos:")
        structured_chaos_label_text.grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
        self.structured_chaos_var = tk.StringVar(value="0")
        structured_chaos_value = ttk.Label(results_frame, textvariable=self.structured_chaos_var, font=("Arial", 10), foreground="red")
        structured_chaos_value.grid(row=8, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))

        active_clusters_label_text = ttk.Label(results_frame, text="Active Clusters:")
        active_clusters_label_text.grid(row=9, column=0, sticky=tk.W, pady=(5, 0))
        self.clusters_var = tk.StringVar(value="0")
        clusters_value = ttk.Label(results_frame, textvariable=self.clusters_var, font=("Arial", 10))
        clusters_value.grid(row=9, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Attendance tracking section
        attendance_frame = ttk.LabelFrame(main_frame, text="Attendance Tracking", padding="10")
        attendance_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Attendance controls (automatic lecture management)
        ttk.Button(attendance_frame, text="View Attendance Report", command=self.view_attendance_report).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(attendance_frame, text="Export Report", command=self.export_attendance_report).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(attendance_frame, text="Clear All Data", command=self.clear_attendance_data).grid(row=0, column=2, padx=(0, 10))
        
        # Current session info (automatic)
        self.current_session_var = tk.StringVar(value="Auto-managed")
        ttk.Label(attendance_frame, text="Session Status:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(attendance_frame, textvariable=self.current_session_var, font=("Arial", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # People in current session
        self.people_in_session_var = tk.StringVar(value="0 people")
        ttk.Label(attendance_frame, text="People Present:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(attendance_frame, textvariable=self.people_in_session_var, font=("Arial", 10)).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Status
        # Status is not shown in minimal UI, but keep variable for internal messaging
        self.status_var = tk.StringVar(value="Ready")
        if not self.minimal_ui:
            ttk.Label(results_frame, text="Status:").grid(row=10, column=0, sticky=tk.W, pady=(10, 0))
            ttk.Label(results_frame, textvariable=self.status_var, font=("Arial", 10)).grid(row=10, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        video_display_frame.columnconfigure(0, weight=1)
        video_display_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)

        # Hide advanced widgets if minimal UI is requested
        if self.minimal_ui:
            for w in [chaos_status_label_text, chaos_status_label,
                      chaos_clusters_label_text, chaos_clusters_value,
                      chaos_people_label_text, chaos_people_value,
                      analysis_status_label_text, analysis_status_value,
                      individual_work_label_text, individual_work_value,
                      group_work_label_text, group_work_value,
                      structured_chaos_label_text, structured_chaos_value,
                      active_clusters_label_text, clusters_value]:
                try:
                    w.grid_remove()
                except Exception:
                    pass
        
    def browse_video(self):
        """Browse for single video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.current_video = file_path
    
    def browse_multiple_videos(self):
        """Browse for multiple video files"""
        filenames = filedialog.askopenfilenames(
            title="Select Multiple Video Files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filenames:
            self.video_list = list(filenames)
            self.current_video_index = 0
            self.update_video_list_display()
            print(f"📁 Selected {len(filenames)} videos for processing")
    
    def update_video_list_display(self):
        """Update the video list display"""
        if self.video_list:
            video_names = [os.path.basename(video) for video in self.video_list]
            current_video_name = video_names[self.current_video_index] if self.current_video_index < len(video_names) else "None"
            display_text = f"Video {self.current_video_index + 1}/{len(self.video_list)}: {current_video_name}"
            self.video_list_var.set(display_text)
        else:
            self.video_list_var.set("No videos selected")
    
    def clear_video_list(self):
        """Clear the video list"""
        self.video_list = []
        self.current_video_index = 0
        self.update_video_list_display()
        print("🗑️ Video list cleared")
    
    def process_all_videos(self):
        """Process all selected videos in sequence"""
        if not self.video_list:
            messagebox.showwarning("No Videos", "Please select multiple videos first")
            return
        
        if self.is_processing:
            messagebox.showwarning("Already Processing", "Please stop current processing first")
            return
        
        # Start processing all videos
        self.is_processing_all = True
        self.current_video_index = 0
        self.process_next_video()
    
    def next_video(self):
        """Move to the next video manually"""
        if not self.video_list:
            messagebox.showwarning("No Videos", "Please select multiple videos first")
            return
        
        # Stop current video if processing
        if self.is_processing:
            self.stop_processing()
        
        # Move to next video
        self.current_video_index += 1
        if self.current_video_index >= len(self.video_list):
            self.current_video_index = 0  # Loop back to first video
        
        # Process the next video
        self.load_current_video()
    
    def previous_video(self):
        """Move to the previous video manually"""
        if not self.video_list:
            messagebox.showwarning("No Videos", "Please select multiple videos first")
            return
        
        # Stop current video if processing
        if self.is_processing:
            self.stop_processing()
        
        # Move to previous video
        self.current_video_index -= 1
        if self.current_video_index < 0:
            self.current_video_index = len(self.video_list) - 1  # Loop to last video
        
        # Process the previous video
        self.load_current_video()
    
    def load_current_video(self):
        """Load and start processing the current video"""
        if not self.video_list or self.current_video_index >= len(self.video_list):
            return
        
        current_video = self.video_list[self.current_video_index]
        self.current_video = current_video
        self.video_path_var.set(current_video)
        
        # Start new lecture for this video
        self.attendance_tracker.start_new_session(current_video)
        self.current_session_var.set(f"Session {self.current_video_index + 1}: {os.path.basename(current_video)}")
        
        # Start processing this video
        self.start_processing()
        
        # Update status and display
        self.status_var.set(f"Video {self.current_video_index + 1}/{len(self.video_list)}: {os.path.basename(current_video)}")
        self.update_video_list_display()
        print(f"🎬 Loading video {self.current_video_index + 1}/{len(self.video_list)}: {os.path.basename(current_video)}")
    
    def process_next_video(self):
        """Process the next video in the list"""
        if not self.is_processing_all or self.current_video_index >= len(self.video_list):
            # Finished processing all videos
            self.is_processing_all = False
            self.status_var.set("All videos processed - window stays open until you close it")
            print("✅ All videos processed! Window stays open - you can view reports or process more videos.")
            return
        
        # Get current video
        current_video = self.video_list[self.current_video_index]
        self.current_video = current_video
        self.video_path_var.set(current_video)
        
        # Start new lecture for this video
        self.attendance_tracker.start_new_session(current_video)
        self.current_session_var.set(f"Session {self.current_video_index + 1}: {os.path.basename(current_video)}")
        
        # Start processing this video
        self.start_processing()
        
        # Update status
        self.status_var.set(f"Processing video {self.current_video_index + 1}/{len(self.video_list)}")
        print(f"🎬 Processing video {self.current_video_index + 1}/{len(self.video_list)}: {os.path.basename(current_video)}")
            
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
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        
        # If processing multiple videos automatically, end current lecture and move to next
        if self.is_processing_all:
            # End current lecture
            self.attendance_tracker.end_session()
            
            # Move to next video
            self.current_video_index += 1
            if self.current_video_index < len(self.video_list):
                # Process next video after a short delay
                self.root.after(1000, self.process_next_video)
            else:
                # Finished all videos
                self.is_processing_all = False
                self.status_var.set("All videos processed - window stays open until you close it")
                print("✅ All videos processed! Window stays open - you can view reports or process more videos.")
        else:
            # Manual control - just stop, don't move to next video
            self.status_var.set("Stopped - Use Next/Previous buttons to navigate")
        
    def process_video(self, video_path):
        """Process video file"""
        try:
            print(f"Opening video: {video_path}")
            # Prefer FFmpeg backend, fall back if unavailable
            self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                error_msg = "Could not open video file. Please check if the file exists and is a valid video format."
                print(error_msg)
                # Do not call Tk from background thread
                try:
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                except Exception:
                    pass
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
                    # Stop at end-of-video to avoid FFmpeg async_lock issues from rapid looping
                    self.is_processing = False
                    break
                
                try:
                    # Analyze frame
                    analysis = self.analyzer.analyze_frame(frame, prev_frame, frame_count)
                    
                    # Draw analysis results on frame
                    self.draw_analysis_on_frame(frame, analysis)
                    
                    # Update attendance tracking
                    if self.attendance_tracker.current_session_id:
                        # Register people in current lecture
                        current_people = analysis.get('current_people', [])
                        for person in current_people:
                            person_id = person['id']
                            bbox = person['rect']
                            dummy_name = self.attendance_tracker.register_person(person_id, bbox)
                            if dummy_name:
                                person['dummy_name'] = dummy_name
                        
                        # Update current session info
                        session_info = self.attendance_tracker.get_current_session_info()
                        if session_info:
                            self.people_in_session_var.set(f"{session_info['people_present']} people")
                        
                        # Debug: Print attendance tracking info every 100 frames
                        if frame_count % 100 == 0:
                            print(f"📊 Attendance Debug - Frame {frame_count}:")
                            print(f"   Current people detected: {len(current_people)}")
                            print(f"   People in session: {session_info['people_present'] if session_info else 0}")
                            print(f"   Session ID: {self.attendance_tracker.current_session_id}")
                    else:
                        # Auto-start lecture if not started
                        if frame_count == 0:  # Only auto-start on first frame
                            lecture_name = os.path.basename(self.current_video) if self.current_video else "auto_lecture.mp4"
                            print(f"🚨 No lecture started - auto-starting lecture: {lecture_name}")
                            
                            # Clear old data if this is a new video (not continuing from previous)
                            if not hasattr(self, '_lecture_started_for_video') or self._lecture_started_for_video != self.current_video:
                                print("🔄 Starting fresh session - clearing old data")
                                self.attendance_tracker.clear_all_data()
                                self._lecture_started_for_video = self.current_video
                            
                            self.attendance_tracker.start_new_session(lecture_name)
                            self.current_session_var.set(f"Auto: {lecture_name}")
                            self.status_var.set("Auto-started session - tracking attendance")
                    
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
            # Avoid Tk calls off the main thread
            try:
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            except Exception:
                pass
        finally:
            if self.cap:
                self.cap.release()
            self.is_processing = False
            
            # Handle multiple video processing
            if self.is_processing_all:
                # End current session
                self.attendance_tracker.end_session()
                
                # Move to next video
                self.current_video_index += 1
                if self.current_video_index < len(self.video_list):
                    # Process next video after a short delay
                    self.root.after(2000, self.process_next_video)
                else:
                    # Finished all videos
                    self.is_processing_all = False
                    self.status_var.set("All videos processed - window stays open until you close it")
                    print("✅ All videos processed! Window stays open - you can view reports or process more videos.")
            else:
                # End current session for single video processing
                if self.attendance_tracker.current_session_id:
                    self.attendance_tracker.end_session()
                    self.current_session_var.set("Auto-managed")
                    self.status_var.set("Video finished - Window stays open until YOU close it")
                else:
                    self.status_var.set("Video finished - Window stays open until YOU close it")
            
            print("Video processing stopped")
            
    def draw_analysis_on_frame(self, frame, analysis):
        """Draw analysis results on the frame (minimal: only people count, mood, and boxes)"""
        height, width = frame.shape[:2]

        # Draw people count
        people_count = analysis['people_count']
        cv2.rectangle(frame, (5, 5), (320, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"People: {people_count}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw dominant mood
        mood_text = analysis['dominant_mood'].upper()
        mood_color = self.get_mood_color(analysis['dominant_mood'])
        cv2.putText(frame, f"Mood: {mood_text}", (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mood_color, 2)

        # Draw bounding boxes for current people (simple style)
        current_people = analysis.get('current_people', [])
        for person in current_people:
            x, y, w, h = person['rect']
            person_id = person.get('id', 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 2)
            cv2.putText(frame, f"ID:{person_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
    
    def get_chaos_color(self, chaos_level):
        """Get color for chaos level display"""
        colors = {
            'CALM': (0, 255, 0),        # Green
            'MODERATE': (255, 255, 0),  # Yellow
            'ACTIVE': (255, 165, 0),    # Orange
            'DISRUPTIVE': (255, 0, 0),  # Red
            'CHAOTIC': (128, 0, 128)    # Purple
        }
        return colors.get(chaos_level, (128, 128, 128))
        
    def update_ui(self, analysis):
        """Update UI with analysis results"""
        self.people_count_var.set(str(analysis['people_count']))
        self.mood_var.set(analysis['dominant_mood'].upper())
        if not self.minimal_ui:
            chaos_status = analysis.get('overall_chaos_status', 'ANALYZING')
            self.chaos_status_var.set(chaos_status)
            self.chaos_clusters_var.set(str(analysis.get('chaos_cluster_count', 0)))
            self.chaos_people_var.set(str(analysis.get('chaos_people_count', 0)))
            if chaos_status == 'ANALYZING':
                self.analysis_status_var.set("Collecting data...")
            elif chaos_status == 'CHAOS':
                self.analysis_status_var.set("CHAOS DETECTED!")
            else:
                self.analysis_status_var.set("All calm")
            self.individual_work_var.set(str(analysis.get('individual_work_count', 0)))
            self.group_work_var.set(str(analysis.get('structured_group_work_count', 0)))
            self.structured_chaos_var.set(str(analysis.get('structured_chaos_count', 0)))
            self.clusters_var.set(str(analysis.get('active_clusters', 0)))
        
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
        
    
    def view_attendance_report(self):
        """View attendance report in a new window"""
        summary = self.attendance_tracker.get_attendance_summary()
        person_stats = self.attendance_tracker.get_person_attendance_stats()
        
        # Create new window for report
        report_window = tk.Toplevel(self.root)
        report_window.title("Attendance Report")
        report_window.geometry("800x600")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(report_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate report text
        report_text = "📊 ATTENDANCE REPORT\n"
        report_text += "=" * 50 + "\n\n"
        report_text += f"📚 Total Sessions: {summary['total_sessions']}\n"
        report_text += f"👥 Total People: {summary['total_people']}\n\n"
        
        report_text += "👤 PERSON ATTENDANCE:\n"
        report_text += "-" * 30 + "\n"
        for stat in person_stats:
            report_text += f"{stat['dummy_name']}: {stat['total_sessions_attended']} sessions ({stat['session_percentage']:.1f}%)\n"
        
        report_text += "\n📚 SESSION ATTENDANCE:\n"
        report_text += "-" * 30 + "\n"
        for session_id, session_info in summary['session_attendance'].items():
            report_text += f"{session_info['session_name']}: {session_info['people_count']} people\n"
            report_text += f"  People: {', '.join(session_info['people_present'])}\n\n"
        
        text_widget.insert(tk.END, report_text)
        text_widget.config(state=tk.DISABLED)
    
    def export_attendance_report(self):
        """Export attendance report to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.attendance_tracker.export_attendance_report(filename)
            messagebox.showinfo("Export Complete", f"Attendance report exported to:\n{filename}")
    
    def clear_attendance_data(self):
        """Clear all attendance data for a fresh start"""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all attendance data?\nThis action cannot be undone."):
            self.attendance_tracker.clear_all_data()
            self.current_session_var.set("Auto-managed")
            self.people_in_session_var.set("0 people")
            self.status_var.set("All attendance data cleared - ready for fresh start")
            messagebox.showinfo("Data Cleared", "All attendance data has been cleared.\nYou can now start fresh with new videos.")
    
    def tk_error_handler(self, exc, val, tb):
        """Global Tkinter callback exception handler - keep app running"""
        try:
            print(f"Tkinter callback error: {exc}: {val}")
            if hasattr(self, 'status_var'):
                self.status_var.set("An internal error occurred; window will remain open")
            messagebox.showerror("Unexpected Error", f"{exc.__name__}: {val}\n\nThe application will remain open.")
        except Exception:
            # Ensure no secondary crash in handler
            pass
    
    def on_closing(self):
        """Handle window closing - ask for confirmation"""
        if messagebox.askyesno("Exit Application", "Are you sure you want to close the application?\nThis will stop all processing and close the window."):
            if self.is_processing:
                self.stop_processing()
            
            # End current session if active
            if self.attendance_tracker.current_session_id:
                self.attendance_tracker.end_session()
            
            # Clean up resources
            if self.cap is not None:
                self.cap.release()
            
            # Destroy the window
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleVideoProcessor()
    app.run()
