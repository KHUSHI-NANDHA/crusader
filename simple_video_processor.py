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
        # Set OpenCV to single-threaded mode to avoid threading conflicts
        cv2.setNumThreads(1)
        
        self.root = tk.Tk()
        self.root.title("Student Mood & Chaos Detection System (YOLO Version) - Window stays open until you close it")
        self.root.geometry("1200x800")
        
        # Prevent window from closing accidentally
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.analyzer = SimpleStudentMoodAnalyzer(use_yolo=True)
        self.attendance_tracker = AttendanceTracker()
        self.is_processing = False
        self.current_video = None
        self.cap = None
        self.video_list = []  # List of selected videos
        self.current_video_index = 0
        self.is_processing_all = False
        
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
        
        # Advanced Chaos Status
        ttk.Label(results_frame, text="Chaos Status:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_status_var = tk.StringVar(value="ANALYZING...")
        chaos_status_label = ttk.Label(results_frame, textvariable=self.chaos_status_var, font=("Arial", 14, "bold"))
        chaos_status_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        
        # Analysis Status
        ttk.Label(results_frame, text="Analysis Status:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.analysis_status_var = tk.StringVar(value="Collecting data...")
        ttk.Label(results_frame, textvariable=self.analysis_status_var, font=("Arial", 10)).grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Activity Type
        ttk.Label(results_frame, text="Activity Type:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.activity_type_var = tk.StringVar(value="Unknown")
        ttk.Label(results_frame, textvariable=self.activity_type_var, font=("Arial", 12, "bold")).grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Group Work Indicator
        ttk.Label(results_frame, text="Group Work:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.group_work_var = tk.StringVar(value="0%")
        ttk.Label(results_frame, textvariable=self.group_work_var, font=("Arial", 10)).grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Structured Activity Indicator
        ttk.Label(results_frame, text="Structured Activity:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        self.structured_activity_var = tk.StringVar(value="0%")
        ttk.Label(results_frame, textvariable=self.structured_activity_var, font=("Arial", 10)).grid(row=6, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Attendance tracking section
        attendance_frame = ttk.LabelFrame(main_frame, text="Attendance Tracking", padding="10")
        attendance_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Attendance controls (automatic lecture management)
        ttk.Button(attendance_frame, text="View Attendance Report", command=self.view_attendance_report).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(attendance_frame, text="Export Report", command=self.export_attendance_report).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(attendance_frame, text="Activity Analysis", command=self.show_activity_analysis).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(attendance_frame, text="Clear All Data", command=self.clear_attendance_data).grid(row=0, column=3, padx=(0, 10))
        
        # Current session info (automatic)
        self.current_session_var = tk.StringVar(value="Auto-managed")
        ttk.Label(attendance_frame, text="Session Status:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(attendance_frame, textvariable=self.current_session_var, font=("Arial", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # People in current session
        self.people_in_session_var = tk.StringVar(value="0 people")
        ttk.Label(attendance_frame, text="People Present:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(attendance_frame, textvariable=self.people_in_session_var, font=("Arial", 10)).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Status
        ttk.Label(results_frame, text="Status:").grid(row=10, column=0, sticky=tk.W, pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready - Sessions auto-managed, window stays open")
        ttk.Label(results_frame, textvariable=self.status_var, font=("Arial", 10)).grid(row=10, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Add restart button for error recovery
        
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
        """Browse for single video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            # Clear all previous data when selecting new video
            self.attendance_tracker.clear_all_data()
            self.clear_session_data()
            
            # Set new video
            self.video_path_var.set(file_path)
            self.current_video = file_path
            
            # Reset status
            self.status_var.set("Selected new video - previous data cleared")
            
            print(f"📁 Selected new video: {os.path.basename(file_path)}")
            print("🗑️ Previous data automatically cleared for fresh start")
    
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
            # Clear all previous data when selecting new videos
            self.attendance_tracker.clear_all_data()
            self.clear_session_data()
            
            # Set new video list
            self.video_list = list(filenames)
            self.current_video_index = 0
            self.update_video_list_display()
            
            # Reset status
            self.status_var.set(f"Selected {len(filenames)} new videos - previous data cleared")
            
            print(f"📁 Selected {len(filenames)} videos for processing")
            print("🗑️ Previous data automatically cleared for fresh start")
    
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
        
        # Clear all previous data for fresh start
        self.clear_session_data()
        
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
            self.status_var.set("All videos processed - attendance tracking complete")
            messagebox.showinfo("Processing Complete", "All videos have been processed!\nCheck the attendance report for results.")
            return
        
        # Get current video
        current_video = self.video_list[self.current_video_index]
        self.current_video = current_video
        self.video_path_var.set(current_video)
        
        # Clear all previous data for fresh start
        self.clear_session_data()
        
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
        
        # Clear all previous data for fresh start
        self.clear_session_data()
        
        self.is_processing = True
        self.status_var.set("Processing...")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video, args=(video_path,))
        thread.daemon = True
        thread.start()
        
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        
        # Properly release video capture
        if self.cap:
            try:
                self.cap.release()
            except:
                pass  # Ignore release errors
            self.cap = None
        
        # If processing multiple videos automatically, end current lecture and move to next
        if self.is_processing_all:
            # End current lecture
            self.attendance_tracker.end_session()
            
            # Move to next video
            self.current_video_index += 1
            if self.current_video_index < len(self.video_list):
                # Process next video after a longer delay to avoid threading conflicts
                self.root.after(2000, self.process_next_video)
            else:
                # Finished all videos
                self.is_processing_all = False
                self.status_var.set("All videos processed - attendance tracking complete")
                messagebox.showinfo("Processing Complete", "All videos have been processed!\nCheck the attendance report for results.")
        else:
            # Manual control - just stop, don't move to next video
            self.status_var.set("Stopped - Use Next/Previous buttons to navigate")
        
    def process_video(self, video_path):
        """Process video file"""
        try:
            print(f"Opening video: {video_path}")
            
            # Set OpenCV threading to single thread to avoid conflicts
            cv2.setNumThreads(1)
            
            # Create new VideoCapture with proper backend
            self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                # Try with default backend if FFMPEG fails
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
                    print(f"⚠️ Error processing frame {frame_count} (continuing): {e}")
                    # Continue processing instead of crashing
                    continue
                
                # No delay for maximum speed
                # cv2.waitKey(1)  # Commented out for maximum speed
                
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(f"⚠️ Video processing error (continuing): {error_msg}")
            # Don't show error dialog for known threading issues, just log them
            if "async_lock" not in str(e) and "pthread_frame" not in str(e) and "cascadedetect" not in str(e):
                messagebox.showerror("Error", error_msg)
            # Continue processing instead of crashing
        finally:
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass  # Ignore release errors
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
                    self.status_var.set("All videos processed - attendance tracking complete")
                    messagebox.showinfo("Processing Complete", "All videos have been processed!\nCheck the attendance report for results.")
            else:
                # End current session for single video processing
                if self.attendance_tracker.current_session_id:
                    self.attendance_tracker.end_session()
                    self.current_session_var.set("Auto-managed")
                    self.status_var.set("Video processed - session ended automatically")
                else:
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
        
        # Draw advanced chaos status
        chaos_status = analysis.get('overall_chaos_status', 'ANALYZING')
        chaos_clusters = analysis.get('chaos_cluster_count', 0)
        chaos_people = analysis.get('chaos_people_count', 0)
        
        if chaos_status == 'CHAOS':
            chaos_color = (0, 0, 255)  # Red
            status_text = f"CHAOS DETECTED!"
        elif chaos_status == 'CALM':
            chaos_color = (0, 255, 0)  # Green
            status_text = f"All Calm"
        else:
            chaos_color = (255, 255, 0)  # Yellow
            status_text = f"Analyzing..."
        
        cv2.putText(frame, f"Status: {status_text}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, chaos_color, 2)
        cv2.putText(frame, f"Chaos Clusters: {chaos_clusters}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, chaos_color, 2)
        cv2.putText(frame, f"Chaos People: {chaos_people}", 
                   (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, chaos_color, 2)
        
        # Add real-time chaos alert overlay
        if chaos_status == 'CHAOS':
            # Add red alert overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Add flashing "CHAOS DETECTED" text
            flash_alpha = 0.8 * (1 + np.sin(time.time() * 15) / 2)
            cv2.putText(frame, "⚠️ CHAOS DETECTED! ⚠️", 
                       (frame.shape[1]//2 - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Add countdown or timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw activity counts
        individual_work = analysis.get('individual_work_count', 0)
        group_work = analysis.get('structured_group_work_count', 0)
        structured_chaos = analysis.get('structured_chaos_count', 0)
        clusters = analysis.get('active_clusters', 0)
        
        cv2.putText(frame, f"Individual Work: {individual_work}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Group Work: {group_work}", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Structured Chaos: {structured_chaos}", 
                   (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Active Clusters: {clusters}", 
                   (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw mood distribution
        mood_text = f"Avg People: {int(analysis['average_people'])}"
        cv2.putText(frame, mood_text, 
                   (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw individual face rectangles with different colors for chaos creators
        current_people = analysis.get('current_people', [])
        chaos_people = analysis.get('chaos_people', [])
        
        # Get advanced chaos data
        chaos_clusters = analysis.get('chaos_clusters', [])
        individual_chaos_people = analysis.get('individual_chaos_people', [])
        chaos_status = analysis.get('overall_chaos_status', 'ANALYZING')
        
        # Draw debug info
        debug_text = f"Status: {chaos_status} | People: {len(current_people)}"
        cv2.putText(frame, debug_text, 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Create sets for chaos detection
        chaos_ids = {person['id'] for person in chaos_people}
        individual_chaos_ids = {person['person_id'] for person in individual_chaos_people}
        
        # Get cluster information for highlighting
        clusters = analysis.get('clusters', [])
        chaos_cluster_ids = {cluster['cluster_id'] for cluster in chaos_clusters}
        
        for person in current_people:
            x, y, w, h = person['rect']
            person_id = person['id']
            chaos_level = person.get('chaos_level', 0)
            chaos_level_name = person.get('chaos_level_name', 'CALM')
            movement_type = person.get('movement_type', 'STATIONARY')
            activity_type = person.get('activity_type', 'CALM')
            activity_name = person.get('activity_name', 'Calm Activity')
            movement_speed = person.get('movement_speed', 0)
            
            # Check if person is in chaos (from advanced detection)
            is_individual_chaos = person_id in individual_chaos_ids
            is_basic_chaos = person_id in chaos_ids
            
            # Determine if person is chaotic
            is_chaotic = is_individual_chaos or is_basic_chaos
            
            # Show activity type and speed
            cv2.putText(frame, f"{activity_name}", 
                       (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Speed: {movement_speed:.1f}", 
                       (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Choose color and thickness based on chaos detection
            if is_chaotic:
                # RED for chaotic people - real-time chaos detection
                color = (0, 0, 255)  # Red
                thickness = 4
                chaos_label = "CHAOS!"
            elif activity_type == 'INDIVIDUAL_WORK':
                color = (0, 255, 0)  # Green for individual work
                thickness = 2
                chaos_label = "WORK"
            elif activity_type == 'STRUCTURED_GROUP_WORK':
                color = (255, 255, 0)  # Yellow for group work
                thickness = 2
                chaos_label = "GROUP"
            else:
                color = (255, 0, 0)  # Blue for calm activity
                thickness = 2
                chaos_label = "CALM"
            
            # Draw rectangle with chaos-based color
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Add special effects for chaotic people
            if is_chaotic:
                # Add pulsing red effect for chaotic people
                pulse = int(30 * (1 + np.sin(time.time() * 15) / 2))
                cv2.rectangle(frame, (x-pulse//8, y-pulse//8), (x+w+pulse//8, y+h+pulse//8), (0, 0, 255), 2)
                
                # Add flashing background
                flash_alpha = 0.3 * (1 + np.sin(time.time() * 20) / 2)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, flash_alpha, frame, 1 - flash_alpha, 0, frame)
            
            # Draw chaos label
            cv2.putText(frame, chaos_label, 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw person ID and dummy name
            dummy_name = person.get('dummy_name', f'Person {person_id}')
            cv2.putText(frame, f"ID:{person_id}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, dummy_name, 
                       (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Highlight chaotic clusters
        for cluster in clusters:
            cluster_id = cluster.get('id', '')
            if cluster_id in chaos_cluster_ids:
                # Get cluster center and size
                center = cluster.get('center', (0, 0))
                cluster_size = cluster.get('size', 0)
                
                # Draw cluster boundary
                cluster_radius = 80 + cluster_size * 10
                cv2.circle(frame, (int(center[0]), int(center[1])), cluster_radius, (0, 0, 255), 3)
                
                # Add pulsing effect for chaotic clusters
                pulse = int(20 * (1 + np.sin(time.time() * 12) / 2))
                cv2.circle(frame, (int(center[0]), int(center[1])), cluster_radius + pulse, (0, 0, 255), 2)
                
                # Draw cluster label
                cv2.putText(frame, f"CHAOTIC CLUSTER", 
                           (int(center[0]) - 60, int(center[1]) - cluster_radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
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
        
        # Update advanced chaos status
        chaos_status = analysis.get('overall_chaos_status', 'ANALYZING')
        self.chaos_status_var.set(chaos_status)
        
        # Update activity type
        activity_type = analysis.get('activity_type', 'unknown')
        self.activity_type_var.set(activity_type.replace('_', ' ').title())
        
        # Update activity summary
        activity_summary = analysis.get('activity_summary', {})
        if activity_summary:
            # Calculate group work percentage
            group_work_count = (activity_summary.get('structured_group_work', 0) + 
                              activity_summary.get('unstructured_group_work', 0) +
                              activity_summary.get('distractive_group_chaos', 0))
            total_periods = activity_summary.get('total_periods', 1)
            group_work_percentage = (group_work_count / total_periods) * 100 if total_periods > 0 else 0
            self.group_work_var.set(f"{group_work_percentage:.1f}%")
            
            # Calculate structured activity percentage
            structured_count = (activity_summary.get('structured_group_work', 0) + 
                              activity_summary.get('structured_individual_work', 0))
            structured_percentage = (structured_count / total_periods) * 100 if total_periods > 0 else 0
            self.structured_activity_var.set(f"{structured_percentage:.1f}%")
        
        # Update analysis status
        if chaos_status == 'ANALYZING':
            self.analysis_status_var.set("Collecting data...")
        elif chaos_status == 'CHAOS':
            self.analysis_status_var.set("CHAOS DETECTED!")
        else:
            self.analysis_status_var.set("All calm")
        
        
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
    
    def show_activity_analysis(self):
        """Show detailed activity analysis window"""
        # Get activity summary from analyzer
        activity_summary = self.analyzer.get_activity_summary()
        
        # Create new window
        activity_window = tk.Toplevel(self.root)
        activity_window.title("Classroom Activity Analysis")
        activity_window.geometry("600x500")
        activity_window.resizable(True, True)
        
        # Create main frame
        main_frame = ttk.Frame(activity_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="📊 CLASSROOM ACTIVITY ANALYSIS", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for different analysis views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Activity Distribution Tab
        distribution_frame = ttk.Frame(notebook)
        notebook.add(distribution_frame, text="Activity Distribution")
        
        # Activity summary
        ttk.Label(distribution_frame, text="Activity Type Distribution:", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        
        # Create activity breakdown
        activities = [
            ("Structured Group Work", "structured_group_work", "green"),
            ("Unstructured Group Work", "unstructured_group_work", "orange"),
            ("Structured Individual Work", "structured_individual_work", "blue"),
            ("Unstructured Individual Work", "unstructured_individual_work", "yellow"),
            ("Distractive Group Chaos", "distractive_group_chaos", "red"),
            ("Distractive Individual Chaos", "distractive_individual_chaos", "red"),
            ("No Activity", "no_activity", "gray")
        ]
        
        for activity_name, activity_key, color in activities:
            count = activity_summary.get(activity_key, 0)
            percentage = activity_summary.get(f"{activity_key}_percentage", 0)
            
            frame = ttk.Frame(distribution_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{activity_name}:", width=25, anchor="w").pack(side=tk.LEFT)
            ttk.Label(frame, text=f"{count} periods", width=10).pack(side=tk.LEFT, padx=(10, 5))
            ttk.Label(frame, text=f"({percentage:.1f}%)", width=10, foreground=color).pack(side=tk.LEFT)
        
        # Productivity Analysis Tab
        productivity_frame = ttk.Frame(notebook)
        notebook.add(productivity_frame, text="Productivity Analysis")
        
        ttk.Label(productivity_frame, text="Productivity Metrics:", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        
        # Calculate productivity metrics
        total_periods = activity_summary.get('total_periods', 0)
        if total_periods > 0:
            productive_work = (activity_summary.get('structured_group_work', 0) + 
                             activity_summary.get('structured_individual_work', 0))
            group_work = (activity_summary.get('structured_group_work', 0) + 
                         activity_summary.get('unstructured_group_work', 0))
            chaos_periods = (activity_summary.get('distractive_group_chaos', 0) + 
                           activity_summary.get('distractive_individual_chaos', 0))
            
            productive_percentage = (productive_work / total_periods) * 100
            group_work_percentage = (group_work / total_periods) * 100
            chaos_percentage = (chaos_periods / total_periods) * 100
            
            metrics = [
                ("Total Analysis Periods:", f"{total_periods}"),
                ("Productive Work:", f"{productive_percentage:.1f}%"),
                ("Group Work:", f"{group_work_percentage:.1f}%"),
                ("Chaos/Distraction:", f"{chaos_percentage:.1f}%"),
                ("", ""),
                ("Recommendations:", ""),
            ]
            
            for metric_name, metric_value in metrics:
                if metric_name == "Recommendations:":
                    ttk.Label(productivity_frame, text=metric_name, font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")
                    
                    # Add recommendations based on analysis
                    recommendations = []
                    if chaos_percentage > 30:
                        recommendations.append("• High chaos levels detected - consider classroom management strategies")
                    if group_work_percentage < 20:
                        recommendations.append("• Low group work activity - encourage collaborative learning")
                    if productive_percentage < 50:
                        recommendations.append("• Low structured activity - consider more guided instruction")
                    if group_work_percentage > 70 and chaos_percentage < 10:
                        recommendations.append("• Excellent group work environment - maintain current approach")
                    
                    for rec in recommendations:
                        ttk.Label(productivity_frame, text=rec, wraplength=500, justify="left").pack(anchor="w", pady=2)
                else:
                    frame = ttk.Frame(productivity_frame)
                    frame.pack(fill=tk.X, pady=2)
                    ttk.Label(frame, text=metric_name, width=25, anchor="w").pack(side=tk.LEFT)
                    ttk.Label(frame, text=metric_value, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        else:
            ttk.Label(productivity_frame, text="No activity data available yet. Process some video to see analysis.", 
                     foreground="gray").pack(pady=20)
        
        # Timeline Tab
        timeline_frame = ttk.Frame(notebook)
        notebook.add(timeline_frame, text="Activity Timeline")
        
        ttk.Label(timeline_frame, text="Recent Activity Timeline:", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        
        # Show recent activity history
        if hasattr(self.analyzer, 'activity_history') and self.analyzer.activity_history:
            timeline_text = tk.Text(timeline_frame, height=15, width=70, wrap=tk.WORD)
            timeline_scrollbar = ttk.Scrollbar(timeline_frame, orient=tk.VERTICAL, command=timeline_text.yview)
            timeline_text.configure(yscrollcommand=timeline_scrollbar.set)
            
            timeline_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            timeline_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create timeline text
            timeline_content = "Recent Activity Sequence:\n" + "="*50 + "\n\n"
            
            for i, activity in enumerate(reversed(list(self.analyzer.activity_history)[-50:]), 1):
                activity_display = activity.replace('_', ' ').title()
                timeline_content += f"{i:2d}. {activity_display}\n"
            
            timeline_text.insert(tk.END, timeline_content)
            timeline_text.config(state=tk.DISABLED)
        else:
            ttk.Label(timeline_frame, text="No timeline data available yet.", foreground="gray").pack(pady=20)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=activity_window.destroy).pack(pady=(20, 0))
    
    def export_attendance_report(self):
        """Export attendance report to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.attendance_tracker.export_attendance_report(filename)
            messagebox.showinfo("Export Complete", f"Attendance report exported to:\n{filename}")
    
    def clear_session_data(self):
        """Clear all session data for fresh video processing"""
        # Clear analyzer data
        if hasattr(self.analyzer, 'reset_session'):
            self.analyzer.reset_session()
        
        # Reset UI variables to show fresh state
        self.people_count_var.set("0")
        self.mood_var.set("Unknown")
        self.chaos_status_var.set("ANALYZING...")
        self.analysis_status_var.set("Starting fresh analysis...")
        self.activity_type_var.set("Unknown")
        self.group_work_var.set("0%")
        self.structured_activity_var.set("0%")
        self.people_in_session_var.set("0 people")
        
        # Clear video display
        self.video_label.config(image='', text="No video loaded", background="black", foreground="white")
        
        print("🧹 Session data cleared for fresh start")
    
    def clear_attendance_data(self):
        """Clear all attendance data for a fresh start"""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all attendance data?\nThis will remove ALL previous session data and start fresh."):
            # Clear all data completely
            self.attendance_tracker.clear_all_data()
            
            # Reset all UI variables
            self.clear_session_data()
            
            # Reset video list and index
            self.video_list = []
            self.current_video_index = 0
            self.video_path_var.set("")
            self.video_list_var.set("No videos selected")
            
            # Reset status
            self.current_session_var.set("Auto-managed")
            self.people_in_session_var.set("0 people")
            self.status_var.set("All data cleared - ready for fresh start with new videos")
            
            # Clear video display
            self.video_label.config(image='', text="No video loaded", background="black", foreground="white")
            
            print("🗑️ ALL DATA CLEARED - Starting completely fresh")
            messagebox.showinfo("Data Cleared", "ALL previous session data has been completely cleared.\nYou can now upload new videos and start fresh.")
    
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_processing:
            self.stop_processing()
        
        # End current session if active
        if self.attendance_tracker.current_session_id:
            self.attendance_tracker.end_session()
        
        # Clean up resources
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        
        # Ask for confirmation before closing
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.root.destroy()
        else:
            # Don't close, just minimize or continue
            self.root.iconify()
    
    def run(self):
        """Run the application"""
        try:
            # Set up error handling for the main loop
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start the main loop with error handling
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
            self.on_closing()
        except Exception as e:
            print(f"⚠️ Error in main loop (continuing): {e}")
            # Don't close the application, just log the error and continue
            try:
                self.root.mainloop()
            except:
                pass

if __name__ == "__main__":
    app = SimpleVideoProcessor()
    app.run()
