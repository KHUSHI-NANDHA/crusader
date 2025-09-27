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
        self.root.title("Student Mood & Chaos Detection System (YOLO Version) - Window stays open until you close it")
        self.root.geometry("1200x800")
        
        # Prevent window from closing accidentally
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
        
        # Advanced Chaos Status
        ttk.Label(results_frame, text="Chaos Status:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_status_var = tk.StringVar(value="ANALYZING...")
        chaos_status_label = ttk.Label(results_frame, textvariable=self.chaos_status_var, font=("Arial", 14, "bold"))
        chaos_status_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Chaos Clusters
        ttk.Label(results_frame, text="Chaos Clusters:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_clusters_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.chaos_clusters_var, font=("Arial", 14, "bold"), foreground="red").grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Chaos People
        ttk.Label(results_frame, text="Chaos People:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.chaos_people_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.chaos_people_var, font=("Arial", 14, "bold"), foreground="red").grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Analysis Status
        ttk.Label(results_frame, text="Analysis Status:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.analysis_status_var = tk.StringVar(value="Collecting data...")
        ttk.Label(results_frame, textvariable=self.analysis_status_var, font=("Arial", 10)).grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Activity distribution
        ttk.Label(results_frame, text="Individual Work:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        self.individual_work_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.individual_work_var, font=("Arial", 10)).grid(row=6, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        ttk.Label(results_frame, text="Group Work:").grid(row=7, column=0, sticky=tk.W, pady=(5, 0))
        self.group_work_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.group_work_var, font=("Arial", 10)).grid(row=7, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(results_frame, text="Structured Chaos:").grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
        self.structured_chaos_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.structured_chaos_var, font=("Arial", 10), foreground="red").grid(row=8, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(results_frame, text="Active Clusters:").grid(row=9, column=0, sticky=tk.W, pady=(5, 0))
        self.clusters_var = tk.StringVar(value="0")
        ttk.Label(results_frame, textvariable=self.clusters_var, font=("Arial", 10)).grid(row=9, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Status
        ttk.Label(results_frame, text="Status:").grid(row=10, column=0, sticky=tk.W, pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready - Window stays open until you close it")
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
            
            # Draw person ID
            cv2.putText(frame, f"ID:{person_id}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
        
        # Update chaos clusters and people
        chaos_clusters = analysis.get('chaos_cluster_count', 0)
        chaos_people = analysis.get('chaos_people_count', 0)
        self.chaos_clusters_var.set(str(chaos_clusters))
        self.chaos_people_var.set(str(chaos_people))
        
        # Update analysis status
        if chaos_status == 'ANALYZING':
            self.analysis_status_var.set("Collecting data...")
        elif chaos_status == 'CHAOS':
            self.analysis_status_var.set("CHAOS DETECTED!")
        else:
            self.analysis_status_var.set("All calm")
        
        # Update activity counts
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
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_processing:
            self.stop_processing()
        
        # Clean up resources
        if self.cap is not None:
            self.cap.release()
        
        # Destroy the window
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
            self.on_closing()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.on_closing()

if __name__ == "__main__":
    app = SimpleVideoProcessor()
    app.run()
