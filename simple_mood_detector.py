import cv2
import numpy as np
from collections import deque
import threading
import time
from yolo_human_detector import YOLOHumanDetector
from chaos_analyzer import ChaosAnalyzer
from enhanced_chaos_analyzer import EnhancedChaosAnalyzer
from advanced_chaos_detector import AdvancedChaosDetector

class SimpleStudentMoodAnalyzer:
    def __init__(self, use_yolo=True):
        # Initialize YOLO human detector
        self.use_yolo = use_yolo
        if use_yolo:
            try:
                print("Initializing YOLO human detector...")
                self.yolo_detector = YOLOHumanDetector(model_size='s', confidence_threshold=0.15)
                print("YOLO detector initialized successfully")
            except Exception as e:
                print(f"Error initializing YOLO detector: {e}")
                print("Falling back to OpenCV face detection...")
                self.use_yolo = False
                self.yolo_detector = None
        else:
            self.yolo_detector = None
        
        # Initialize OpenCV face detection as fallback
        if not self.use_yolo:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
                
                # Check if cascades loaded properly
                if self.face_cascade.empty():
                    print("Warning: Face cascade not loaded properly")
                if self.eye_cascade.empty():
                    print("Warning: Eye cascade not loaded properly")
                if self.smile_cascade.empty():
                    print("Warning: Smile cascade not loaded properly")
                    
            except Exception as e:
                print(f"Error loading cascade classifiers: {e}")
                # Create dummy cascades to prevent crashes
                self.face_cascade = None
                self.eye_cascade = None
                self.smile_cascade = None
        else:
            # Initialize face detection for mood analysis only
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            except Exception as e:
                print(f"Error loading face detection cascades: {e}")
                self.face_cascade = None
                self.eye_cascade = None
                self.smile_cascade = None
        
        # Mood detection based on facial features
        self.mood_history = deque(maxlen=30)  # Keep last 30 frames
        self.people_count_history = deque(maxlen=10)
        
        # Initialize advanced chaos analyzer
        self.chaos_analyzer = ChaosAnalyzer()
        self.enhanced_chaos_analyzer = EnhancedChaosAnalyzer()
        self.advanced_chaos_detector = AdvancedChaosDetector()
        
        # Activity classification parameters
        self.activity_history = deque(maxlen=50)  # Store recent activity classifications
        self.group_work_threshold = 0.2  # Minimum proximity for group work
        self.structured_threshold = 0.4  # Threshold for structured activity
        self.chaos_threshold = 0.6  # Threshold for distractive chaos (18 on 1-30 scale)
        
        # Individual person tracking for chaos detection
        self.person_tracks = {}  # Track individual people and their chaos levels
        self.next_person_id = 0
        
        # Chaos detection parameters
        self.movement_threshold = 0.1
        self.noise_threshold = 30

        # Throttled error logging to avoid terminal spam
        self._last_error_log_time = 0.0
        self._suppressed_error_count = 0
        self._cascade_error_count = 0
        self._cascade_error_window_start = time.time()

    def _record_cascade_error(self, window_seconds: float = 30.0, max_errors: int = 5):
        """Track cascade errors and auto-disable cascades if too frequent."""
        now = time.time()
        if (now - self._cascade_error_window_start) > window_seconds:
            self._cascade_error_window_start = now
            self._cascade_error_count = 0
        self._cascade_error_count += 1
        if self._cascade_error_count >= max_errors:
            # Auto-disable cascades to prevent repeated cv2 errors
            self.face_cascade = None
            self.eye_cascade = None
            self.smile_cascade = None
            print("⚠️ Disabling OpenCV cascades due to repeated errors. Continuing without face-based mood detection.")

    def _log_error_throttled(self, label: str, error: Exception, min_interval_seconds: float = 5.0):
        """Log errors at most once per min_interval_seconds; count suppressed ones."""
        now = time.time()
        if (now - self._last_error_log_time) >= min_interval_seconds:
            message = f"{label}: {error}"
            if self._suppressed_error_count > 0:
                message += f" (and {self._suppressed_error_count} more similar errors suppressed)"
            print(message)
            self._last_error_log_time = now
            self._suppressed_error_count = 0
        else:
            self._suppressed_error_count += 1
        
    def detect_faces(self, frame, frame_count=0):
        """Detect faces in the frame"""
        try:
            if self.face_cascade is None or self.face_cascade.empty():
                print("Warning: Face cascade not available")
                return [], None
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try multiple detection parameters for better results
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # More sensitive
                minNeighbors=3,   # Less strict
                minSize=(30, 30), # Minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Debug information (minimal frequency for speed)
            if len(faces) > 0 and frame_count % 50 == 0:  # Only print every 50th frame
                print(f"Detected {len(faces)} faces")
            
            return faces, gray
        except Exception as e:
            self._log_error_throttled("Error in face detection", e)
            self._record_cascade_error()
            return [], None
    
    def detect_mood_from_face(self, face_roi, gray_roi):
        """Detect mood based on facial features in the face region"""
        try:
            if gray_roi is None or self.eye_cascade is None or self.smile_cascade is None:
                return "unknown"
            
            # Detect eyes
            try:
                eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 3) if not self.eye_cascade.empty() else []
            except Exception as e:
                self._log_error_throttled("Error in eye detect", e)
                self._record_cascade_error()
                eyes = []
            
            # Detect smiles
            try:
                smiles = self.smile_cascade.detectMultiScale(gray_roi, 1.8, 20) if not self.smile_cascade.empty() else []
            except Exception as e:
                self._log_error_throttled("Error in smile detect", e)
                self._record_cascade_error()
                smiles = []
            
            # Simple mood detection based on detected features
            if len(smiles) > 0:
                return "happy"
            elif len(eyes) >= 2:
                # Check if eyes are wide open (excited) or normal (calm)
                eye_areas = [w * h for (x, y, w, h) in eyes]
                avg_eye_area = sum(eye_areas) / len(eye_areas) if eye_areas else 0
                
                if avg_eye_area > 1000:  # Wide eyes - excited
                    return "excited"
                else:
                    return "calm"
            else:
                return "neutral"
        except Exception as e:
            self._log_error_throttled("Error in mood detection", e)
            return "unknown"
    
    def count_people(self, frame, frame_count=0):
        """Count number of people in the frame"""
        faces, _ = self.detect_faces(frame, frame_count)
        return len(faces)
    
    def calculate_distance(self, rect1, rect2):
        """Calculate distance between two rectangles"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate center points
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance
    
    def track_people(self, faces, frame, prev_frame):
        """Track individual people and calculate their chaos levels"""
        current_people = []
        chaos_people = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Calculate movement for this person
            person_chaos = self.calculate_person_chaos(frame, prev_frame, (x, y, w, h))
            
            # Try to match with existing tracks
            person_id = self.match_or_create_track((x, y, w, h), person_chaos)
            
            current_people.append({
                'id': person_id,
                'rect': (x, y, w, h),
                'chaos_level': person_chaos,
                'is_chaotic': person_chaos > self.chaos_threshold
            })
            
            if person_chaos > self.chaos_threshold:
                chaos_people.append({
                    'id': person_id,
                    'rect': (x, y, w, h),
                    'chaos_level': person_chaos
                })
        
        # Update person tracks
        self.update_person_tracks(current_people)
        
        return current_people, chaos_people
    
    def calculate_person_chaos(self, frame, prev_frame, person_rect):
        """Calculate chaos level for a specific person"""
        if prev_frame is None:
            return 0
        
        x, y, w, h = person_rect
        
        # Extract person region from both frames
        person_region = frame[y:y+h, x:x+w]
        prev_person_region = prev_frame[y:y+h, x:x+w]
        
        if person_region.size == 0 or prev_person_region.size == 0:
            return 0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(prev_person_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate movement in this person's region
        diff = cv2.absdiff(gray1, gray2)
        movement = np.mean(diff)
        
        # Calculate noise level
        noise = np.std(diff)
        
        # Calculate edge changes (more sensitive to movement)
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        edge_diff = cv2.absdiff(edges1, edges2)
        edge_movement = np.mean(edge_diff)
        
        # More sensitive chaos calculation
        chaos_score = (movement * 25 + noise * 3 + edge_movement * 10) / 100
        
        return min(chaos_score, 1.0)
    
    def match_or_create_track(self, face_rect, chaos_level):
        """Match face to existing track or create new one"""
        best_match_id = None
        best_distance = float('inf')
        
        for person_id, track in self.person_tracks.items():
            if track['last_seen'] < 5:  # Only consider recently seen people
                distance = self.calculate_distance(face_rect, track['last_rect'])
                if distance < best_distance and distance < 100:  # Max distance threshold
                    best_distance = distance
                    best_match_id = person_id
        
        if best_match_id is not None:
            # Update existing track
            self.person_tracks[best_match_id]['last_rect'] = face_rect
            self.person_tracks[best_match_id]['last_seen'] = 0
            if 'chaos_levels' not in self.person_tracks[best_match_id]:
                self.person_tracks[best_match_id]['chaos_levels'] = deque(maxlen=10)
            self.person_tracks[best_match_id]['chaos_levels'].append(chaos_level)
            return best_match_id
        else:
            # Create new track
            person_id = self.next_person_id
            self.next_person_id += 1
            self.person_tracks[person_id] = {
                'last_rect': face_rect,
                'last_seen': 0,
                'chaos_levels': deque([chaos_level], maxlen=10)
            }
            return person_id
    
    def update_person_tracks(self, current_people):
        """Update person tracks and remove old ones"""
        # Increment last_seen for all tracks
        for person_id in self.person_tracks:
            self.person_tracks[person_id]['last_seen'] += 1
        
        # Remove tracks that haven't been seen for too long
        to_remove = [pid for pid, track in self.person_tracks.items() if track['last_seen'] > 10]
        for pid in to_remove:
            del self.person_tracks[pid]
    
    def detect_chaos_level(self, frame, prev_frame):
        """Detect chaos level based on movement and visual changes"""
        if prev_frame is None:
            return 0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        movement = np.mean(diff)
        
        # Calculate noise level
        noise = np.std(diff)
        
        # Combine movement and noise for chaos score
        chaos_score = (movement * 10 + noise) / 100
        
        return min(chaos_score, 1.0)  # Normalize to 0-1
    
    def analyze_frame(self, frame, prev_frame=None, frame_count=0):
        """Main analysis function for each frame"""
        try:
            if self.use_yolo and self.yolo_detector is not None:
                # Use YOLO for human detection
                detections, detection_time = self.yolo_detector.detect_humans(frame)
                people_count = len(detections)
                self.people_count_history.append(people_count)
                
                # Track people using YOLO
                current_people, _ = self.yolo_detector.track_humans(detections, frame, prev_frame)
                
                # Debug: Print YOLO tracking info
                if frame_count % 100 == 0:  # Print every 100 frames
                    print(f"🔍 YOLO Tracking Debug - Frame {frame_count}:")
                    print(f"   YOLO detections: {len(detections)}")
                    print(f"   Tracked people: {len(current_people)}")
                    if len(detections) > 0 and len(current_people) == 0:
                        print("   ⚠️ WARNING: YOLO detected people but tracking returned 0 people!")
                
                # Fallback: If tracking failed but we have detections, create people manually
                if len(detections) > 0 and len(current_people) == 0:
                    print(f"🔧 Creating fallback people from {len(detections)} detections")
                    current_people = []
                    for i, detection in enumerate(detections):
                        person_data = {
                            'id': i,  # Simple ID assignment
                            'rect': detection['bbox'],
                            'confidence': detection['confidence'],
                            'center': detection['center'],
                            'chaos_level': 0.0,
                            'is_chaotic': False
                        }
                        current_people.append(person_data)
                
                # Use advanced chaos detector for real chaos detection
                current_people, clusters = self.advanced_chaos_detector.analyze_advanced_chaos(current_people, prev_frame, frame)
                
                # Get chaos report
                chaos_report = self.advanced_chaos_detector.get_chaos_report()
                
                # Also use basic chaos analyzer for compatibility
                current_people, chaos_people = self.chaos_analyzer.analyze_frame_chaos(current_people, prev_frame, frame)
                
                # Convert YOLO detections to face-like format for mood analysis
                faces = [det['bbox'] for det in detections]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            else:
                # Fallback to OpenCV face detection
                faces, gray = self.detect_faces(frame, frame_count)
                people_count = len(faces)
                self.people_count_history.append(people_count)
                
                # Track people and detect chaos creators using OpenCV
                current_people, chaos_people = self.track_people(faces, frame, prev_frame)
            
            # Analyze moods for detected faces
            moods = []
            if gray is not None and len(faces) > 0:
                for (x, y, w, h) in faces:
                    try:
                        # Extract face region
                        face_roi = frame[y:y+h, x:x+w]
                        gray_roi = gray[y:y+h, x:x+w]
                        
                        # Detect mood for this face
                        mood = self.detect_mood_from_face(face_roi, gray_roi)
                        moods.append(mood)
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        moods.append("unknown")
            
            # Get overall chaos summary from the advanced analyzer
            chaos_summary = self.chaos_analyzer.get_overall_chaos_summary()
            
            # Get advanced chaos report
            advanced_chaos_report = self.advanced_chaos_detector.get_chaos_report()
            
            # Get enhanced activity summary
            activity_summary = self.enhanced_chaos_analyzer.get_activity_summary()
            
            # Ensure all required keys exist in advanced_chaos_report
            if 'chaos_people_count' not in advanced_chaos_report:
                advanced_chaos_report['chaos_people_count'] = 0
            if 'chaos_cluster_count' not in advanced_chaos_report:
                advanced_chaos_report['chaos_cluster_count'] = 0
            
            # Classify activity type
            activity_type = self.classify_activity_type(current_people, chaos_summary['overall_chaos'])
            self.activity_history.append(activity_type)
            
            # Store mood data
            self.mood_history.append({
                'moods': moods,
                'people_count': people_count,
                'chaos_level': chaos_summary['overall_chaos'],
                'current_people': current_people,
                'chaos_people': chaos_people,
                'individual_activities': individual_activities if 'individual_activities' in locals() else {},
                'cluster_activities': cluster_activities if 'cluster_activities' in locals() else {},
                'clusters': clusters if 'clusters' in locals() else [],
                'activity_type': activity_type,
                'timestamp': time.time()
            })
            
            return {
                'moods': moods,
                'people_count': people_count,
                'chaos_level': chaos_summary['overall_chaos'],
                'chaos_level_name': chaos_summary['chaos_level'],
                'current_people': current_people,
                'chaos_people': chaos_people,
                'chaos_creators_count': chaos_summary['chaos_creators_count'],
                'chaos_percentage': chaos_summary['chaos_percentage'],
                'individual_work_count': activity_summary['individual_work'],
                'structured_group_work_count': activity_summary['structured_group_work'],
                'structured_chaos_count': activity_summary['structured_chaos'],
                'individual_chaos_count': activity_summary['individual_chaos'],
                'calm_count': activity_summary['calm'],
                'active_clusters': activity_summary['active_clusters'],
                'individual_activities': individual_activities if 'individual_activities' in locals() else {},
                'cluster_activities': cluster_activities if 'cluster_activities' in locals() else {},
                # Advanced chaos detection results
                'overall_chaos_status': advanced_chaos_report['overall_status'],
                'chaos_clusters': advanced_chaos_report['chaos_clusters'],
                'calm_clusters': advanced_chaos_report['calm_clusters'],
                'individual_chaos_people': advanced_chaos_report['individual_chaos'],
                'individual_calm_people': advanced_chaos_report['individual_calm'],
                'chaos_people_count': advanced_chaos_report['chaos_people_count'],
                'chaos_cluster_count': advanced_chaos_report['chaos_cluster_count'],
                'clusters': clusters if 'clusters' in locals() else [],
                'activity_type': activity_type,
                'activity_summary': self.get_activity_summary(),
                'dominant_mood': self.get_dominant_mood(),
                'average_people': self.get_average_people_count(),
                'overall_chaos': chaos_summary['overall_chaos']
            }
        except Exception as e:
            self._log_error_throttled("Error in analyze_frame", e)
            # Return safe default values
            return {
                'moods': [],
                'people_count': 0,
                'chaos_level': 0,
                'current_people': [],
                'chaos_people': [],
                'dominant_mood': 'unknown',
                'average_people': 0,
                'overall_chaos': 0
            }
    
    def get_dominant_mood(self):
        """Get the most common mood from recent frames"""
        if not self.mood_history:
            return "unknown"
        
        all_moods = []
        for frame_data in self.mood_history:
            all_moods.extend(frame_data['moods'])
        
        if not all_moods:
            return "unknown"
        
        from collections import Counter
        mood_counts = Counter(all_moods)
        return mood_counts.most_common(1)[0][0]
    
    def get_average_people_count(self):
        """Get average people count from recent frames"""
        if not self.people_count_history:
            return 0
        return sum(self.people_count_history) / len(self.people_count_history)
    
    def set_chaos_threshold_scale(self, scale_value: int):
        """Set chaos threshold from a 1-30 scale (e.g., 16-18 is where chaos is detected).
        Maps 1..30 -> 0..1 used internally.
        """
        try:
            value = int(scale_value)
        except Exception:
            value = 18
        value = max(1, min(30, value))
        self.chaos_threshold = value / 30.0
        return self.chaos_threshold

    def get_overall_chaos_level(self):
        """Get overall chaos level from recent frames"""
        if not self.mood_history:
            return 0
        
        recent_chaos = [frame_data['chaos_level'] for frame_data in list(self.mood_history)[-10:]]
        return sum(recent_chaos) / len(recent_chaos) if recent_chaos else 0
    
    def reset_session(self):
        """Reset all session data for fresh video processing"""
        # Clear all history data
        self.mood_history.clear()
        self.people_count_history.clear()
        self.activity_history.clear()
        
        # Reset tracking data
        self.person_tracks = {}
        self.next_person_id = 0
        
        # Reset chaos analyzers if they have reset methods
        if hasattr(self, 'chaos_analyzer') and hasattr(self.chaos_analyzer, 'reset_session'):
            self.chaos_analyzer.reset_session()
        
        if hasattr(self, 'enhanced_chaos_analyzer') and hasattr(self.enhanced_chaos_analyzer, 'reset_session'):
            self.enhanced_chaos_analyzer.reset_session()
        
        if hasattr(self, 'advanced_chaos_detector') and hasattr(self.advanced_chaos_detector, 'reset_session'):
            self.advanced_chaos_detector.reset_session()
        
        print("🔄 Analyzer session data reset for fresh start")
    
    def classify_activity_type(self, people_data, chaos_level):
        """Classify the type of classroom activity based on people positions and chaos level"""
        if not people_data:
            return "no_activity"
        
        # Calculate group work indicators
        group_work_score = self._calculate_group_work_score(people_data)
        
        # Calculate structured activity indicators
        structured_score = self._calculate_structured_score(people_data, chaos_level)
        
        # Determine activity type
        if chaos_level > self.chaos_threshold:
            if group_work_score > 0.5:
                return "distractive_group_chaos"
            else:
                return "distractive_individual_chaos"
        elif group_work_score > self.group_work_threshold:
            if structured_score > self.structured_threshold:
                return "structured_group_work"
            else:
                return "unstructured_group_work"
        else:
            if structured_score > self.structured_threshold:
                return "structured_individual_work"
            else:
                return "unstructured_individual_work"
    
    def _calculate_group_work_score(self, people_data):
        """Calculate how likely people are working in groups based on proximity"""
        if len(people_data) < 2:
            return 0.0
        
        group_score = 0.0
        total_pairs = 0
        
        for i, person1 in enumerate(people_data):
            for j, person2 in enumerate(people_data[i+1:], i+1):
                # Calculate distance between people
                x1, y1, w1, h1 = person1['rect']
                x2, y2, w2, h2 = person2['rect']
                
                # Calculate center points
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Normalize distance (assuming frame is roughly 640x480)
                normalized_distance = distance / 500.0
                
                # Closer people = higher group work score
                if normalized_distance < 0.2:  # Very close
                    group_score += 1.0
                elif normalized_distance < 0.4:  # Close
                    group_score += 0.7
                elif normalized_distance < 0.6:  # Moderate
                    group_score += 0.3
                
                total_pairs += 1
        
        return group_score / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_structured_score(self, people_data, chaos_level):
        """Calculate how structured the activity appears to be"""
        if not people_data:
            return 0.0
        
        # Factors that indicate structured activity:
        # 1. Low chaos level
        # 2. People are relatively stationary (low movement)
        # 3. People are facing similar directions (if we can detect orientation)
        # 4. People are distributed evenly in space
        
        structured_score = 0.0
        
        # Chaos level factor (lower chaos = more structured)
        chaos_factor = 1.0 - min(chaos_level, 1.0)
        structured_score += chaos_factor * 0.4
        
        # Movement factor (less movement = more structured)
        movement_scores = []
        for person in people_data:
            if 'last_rect' in person and 'last_seen' in person:
                # Calculate movement since last frame
                current_rect = person['rect']
                last_rect = person['last_rect']
                
                # Calculate center movement
                current_center = (current_rect[0] + current_rect[2]/2, current_rect[1] + current_rect[3]/2)
                last_center = (last_rect[0] + last_rect[2]/2, last_rect[1] + last_rect[3]/2)
                
                movement = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
                movement_scores.append(movement)
        
        if movement_scores:
            avg_movement = np.mean(movement_scores)
            movement_factor = max(0, 1.0 - (avg_movement / 50.0))  # Normalize movement
            structured_score += movement_factor * 0.3
        
        # Distribution factor (even distribution = more structured)
        if len(people_data) > 1:
            # Calculate how evenly distributed people are
            x_positions = [p['rect'][0] + p['rect'][2]/2 for p in people_data]
            y_positions = [p['rect'][1] + p['rect'][3]/2 for p in people_data]
            
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            # Moderate variance indicates good distribution
            distribution_factor = min(1.0, (x_variance + y_variance) / 10000.0)
            structured_score += distribution_factor * 0.3
        
        return min(structured_score, 1.0)
    
    def get_activity_summary(self):
        """Get a summary of recent activity types"""
        if not self.activity_history:
            return {
                'structured_group_work': 0,
                'unstructured_group_work': 0,
                'structured_individual_work': 0,
                'unstructured_individual_work': 0,
                'distractive_group_chaos': 0,
                'distractive_individual_chaos': 0,
                'no_activity': 0,
                'total_periods': 0
            }
        
        activity_counts = {}
        for activity in self.activity_history:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        total_periods = len(self.activity_history)
        
        # Calculate percentages
        summary = {}
        for activity_type in ['structured_group_work', 'unstructured_group_work', 
                             'structured_individual_work', 'unstructured_individual_work',
                             'distractive_group_chaos', 'distractive_individual_chaos', 'no_activity']:
            count = activity_counts.get(activity_type, 0)
            summary[activity_type] = count
            summary[f'{activity_type}_percentage'] = (count / total_periods) * 100 if total_periods > 0 else 0
        
        summary['total_periods'] = total_periods
        return summary