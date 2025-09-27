import cv2
import numpy as np
from collections import deque
import threading
import time
from yolo_human_detector import YOLOHumanDetector

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
        
        # Individual person tracking for chaos detection
        self.person_tracks = {}  # Track individual people and their chaos levels
        self.next_person_id = 0
        self.chaos_threshold = 0.1  # Lower threshold for better detection
        
        # Chaos detection parameters
        self.movement_threshold = 0.1
        self.noise_threshold = 30
        
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
            print(f"Error in face detection: {e}")
            return [], None
    
    def detect_mood_from_face(self, face_roi, gray_roi):
        """Detect mood based on facial features in the face region"""
        try:
            if gray_roi is None or self.eye_cascade is None or self.smile_cascade is None:
                return "unknown"
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 3) if not self.eye_cascade.empty() else []
            
            # Detect smiles
            smiles = self.smile_cascade.detectMultiScale(gray_roi, 1.8, 20) if not self.smile_cascade.empty() else []
            
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
            print(f"Error in mood detection: {e}")
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
            self.person_tracks[best_match_id]['chaos_history'].append(chaos_level)
            return best_match_id
        else:
            # Create new track
            person_id = self.next_person_id
            self.next_person_id += 1
            self.person_tracks[person_id] = {
                'last_rect': face_rect,
                'last_seen': 0,
                'chaos_history': deque([chaos_level], maxlen=10)
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
                
                # Track people and detect chaos creators using YOLO
                current_people, chaos_people = self.yolo_detector.track_humans(detections, frame, prev_frame)
                
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
            
            # Detect overall chaos level
            chaos_level = self.detect_chaos_level(frame, prev_frame)
            
            # Store mood data
            self.mood_history.append({
                'moods': moods,
                'people_count': people_count,
                'chaos_level': chaos_level,
                'current_people': current_people,
                'chaos_people': chaos_people,
                'timestamp': time.time()
            })
            
            return {
                'moods': moods,
                'people_count': people_count,
                'chaos_level': chaos_level,
                'current_people': current_people,
                'chaos_people': chaos_people,
                'dominant_mood': self.get_dominant_mood(),
                'average_people': self.get_average_people_count(),
                'overall_chaos': self.get_overall_chaos_level()
            }
        except Exception as e:
            print(f"Error in analyze_frame: {e}")
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
    
    def get_overall_chaos_level(self):
        """Get overall chaos level from recent frames"""
        if not self.mood_history:
            return 0
        
        recent_chaos = [frame_data['chaos_level'] for frame_data in list(self.mood_history)[-10:]]
        return sum(recent_chaos) / len(recent_chaos) if recent_chaos else 0
