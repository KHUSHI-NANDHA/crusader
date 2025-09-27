import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import threading
import time

class StudentMoodAnalyzer:
    def __init__(self):
        # Initialize MediaPipe face detection and facial landmarks
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, 
                                                   max_num_faces=10, 
                                                   min_detection_confidence=0.5)
        
        # Mood detection based on facial landmarks
        self.mood_history = deque(maxlen=30)  # Keep last 30 frames
        self.people_count_history = deque(maxlen=10)
        
        # Chaos detection parameters
        self.movement_threshold = 0.1
        self.noise_threshold = 30
        
    def detect_mood_from_landmarks(self, landmarks):
        """Detect mood based on facial landmark positions"""
        if not landmarks:
            return "unknown"
        
        # Extract key facial points
        left_eye = landmarks[33]
        right_eye = landmarks[362]
        nose_tip = landmarks[1]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_center = landmarks[13]
        
        # Calculate facial ratios for mood detection
        eye_distance = abs(left_eye.x - right_eye.x)
        mouth_width = abs(mouth_left.x - mouth_right.x)
        mouth_height = abs(mouth_center.y - (mouth_left.y + mouth_right.y) / 2)
        
        # Simple mood detection based on facial geometry
        if mouth_height > 0.02:  # Open mouth - excited/happy
            return "excited"
        elif mouth_width > eye_distance * 0.6:  # Wide mouth - happy
            return "happy"
        elif mouth_height < 0.005 and mouth_width < eye_distance * 0.4:  # Small mouth - calm
            return "calm"
        elif mouth_center.y > nose_tip.y + 0.02:  # Frown - sad/angry
            return "sad"
        else:
            return "neutral"
    
    def count_people(self, frame):
        """Count number of people in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            return len(results.detections)
        return 0
    
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
    
    def analyze_frame(self, frame, prev_frame=None):
        """Main analysis function for each frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Count people
        people_count = self.count_people(frame)
        self.people_count_history.append(people_count)
        
        # Detect faces and get landmarks
        face_results = self.face_mesh.process(rgb_frame)
        moods = []
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Convert landmarks to list of (x, y) coordinates
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                mood = self.detect_mood_from_landmarks(landmarks)
                moods.append(mood)
        
        # Detect chaos level
        chaos_level = self.detect_chaos_level(frame, prev_frame)
        
        # Store mood data
        self.mood_history.append({
            'moods': moods,
            'people_count': people_count,
            'chaos_level': chaos_level,
            'timestamp': time.time()
        })
        
        return {
            'moods': moods,
            'people_count': people_count,
            'chaos_level': chaos_level,
            'dominant_mood': self.get_dominant_mood(),
            'average_people': self.get_average_people_count(),
            'overall_chaos': self.get_overall_chaos_level()
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
