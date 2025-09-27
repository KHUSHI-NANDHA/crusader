import cv2
import numpy as np
from collections import deque
import time

class ChaosAnalyzer:
    """
    Advanced chaos detection system with proper scales and thresholds.
    Distinguishes between normal movement and actual disruptive behavior.
    """
    
    def __init__(self):
        # Chaos scale levels (0-100)
        self.CHAOS_LEVELS = {
            'CALM': (0, 20),           # 0-20: Normal classroom activity
            'MODERATE': (20, 40),      # 20-40: Some movement, but controlled
            'ACTIVE': (40, 60),        # 40-60: More movement, getting restless
            'DISRUPTIVE': (60, 80),    # 60-80: Clear disruptive behavior
            'CHAOTIC': (80, 100)       # 80-100: Complete chaos
        }
        
        # Movement classification thresholds
        self.MOVEMENT_THRESHOLDS = {
            'MINIMAL': 0.05,      # Very small movements (breathing, small gestures)
            'LIGHT': 0.15,        # Light movements (hand gestures, note-taking)
            'MODERATE': 0.30,     # Moderate movements (leaning, shifting)
            'HEAVY': 0.50,        # Heavy movements (standing, walking)
            'EXTREME': 0.70       # Extreme movements (running, jumping)
        }
        
        # Chaos detection criteria
        self.CHAOS_CRITERIA = {
            'movement_intensity': 0.4,      # Minimum movement intensity for chaos
            'movement_duration': 2.0,       # Minimum duration in seconds
            'movement_frequency': 0.3,      # Minimum frequency of movements
            'spatial_spread': 0.2,          # How spread out the movements are
            'acceleration_threshold': 0.1,  # Sudden acceleration changes
            'group_activity': 0.3          # Multiple people moving together
        }
        
        # Tracking variables
        self.person_histories = {}  # Track each person's movement history
        self.frame_history = deque(maxlen=30)  # Last 30 frames for analysis
        self.chaos_events = deque(maxlen=100)  # Recent chaos events
        
        # Analysis parameters
        self.analysis_window = 30  # Frames to analyze for chaos detection
        self.min_chaos_duration = 1.0  # Minimum seconds for chaos classification
        
    def calculate_movement_intensity(self, current_frame, previous_frame, bbox):
        """
        Calculate movement intensity for a specific person.
        Returns a value between 0 and 1.
        """
        if previous_frame is None:
            return 0.0
        
        x, y, w, h = bbox
        
        # Ensure bounding box is within frame bounds
        x = max(0, min(x, current_frame.shape[1] - w))
        y = max(0, min(y, current_frame.shape[0] - h))
        w = min(w, current_frame.shape[1] - x)
        h = min(h, current_frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 0.0
        
        try:
            # Extract regions
            current_region = current_frame[y:y+h, x:x+w]
            prev_region = previous_frame[y:y+h, x:x+w]
            
            if current_region.size == 0 or prev_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray_current = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(gray_current, gray_prev)
            
            # Calculate multiple movement metrics
            mean_movement = np.mean(diff)
            std_movement = np.std(diff)
            
            # Calculate optical flow for more accurate movement detection
            try:
                # Use Lucas-Kanade optical flow
                p0 = np.array([[[x + w//2, y + h//2]]], dtype=np.float32)
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_current, p0, None)
                
                if st[0][0] == 1:  # If tracking was successful
                    flow_magnitude = np.sqrt(np.sum((p1[0][0] - p0[0][0])**2))
                else:
                    flow_magnitude = 0
            except:
                flow_magnitude = 0
            
            # Calculate edge changes (more sensitive to movement)
            edges_current = cv2.Canny(gray_current, 50, 150)
            edges_prev = cv2.Canny(gray_prev, 50, 150)
            edge_diff = cv2.absdiff(edges_current, edges_prev)
            edge_movement = np.mean(edge_diff)
            
            # Combine metrics for comprehensive movement intensity
            intensity = (
                mean_movement * 0.3 +           # Basic pixel difference
                std_movement * 0.2 +            # Movement variability
                flow_magnitude * 0.3 +          # Optical flow magnitude
                edge_movement * 0.2             # Edge changes
            ) / 255.0  # Normalize to 0-1
            
            return min(intensity, 1.0)
            
        except Exception as e:
            print(f"Error calculating movement intensity: {e}")
            return 0.0
    
    def classify_movement_type(self, intensity):
        """Classify movement based on intensity."""
        if intensity < self.MOVEMENT_THRESHOLDS['MINIMAL']:
            return 'STATIONARY'
        elif intensity < self.MOVEMENT_THRESHOLDS['LIGHT']:
            return 'MINIMAL'
        elif intensity < self.MOVEMENT_THRESHOLDS['MODERATE']:
            return 'LIGHT'
        elif intensity < self.MOVEMENT_THRESHOLDS['HEAVY']:
            return 'MODERATE'
        elif intensity < self.MOVEMENT_THRESHOLDS['EXTREME']:
            return 'HEAVY'
        else:
            return 'EXTREME'
    
    def calculate_chaos_level(self, person_data, frame_data):
        """
        Calculate chaos level for a person based on multiple criteria.
        Returns a chaos score from 0-100.
        """
        person_id = person_data['id']
        movement_intensity = person_data.get('movement_intensity', 0)
        
        # Initialize person history if not exists
        if person_id not in self.person_histories:
            self.person_histories[person_id] = {
                'movement_history': deque(maxlen=self.analysis_window),
                'position_history': deque(maxlen=self.analysis_window),
                'intensity_history': deque(maxlen=self.analysis_window),
                'last_chaos_time': 0,
                'chaos_duration': 0
            }
        
        person_history = self.person_histories[person_id]
        
        # Update history
        person_history['movement_history'].append(movement_intensity)
        person_history['position_history'].append(person_data['center'])
        person_history['intensity_history'].append(movement_intensity)
        
        # Calculate chaos factors
        chaos_factors = {}
        
        # 1. Movement Intensity Factor (0-30 points)
        avg_intensity = np.mean(list(person_history['intensity_history'])) if person_history['intensity_history'] else 0
        if avg_intensity > self.CHAOS_CRITERIA['movement_intensity']:
            chaos_factors['intensity'] = min(30, avg_intensity * 60)
        else:
            chaos_factors['intensity'] = 0
        
        # 2. Movement Duration Factor (0-20 points)
        current_time = time.time()
        if avg_intensity > self.CHAOS_CRITERIA['movement_intensity']:
            if person_history['last_chaos_time'] == 0:
                person_history['last_chaos_time'] = current_time
            person_history['chaos_duration'] = current_time - person_history['last_chaos_time']
            
            if person_history['chaos_duration'] > self.CHAOS_CRITERIA['movement_duration']:
                chaos_factors['duration'] = min(20, person_history['chaos_duration'] * 10)
            else:
                chaos_factors['duration'] = 0
        else:
            person_history['last_chaos_time'] = 0
            person_history['chaos_duration'] = 0
            chaos_factors['duration'] = 0
        
        # 3. Movement Frequency Factor (0-15 points)
        recent_movements = list(person_history['movement_history'])[-10:]  # Last 10 frames
        high_movement_count = sum(1 for m in recent_movements if m > self.CHAOS_CRITERIA['movement_intensity'])
        movement_frequency = high_movement_count / len(recent_movements) if recent_movements else 0
        
        if movement_frequency > self.CHAOS_CRITERIA['movement_frequency']:
            chaos_factors['frequency'] = min(15, movement_frequency * 30)
        else:
            chaos_factors['frequency'] = 0
        
        # 4. Spatial Spread Factor (0-15 points)
        if len(person_history['position_history']) > 5:
            positions = np.array(list(person_history['position_history']))
            if len(positions) > 1:
                position_variance = np.var(positions, axis=0)
                spatial_spread = np.sqrt(np.sum(position_variance))
                
                if spatial_spread > self.CHAOS_CRITERIA['spatial_spread'] * 100:  # Scale factor
                    chaos_factors['spatial'] = min(15, spatial_spread / 10)
                else:
                    chaos_factors['spatial'] = 0
            else:
                chaos_factors['spatial'] = 0
        else:
            chaos_factors['spatial'] = 0
        
        # 5. Acceleration Factor (0-10 points)
        if len(person_history['intensity_history']) > 3:
            intensities = list(person_history['intensity_history'])[-5:]
            if len(intensities) > 2:
                acceleration = np.diff(intensities)
                max_acceleration = np.max(np.abs(acceleration))
                
                if max_acceleration > self.CHAOS_CRITERIA['acceleration_threshold']:
                    chaos_factors['acceleration'] = min(10, max_acceleration * 20)
                else:
                    chaos_factors['acceleration'] = 0
            else:
                chaos_factors['acceleration'] = 0
        else:
            chaos_factors['acceleration'] = 0
        
        # 6. Group Activity Factor (0-10 points)
        current_people = frame_data.get('current_people', [])
        if len(current_people) > 1:
            other_chaotic_people = sum(1 for p in current_people 
                                     if p['id'] != person_id and p.get('chaos_level', 0) > 30)
            group_activity = other_chaotic_people / (len(current_people) - 1)
            
            if group_activity > self.CHAOS_CRITERIA['group_activity']:
                chaos_factors['group'] = min(10, group_activity * 20)
            else:
                chaos_factors['group'] = 0
        else:
            chaos_factors['group'] = 0
        
        # Calculate total chaos score
        total_chaos = sum(chaos_factors.values())
        
        # Apply minimum thresholds
        if total_chaos < 20:  # Minimum threshold for chaos
            total_chaos = 0
        
        return min(total_chaos, 100), chaos_factors
    
    def get_chaos_level_name(self, chaos_score):
        """Get the name of the chaos level based on score."""
        for level_name, (min_score, max_score) in self.CHAOS_LEVELS.items():
            if min_score <= chaos_score < max_score:
                return level_name
        return 'CHAOTIC' if chaos_score >= 100 else 'CALM'
    
    def is_creating_chaos(self, chaos_score):
        """Determine if a person is creating chaos based on score."""
        return chaos_score >= 60  # DISRUPTIVE level and above
    
    def analyze_frame_chaos(self, current_people, prev_frame, current_frame):
        """
        Analyze chaos for all people in the current frame.
        Returns updated people data with chaos levels.
        """
        frame_data = {
            'current_people': current_people,
            'timestamp': time.time()
        }
        
        updated_people = []
        chaos_creators = []
        
        for person in current_people:
            # Calculate movement intensity
            movement_intensity = self.calculate_movement_intensity(
                current_frame, prev_frame, person['rect']
            )
            
            # Update person data
            person['movement_intensity'] = movement_intensity
            person['movement_type'] = self.classify_movement_type(movement_intensity)
            
            # Calculate chaos level
            chaos_score, chaos_factors = self.calculate_chaos_level(person, frame_data)
            person['chaos_level'] = chaos_score
            person['chaos_factors'] = chaos_factors
            person['chaos_level_name'] = self.get_chaos_level_name(chaos_score)
            person['is_chaotic'] = self.is_creating_chaos(chaos_score)
            
            updated_people.append(person)
            
            if person['is_chaotic']:
                chaos_creators.append(person)
        
        # Store frame data for analysis
        self.frame_history.append({
            'people': updated_people,
            'chaos_creators': chaos_creators,
            'timestamp': time.time()
        })
        
        return updated_people, chaos_creators
    
    def get_overall_chaos_summary(self):
        """Get overall chaos summary for the classroom."""
        if not self.frame_history:
            return {
                'overall_chaos': 0,
                'chaos_level': 'CALM',
                'chaos_creators_count': 0,
                'total_people': 0,
                'chaos_percentage': 0
            }
        
        recent_frames = list(self.frame_history)[-10:]  # Last 10 frames
        
        total_people = 0
        chaotic_people = 0
        total_chaos_score = 0
        
        for frame_data in recent_frames:
            people = frame_data['people']
            total_people += len(people)
            chaotic_people += len(frame_data['chaos_creators'])
            
            for person in people:
                total_chaos_score += person.get('chaos_level', 0)
        
        if total_people > 0:
            avg_chaos = total_chaos_score / total_people
            chaos_percentage = (chaotic_people / total_people) * 100
        else:
            avg_chaos = 0
            chaos_percentage = 0
        
        return {
            'overall_chaos': avg_chaos,
            'chaos_level': self.get_chaos_level_name(avg_chaos),
            'chaos_creators_count': chaotic_people,
            'total_people': total_people,
            'chaos_percentage': chaos_percentage
        }


