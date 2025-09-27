import cv2
import numpy as np
from collections import deque
import time
from scipy.spatial.distance import pdist, squareform

class AdvancedChaosDetector:
    """
    Advanced chaos detection that analyzes behavior over time to detect real chaos,
    not just motion. Uses 1-minute analysis windows and movement frequency thresholds.
    """
    
    def __init__(self):
        # Analysis parameters
        self.ANALYSIS_WINDOW = 60  # 1 minute in seconds
        self.FRAME_RATE = 25  # Expected FPS
        self.MIN_CHAOS_FRAMES = 15  # Minimum frames of movement to consider chaos
        self.MOVEMENT_THRESHOLD = 0.1  # Minimum movement intensity
        
        # Chaos detection criteria
        self.CHAOS_CRITERIA = {
            'movement_frequency': 0.6,  # 60% of frames with movement
            'movement_intensity': 0.3,  # Average movement intensity
            'duration_threshold': 5.0,  # Minimum duration in seconds
            'group_chaos_threshold': 0.4,  # Group chaos threshold
            'individual_chaos_threshold': 0.7  # Individual chaos threshold
        }
        
        # Tracking variables
        self.person_histories = {}  # Track each person's behavior over time
        self.cluster_histories = {}  # Track cluster behavior
        self.analysis_timeline = deque(maxlen=self.ANALYSIS_WINDOW * self.FRAME_RATE)
        self.last_analysis_time = 0
        self.analysis_interval = 60  # Analyze every 60 seconds
        
        # Chaos detection results
        self.chaos_clusters = []
        self.calm_clusters = []
        self.individual_chaos = []
        self.individual_calm = []
        
    def calculate_movement_intensity(self, current_frame, previous_frame, bbox):
        """Calculate movement intensity for a person."""
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
            
            # Calculate movement metrics
            mean_movement = np.mean(diff)
            std_movement = np.std(diff)
            
            # Calculate optical flow for more accurate movement
            try:
                p0 = np.array([[[x + w//2, y + h//2]]], dtype=np.float32)
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_current, p0, None)
                
                if st[0][0] == 1:
                    flow_magnitude = np.sqrt(np.sum((p1[0][0] - p0[0][0])**2))
                else:
                    flow_magnitude = 0
            except:
                flow_magnitude = 0
            
            # Combine metrics
            intensity = (mean_movement * 0.4 + std_movement * 0.3 + flow_magnitude * 0.3) / 255.0
            return min(intensity, 1.0)
            
        except Exception as e:
            return 0.0
    
    def detect_clusters(self, people_positions, people_data):
        """Detect clusters of people."""
        if len(people_positions) < 2:
            return []
        
        positions = np.array(people_positions)
        distances = squareform(pdist(positions))
        
        clusters = []
        visited = set()
        
        for i in range(len(people_positions)):
            if i in visited:
                continue
                
            cluster_indices = [i]
            for j in range(i + 1, len(people_positions)):
                if j not in visited and distances[i][j] <= 150:  # 150 pixel cluster distance
                    cluster_indices.append(j)
                    visited.add(j)
            
            visited.add(i)
            
            if len(cluster_indices) >= 2:  # Minimum 2 people for cluster
                cluster_people = [people_data[idx] for idx in cluster_indices]
                cluster_center = np.mean(positions[cluster_indices], axis=0)
                
                clusters.append({
                    'id': f"cluster_{len(clusters)}",
                    'indices': cluster_indices,
                    'people': cluster_people,
                    'center': cluster_center,
                    'size': len(cluster_indices)
                })
        
        return clusters
    
    def update_person_history(self, person_id, movement_intensity, position, timestamp):
        """Update person's movement history."""
        if person_id not in self.person_histories:
            self.person_histories[person_id] = {
                'movements': deque(maxlen=self.ANALYSIS_WINDOW * self.FRAME_RATE),
                'positions': deque(maxlen=self.ANALYSIS_WINDOW * self.FRAME_RATE),
                'timestamps': deque(maxlen=self.ANALYSIS_WINDOW * self.FRAME_RATE),
                'chaos_score': 0,
                'last_chaos_time': 0
            }
        
        history = self.person_histories[person_id]
        history['movements'].append(movement_intensity)
        history['positions'].append(position)
        history['timestamps'].append(timestamp)
        
        # Calculate chaos score based on movement frequency and intensity
        if len(history['movements']) >= self.MIN_CHAOS_FRAMES:
            recent_movements = list(history['movements'])[-self.MIN_CHAOS_FRAMES:]
            movement_frames = sum(1 for m in recent_movements if m > self.MOVEMENT_THRESHOLD)
            movement_frequency = movement_frames / len(recent_movements)
            avg_intensity = np.mean([m for m in recent_movements if m > self.MOVEMENT_THRESHOLD])
            
            # Calculate chaos score
            if movement_frequency >= self.CHAOS_CRITERIA['movement_frequency']:
                chaos_score = (movement_frequency * 0.6 + avg_intensity * 0.4)
                history['chaos_score'] = min(chaos_score, 1.0)
                
                # Check if this is sustained chaos
                if chaos_score >= self.CHAOS_CRITERIA['individual_chaos_threshold']:
                    history['last_chaos_time'] = timestamp
            else:
                history['chaos_score'] = 0
    
    def analyze_cluster_chaos(self, cluster, timestamp):
        """Analyze if a cluster is creating chaos."""
        cluster_id = cluster['id']
        
        if cluster_id not in self.cluster_histories:
            self.cluster_histories[cluster_id] = {
                'chaos_scores': deque(maxlen=self.ANALYSIS_WINDOW * self.FRAME_RATE),
                'timestamps': deque(maxlen=self.ANALYSIS_WINDOW * self.FRAME_RATE),
                'is_chaotic': False,
                'chaos_duration': 0
            }
        
        cluster_history = self.cluster_histories[cluster_id]
        
        # Calculate average chaos score for cluster
        cluster_chaos_scores = []
        for person in cluster['people']:
            person_id = person['id']
            if person_id in self.person_histories:
                chaos_score = self.person_histories[person_id]['chaos_score']
                cluster_chaos_scores.append(chaos_score)
        
        if cluster_chaos_scores:
            avg_cluster_chaos = np.mean(cluster_chaos_scores)
            cluster_history['chaos_scores'].append(avg_cluster_chaos)
            cluster_history['timestamps'].append(timestamp)
            
            # Determine if cluster is chaotic
            if avg_cluster_chaos >= self.CHAOS_CRITERIA['group_chaos_threshold']:
                cluster_history['is_chaotic'] = True
                if cluster_history['chaos_duration'] == 0:
                    cluster_history['chaos_duration'] = timestamp
            else:
                cluster_history['is_chaotic'] = False
                cluster_history['chaos_duration'] = 0
    
    def analyze_advanced_chaos(self, current_people, prev_frame, current_frame):
        """Main analysis function for advanced chaos detection."""
        timestamp = time.time()
        
        # Calculate movement intensities
        people_with_movement = []
        for person in current_people:
            person_id = person['id']
            bbox = person['rect']
            center = person['center']
            
            # Calculate movement intensity
            movement_intensity = self.calculate_movement_intensity(
                current_frame, prev_frame, bbox
            )
            
            # Update person history
            self.update_person_history(person_id, movement_intensity, center, timestamp)
            
            person['movement_intensity'] = movement_intensity
            person['chaos_score'] = self.person_histories[person_id]['chaos_score']
            people_with_movement.append(person)
        
        # Detect clusters
        people_positions = [person['center'] for person in people_with_movement]
        clusters = self.detect_clusters(people_positions, people_with_movement)
        
        # Analyze cluster chaos
        for cluster in clusters:
            self.analyze_cluster_chaos(cluster, timestamp)
        
        # Store analysis data
        self.analysis_timeline.append({
            'timestamp': timestamp,
            'people': people_with_movement,
            'clusters': clusters,
            'total_people': len(people_with_movement)
        })
        
        # Perform periodic analysis
        if timestamp - self.last_analysis_time >= self.analysis_interval:
            self.perform_chaos_analysis()
            self.last_analysis_time = timestamp
        
        return people_with_movement, clusters
    
    def perform_chaos_analysis(self):
        """Perform comprehensive chaos analysis over the analysis window."""
        if not self.analysis_timeline:
            return
        
        # Analyze individual chaos
        self.individual_chaos = []
        self.individual_calm = []
        
        for person_id, history in self.person_histories.items():
            if len(history['movements']) >= self.MIN_CHAOS_FRAMES:
                recent_movements = list(history['movements'])[-self.MIN_CHAOS_FRAMES:]
                movement_frames = sum(1 for m in recent_movements if m > self.MOVEMENT_THRESHOLD)
                movement_frequency = movement_frames / len(recent_movements)
                
                if movement_frequency >= self.CHAOS_CRITERIA['movement_frequency']:
                    self.individual_chaos.append({
                        'person_id': person_id,
                        'chaos_score': history['chaos_score'],
                        'movement_frequency': movement_frequency
                    })
                else:
                    self.individual_calm.append({
                        'person_id': person_id,
                        'chaos_score': history['chaos_score']
                    })
        
        # Analyze cluster chaos
        self.chaos_clusters = []
        self.calm_clusters = []
        
        for cluster_id, history in self.cluster_histories.items():
            if len(history['chaos_scores']) >= 10:  # Minimum data points
                avg_chaos = np.mean(history['chaos_scores'])
                
                if avg_chaos >= self.CHAOS_CRITERIA['group_chaos_threshold']:
                    self.chaos_clusters.append({
                        'cluster_id': cluster_id,
                        'chaos_score': avg_chaos,
                        'is_chaotic': history['is_chaotic']
                    })
                else:
                    self.calm_clusters.append({
                        'cluster_id': cluster_id,
                        'chaos_score': avg_chaos
                    })
    
    def get_chaos_report(self):
        """Get comprehensive chaos report."""
        if not self.analysis_timeline:
            return {
                'overall_status': 'CALM',
                'chaos_clusters': [],
                'calm_clusters': [],
                'individual_chaos': [],
                'individual_calm': [],
                'analysis_timestamp': time.time()
            }
        
        # Determine overall status
        total_chaos_people = len(self.individual_chaos)
        total_chaos_clusters = len(self.chaos_clusters)
        total_people = len(self.person_histories)
        
        if total_chaos_people > 0 or total_chaos_clusters > 0:
            overall_status = 'CHAOS'
        else:
            overall_status = 'CALM'
        
        return {
            'overall_status': overall_status,
            'chaos_clusters': self.chaos_clusters,
            'calm_clusters': self.calm_clusters,
            'individual_chaos': self.individual_chaos,
            'individual_calm': self.individual_calm,
            'total_people': total_people,
            'chaos_people_count': total_chaos_people,
            'chaos_cluster_count': total_chaos_clusters,
            'analysis_timestamp': time.time()
        }
    
    def get_detailed_chaos_info(self):
        """Get detailed chaos information for display."""
        report = self.get_chaos_report()
        
        info = f"=== CHAOS ANALYSIS REPORT ===\n"
        info += f"Overall Status: {report['overall_status']}\n"
        info += f"Total People: {report['total_people']}\n"
        info += f"Chaos People: {report['chaos_people_count']}\n"
        info += f"Chaos Clusters: {report['chaos_cluster_count']}\n"
        
        if report['chaos_clusters']:
            info += f"\nCHAOTIC CLUSTERS:\n"
            for cluster in report['chaos_clusters']:
                info += f"  Cluster {cluster['cluster_id']}: Chaos Score {cluster['chaos_score']:.2f}\n"
        
        if report['individual_chaos']:
            info += f"\nCHAOTIC INDIVIDUALS:\n"
            for person in report['individual_chaos']:
                info += f"  Person {person['person_id']}: Score {person['chaos_score']:.2f}\n"
        
        if report['overall_status'] == 'CALM':
            info += f"\n✓ All clusters and individuals are calm\n"
        
        return info
