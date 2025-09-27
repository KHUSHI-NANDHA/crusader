import cv2
import numpy as np
from collections import deque
import time
from scipy.spatial.distance import pdist, squareform

class EnhancedChaosAnalyzer:
    """
    Enhanced chaos detection system with individual focus and structured activity classification.
    Distinguishes between individual work, structured group work, and structured chaos.
    """
    
    def __init__(self):
        # Movement speed thresholds
        self.SPEED_THRESHOLDS = {
            'WORK_MIN': 10,      # Minimum speed for work classification
            'WORK_MAX': 20,      # Maximum speed for work classification
            'CHAOS_THRESHOLD': 20  # Speed threshold for chaos
        }
        
        # Clustering parameters
        self.CLUSTER_DISTANCE = 100  # Maximum distance for clustering (pixels)
        self.MIN_CLUSTER_SIZE = 2    # Minimum people for a cluster
        
        # Individual chaos detection
        self.INDIVIDUAL_CHAOS_THRESHOLD = 60  # Minimum chaos score for individual chaos
        
        # Activity classification
        self.ACTIVITY_TYPES = {
            'INDIVIDUAL_WORK': 'Individual Work',
            'STRUCTURED_GROUP_WORK': 'Structured Group Work', 
            'STRUCTURED_CHAOS': 'Structured Chaos',
            'INDIVIDUAL_CHAOS': 'Individual Chaos',
            'CALM': 'Calm Activity'
        }
        
        # Tracking variables
        self.person_histories = {}
        self.cluster_history = deque(maxlen=30)
        self.activity_history = deque(maxlen=100)
        
        # Duplicate detection prevention
        self.detection_cache = {}
        self.cache_duration = 1.0  # Cache duration in seconds
        
    def calculate_movement_speed(self, current_frame, previous_frame, bbox):
        """
        Calculate movement speed for a person in pixels per second.
        Returns speed value and movement vector.
        """
        if previous_frame is None:
            return 0, (0, 0)
        
        x, y, w, h = bbox
        
        # Ensure bounding box is within frame bounds
        x = max(0, min(x, current_frame.shape[1] - w))
        y = max(0, min(y, current_frame.shape[0] - h))
        w = min(w, current_frame.shape[1] - x)
        h = min(h, current_frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 0, (0, 0)
        
        try:
            # Extract regions
            current_region = current_frame[y:y+h, x:x+w]
            prev_region = previous_frame[y:y+h, x:x+w]
            
            if current_region.size == 0 or prev_region.size == 0:
                return 0, (0, 0)
            
            # Convert to grayscale
            gray_current = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow for accurate movement detection
            try:
                # Use Lucas-Kanade optical flow
                p0 = np.array([[[x + w//2, y + h//2]]], dtype=np.float32)
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_current, p0, None)
                
                if st[0][0] == 1:  # If tracking was successful
                    movement_vector = p1[0][0] - p0[0][0]
                    speed = np.sqrt(np.sum(movement_vector**2))
                    return speed, movement_vector
                else:
                    return 0, (0, 0)
            except:
                return 0, (0, 0)
                
        except Exception as e:
            print(f"Error calculating movement speed: {e}")
            return 0, (0, 0)
    
    def detect_clusters(self, people_positions, people_data):
        """
        Detect clusters of people based on their positions.
        Returns cluster information.
        """
        if len(people_positions) < self.MIN_CLUSTER_SIZE:
            return []
        
        # Calculate pairwise distances
        positions = np.array(people_positions)
        distances = squareform(pdist(positions))
        
        clusters = []
        visited = set()
        
        for i in range(len(people_positions)):
            if i in visited:
                continue
                
            # Find all people within cluster distance
            cluster_indices = [i]
            for j in range(i + 1, len(people_positions)):
                if j not in visited and distances[i][j] <= self.CLUSTER_DISTANCE:
                    cluster_indices.append(j)
                    visited.add(j)
            
            visited.add(i)
            
            # Only consider clusters with minimum size
            if len(cluster_indices) >= self.MIN_CLUSTER_SIZE:
                cluster_people = [people_data[idx] for idx in cluster_indices]
                cluster_center = np.mean(positions[cluster_indices], axis=0)
                
                clusters.append({
                    'indices': cluster_indices,
                    'people': cluster_people,
                    'center': cluster_center,
                    'size': len(cluster_indices)
                })
        
        return clusters
    
    def classify_individual_activity(self, person_data, speed, movement_vector):
        """
        Classify individual activity based on speed and chaos level.
        """
        chaos_level = person_data.get('chaos_level', 0)
        
        # Individual chaos detection
        if chaos_level >= self.INDIVIDUAL_CHAOS_THRESHOLD:
            return 'INDIVIDUAL_CHAOS'
        
        # Work classification based on speed
        if self.SPEED_THRESHOLDS['WORK_MIN'] <= speed <= self.SPEED_THRESHOLDS['WORK_MAX']:
            return 'INDIVIDUAL_WORK'
        
        # High speed but not chaotic
        if speed > self.SPEED_THRESHOLDS['CHAOS_THRESHOLD']:
            return 'INDIVIDUAL_CHAOS'
        
        return 'CALM'
    
    def classify_cluster_activity(self, cluster, people_data):
        """
        Classify cluster activity based on average speed and chaos levels.
        """
        cluster_people = cluster['people']
        speeds = []
        chaos_levels = []
        
        for person in cluster_people:
            person_id = person['id']
            if person_id in self.person_histories:
                history = self.person_histories[person_id]
                if 'speeds' in history and history['speeds']:
                    speeds.append(np.mean(history['speeds']))
                if 'chaos_levels' in history and history['chaos_levels']:
                    chaos_levels.append(np.mean(history['chaos_levels']))
        
        if not speeds:
            return 'CALM'
        
        avg_speed = np.mean(speeds)
        avg_chaos = np.mean(chaos_levels) if chaos_levels else 0
        
        # Structured group work: clustered + work speed range
        if (self.SPEED_THRESHOLDS['WORK_MIN'] <= avg_speed <= self.SPEED_THRESHOLDS['WORK_MAX'] 
            and avg_chaos < self.INDIVIDUAL_CHAOS_THRESHOLD):
            return 'STRUCTURED_GROUP_WORK'
        
        # Structured chaos: clustered + high speed
        elif avg_speed > self.SPEED_THRESHOLDS['CHAOS_THRESHOLD']:
            return 'STRUCTURED_CHAOS'
        
        return 'CALM'
    
    def prevent_duplicates(self, person_id, activity_type, timestamp):
        """
        Prevent duplicate detections using caching.
        """
        cache_key = f"{person_id}_{activity_type}"
        
        if cache_key in self.detection_cache:
            last_time = self.detection_cache[cache_key]
            if timestamp - last_time < self.cache_duration:
                return False  # Duplicate detected
        
        self.detection_cache[cache_key] = timestamp
        return True  # New detection
    
    def update_person_history(self, person_id, speed, chaos_level, position, timestamp):
        """
        Update person's movement and chaos history.
        """
        if person_id not in self.person_histories:
            self.person_histories[person_id] = {
                'speeds': deque(maxlen=30),
                'chaos_levels': deque(maxlen=30),
                'positions': deque(maxlen=30),
                'timestamps': deque(maxlen=30)
            }
        
        history = self.person_histories[person_id]
        history['speeds'].append(speed)
        history['chaos_levels'].append(chaos_level)
        history['positions'].append(position)
        history['timestamps'].append(timestamp)
    
    def analyze_enhanced_chaos(self, current_people, prev_frame, current_frame):
        """
        Enhanced chaos analysis with individual focus and structured activity detection.
        """
        timestamp = time.time()
        
        # Calculate movement speeds for all people
        people_with_speeds = []
        for person in current_people:
            person_id = person['id']
            bbox = person['rect']
            center = person['center']
            
            # Calculate movement speed
            speed, movement_vector = self.calculate_movement_speed(
                current_frame, prev_frame, bbox
            )
            
            # Update person data
            person['movement_speed'] = speed
            person['movement_vector'] = movement_vector
            
            # Update person history
            self.update_person_history(
                person_id, speed, person.get('chaos_level', 0), center, timestamp
            )
            
            people_with_speeds.append(person)
        
        # Detect clusters
        people_positions = [person['center'] for person in people_with_speeds]
        clusters = self.detect_clusters(people_positions, people_with_speeds)
        
        # Classify individual activities
        individual_activities = {}
        for person in people_with_speeds:
            person_id = person['id']
            speed = person['movement_speed']
            activity_type = self.classify_individual_activity(person, speed, person['movement_vector'])
            
            # Prevent duplicates
            if self.prevent_duplicates(person_id, activity_type, timestamp):
                individual_activities[person_id] = {
                    'activity_type': activity_type,
                    'activity_name': self.ACTIVITY_TYPES[activity_type],
                    'speed': speed,
                    'chaos_level': person.get('chaos_level', 0)
                }
            
            person['activity_type'] = activity_type
            person['activity_name'] = self.ACTIVITY_TYPES[activity_type]
        
        # Classify cluster activities
        cluster_activities = {}
        for i, cluster in enumerate(clusters):
            cluster_activity = self.classify_cluster_activity(cluster, people_with_speeds)
            
            # Update all people in cluster
            for person_idx in cluster['indices']:
                person = people_with_speeds[person_idx]
                person_id = person['id']
                
                # Override individual classification with cluster classification
                if cluster_activity in ['STRUCTURED_GROUP_WORK', 'STRUCTURED_CHAOS']:
                    person['activity_type'] = cluster_activity
                    person['activity_name'] = self.ACTIVITY_TYPES[cluster_activity]
                    
                    # Prevent duplicates for cluster activities
                    if self.prevent_duplicates(f"{person_id}_cluster_{i}", cluster_activity, timestamp):
                        cluster_activities[f"{person_id}_cluster_{i}"] = {
                            'activity_type': cluster_activity,
                            'activity_name': self.ACTIVITY_TYPES[cluster_activity],
                            'cluster_size': cluster['size'],
                            'cluster_center': cluster['center']
                        }
        
        # Store activity history
        self.activity_history.append({
            'timestamp': timestamp,
            'individual_activities': individual_activities,
            'cluster_activities': cluster_activities,
            'clusters': clusters,
            'total_people': len(people_with_speeds)
        })
        
        return people_with_speeds, individual_activities, cluster_activities, clusters
    
    def get_activity_summary(self):
        """
        Get summary of current activities and their distribution.
        """
        if not self.activity_history:
            return {
                'individual_work': 0,
                'structured_group_work': 0,
                'structured_chaos': 0,
                'individual_chaos': 0,
                'calm': 0,
                'total_people': 0,
                'active_clusters': 0
            }
        
        # Analyze recent activities (last 10 frames)
        recent_activities = list(self.activity_history)[-10:]
        
        activity_counts = {
            'individual_work': 0,
            'structured_group_work': 0,
            'structured_chaos': 0,
            'individual_chaos': 0,
            'calm': 0
        }
        
        total_people = 0
        active_clusters = 0
        
        for frame_data in recent_activities:
            total_people = max(total_people, frame_data['total_people'])
            active_clusters = max(active_clusters, len(frame_data['clusters']))
            
            # Count individual activities
            for activity in frame_data['individual_activities'].values():
                activity_type = activity['activity_type']
                if activity_type in activity_counts:
                    activity_counts[activity_type] += 1
            
            # Count cluster activities
            for activity in frame_data['cluster_activities'].values():
                activity_type = activity['activity_type']
                if activity_type in activity_counts:
                    activity_counts[activity_type] += 1
        
        return {
            **activity_counts,
            'total_people': total_people,
            'active_clusters': active_clusters
        }
    
    def get_detailed_activity_report(self):
        """
        Get detailed report of current activities.
        """
        if not self.activity_history:
            return "No activity data available"
        
        latest_frame = list(self.activity_history)[-1]
        summary = self.get_activity_summary()
        
        report = f"Activity Report:\n"
        report += f"Total People: {summary['total_people']}\n"
        report += f"Active Clusters: {summary['active_clusters']}\n"
        report += f"Individual Work: {summary['individual_work']}\n"
        report += f"Structured Group Work: {summary['structured_group_work']}\n"
        report += f"Structured Chaos: {summary['structured_chaos']}\n"
        report += f"Individual Chaos: {summary['individual_chaos']}\n"
        report += f"Calm Activity: {summary['calm']}\n"
        
        return report
