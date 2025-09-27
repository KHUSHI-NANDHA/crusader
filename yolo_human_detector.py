import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import deque
import time

class YOLOHumanDetector:
    """
    YOLO-based human detection system for real-time video processing.
    Uses YOLOv8 for accurate human detection and tracking.
    """
    
    def __init__(self, model_size='n', confidence_threshold=0.5, device='auto'):
        """
        Initialize YOLO human detector.
        
        Args:
            model_size (str): YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device to run on ('auto', 'cpu', 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        try:
            model_name = f'yolov8{model_size}.pt'
            print(f"Loading YOLO model: {model_name} on {self.device}")
            self.model = YOLO(model_name)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to CPU-only model...")
            try:
                self.model = YOLO('yolov8n.pt')
                self.device = 'cpu'
            except Exception as e2:
                print(f"Failed to load YOLO model: {e2}")
                self.model = None
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # Tracking variables
        self.person_tracks = {}
        self.next_person_id = 0
        self.track_history = deque(maxlen=30)
        
        # Performance tracking
        self.detection_times = deque(maxlen=100)
        
        # Adaptive confidence threshold
        self.adaptive_threshold = True
        self.min_confidence = 0.1
        self.max_confidence = 0.5
        self.target_detections = 30  # Target number of detections for adaptive threshold
        
    def detect_humans(self, frame):
        """
        Detect humans in a frame using YOLO.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (detections, processing_time)
                - detections: List of detection dictionaries
                - processing_time: Time taken for detection in seconds
        """
        if self.model is None:
            return [], 0
        
        start_time = time.time()
        
        try:
            # Adaptive confidence threshold adjustment
            current_threshold = self.confidence_threshold
            if self.adaptive_threshold and hasattr(self, '_last_detection_count'):
                if self._last_detection_count < self.target_detections * 0.8:
                    # Too few detections, lower threshold
                    current_threshold = max(self.min_confidence, current_threshold * 0.9)
                elif self._last_detection_count > self.target_detections * 1.5:
                    # Too many detections, raise threshold
                    current_threshold = min(self.max_confidence, current_threshold * 1.1)
            
            # Run YOLO inference with optimized parameters for crowded scenes
            results = self.model(frame, 
                               conf=current_threshold, 
                               device=self.device, 
                               verbose=False,
                               iou=0.5,  # IoU threshold for NMS
                               max_det=100,  # Maximum detections per image
                               agnostic_nms=False)  # Class-agnostic NMS
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # Get first (and only) result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidences
                    class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs
                    
                    # Debug: Print all detections before filtering
                    total_detections = len(class_ids)
                    person_detections = 0
                    
                    # Filter for person detections only
                    for i, class_id in enumerate(class_ids):
                        if int(class_id) == self.person_class_id:
                            person_detections += 1
                            x1, y1, x2, y2 = boxes[i]
                            confidence = confidences[i]
                            
                            # Convert to (x, y, w, h) format
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                            
                            detections.append({
                                'bbox': (x, y, w, h),
                                'confidence': float(confidence),
                                'class_id': int(class_id),
                                'center': (x + w//2, y + h//2)
                            })
                    
                    # Debug output (only print occasionally to avoid spam)
                    if hasattr(self, '_debug_counter'):
                        self._debug_counter += 1
                    else:
                        self._debug_counter = 0
                    
                    if self._debug_counter % 100 == 0:  # Print every 100 frames
                        print(f"YOLO Debug: Total detections: {total_detections}, Person detections: {person_detections}, Confidence threshold: {self.confidence_threshold}")
                        if person_detections > 0:
                            avg_conf = np.mean([d['confidence'] for d in detections])
                            print(f"Average person confidence: {avg_conf:.3f}")
                else:
                    if hasattr(self, '_debug_counter'):
                        self._debug_counter += 1
                    else:
                        self._debug_counter = 0
                    
                    if self._debug_counter % 100 == 0:
                        print("YOLO Debug: No detections found in this frame")
            
            processing_time = time.time() - start_time
            self.detection_times.append(processing_time)
            
            # Store detection count for adaptive thresholding
            self._last_detection_count = len(detections)
            
            return detections, processing_time
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return [], time.time() - start_time
    
    def track_humans(self, detections, frame, prev_frame=None):
        """
        Track humans across frames and calculate movement-based chaos levels.
        
        Args:
            detections: List of current frame detections
            frame: Current frame
            prev_frame: Previous frame for movement calculation
            
        Returns:
            tuple: (tracked_people, chaos_people)
        """
        current_people = []
        chaos_people = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']
            
            # Calculate movement-based chaos level
            chaos_level = self.calculate_movement_chaos(frame, prev_frame, bbox)
            
            # Try to match with existing tracks
            person_id = self.match_or_create_track(bbox, center, chaos_level)
            
            person_data = {
                'id': person_id,
                'rect': bbox,
                'confidence': confidence,
                'center': center,
                'chaos_level': chaos_level,
                'is_chaotic': chaos_level > 0.1  # Threshold for chaos detection
            }
            
            current_people.append(person_data)
            
            if person_data['is_chaotic']:
                chaos_people.append(person_data)
        
        # Update tracking data
        self.update_tracks(current_people)
        
        return current_people, chaos_people
    
    def calculate_movement_chaos(self, frame, prev_frame, bbox):
        """
        Calculate chaos level based on movement in the bounding box region.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            float: Chaos level (0.0 to 1.0)
        """
        if prev_frame is None:
            return 0.0
        
        x, y, w, h = bbox
        
        # Ensure bounding box is within frame bounds
        x = max(0, min(x, frame.shape[1] - w))
        y = max(0, min(y, frame.shape[0] - h))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 0.0
        
        try:
            # Extract regions
            current_region = frame[y:y+h, x:x+w]
            prev_region = prev_frame[y:y+h, x:x+w]
            
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
            
            # Calculate optical flow for more accurate movement detection
            flow = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_current,
                np.array([[x + w//2, y + h//2]], dtype=np.float32),
                None
            )[0]
            
            # Calculate chaos score based on multiple factors
            movement_score = min(mean_movement / 50.0, 1.0)  # Normalize movement
            noise_score = min(std_movement / 30.0, 1.0)  # Normalize noise
            
            # Combine scores
            chaos_score = (movement_score * 0.6 + noise_score * 0.4)
            
            return min(chaos_score, 1.0)
            
        except Exception as e:
            print(f"Error calculating movement chaos: {e}")
            return 0.0
    
    def match_or_create_track(self, bbox, center, chaos_level):
        """
        Match detection to existing track or create new track.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            center: Center point (x, y)
            chaos_level: Current chaos level
            
        Returns:
            int: Person ID
        """
        best_match_id = None
        best_distance = float('inf')
        
        # Look for matching tracks
        for person_id, track in self.person_tracks.items():
            if track['last_seen'] < 5:  # Only consider recently seen people
                distance = self.calculate_distance(center, track['last_center'])
                if distance < best_distance and distance < 100:  # Max distance threshold
                    best_distance = distance
                    best_match_id = person_id
        
        if best_match_id is not None:
            # Update existing track
            self.person_tracks[best_match_id].update({
                'last_bbox': bbox,
                'last_center': center,
                'last_seen': 0,
                'chaos_history': track.get('chaos_history', deque(maxlen=10))
            })
            self.person_tracks[best_match_id]['chaos_history'].append(chaos_level)
            return best_match_id
        else:
            # Create new track
            person_id = self.next_person_id
            self.next_person_id += 1
            self.person_tracks[person_id] = {
                'last_bbox': bbox,
                'last_center': center,
                'last_seen': 0,
                'chaos_history': deque([chaos_level], maxlen=10)
            }
            return person_id
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_tracks(self, current_people):
        """Update tracking data and remove old tracks."""
        # Increment last_seen for all tracks
        for person_id in self.person_tracks:
            self.person_tracks[person_id]['last_seen'] += 1
        
        # Remove tracks that haven't been seen for too long
        to_remove = [pid for pid, track in self.person_tracks.items() if track['last_seen'] > 10]
        for pid in to_remove:
            del self.person_tracks[pid]
    
    def get_average_detection_time(self):
        """Get average detection time for performance monitoring."""
        if not self.detection_times:
            return 0
        return sum(self.detection_times) / len(self.detection_times)
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return "No model loaded"
        
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8',
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'person_class_id': self.person_class_id
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        self.person_tracks.clear()
        self.track_history.clear()
        self.detection_times.clear()
