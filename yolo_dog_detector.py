#!/usr/bin/env python3
"""
YOLO Dog Detector - Modified to detect dogs instead of people
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLODogDetector:
    def __init__(self, model_size='s', confidence_threshold=0.25):
        """
        Initialize YOLO dog detector
        
        Args:
            model_size (str): Model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold (float): Confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        
        # Load YOLO model
        model_path = f"yolov8{model_size}.pt"
        print(f"Loading YOLO model: {model_path} on cpu")
        
        try:
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
        # Dog class ID in COCO dataset
        self.dog_class_id = 15  # 'dog' is class 15 in COCO dataset
        
        # Tracking variables
        self.next_dog_id = 0
        self.dog_tracks = {}
        self.track_history = {}
        
    def detect_dogs(self, frame):
        """
        Detect dogs in the frame using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (detections, detection_time)
        """
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only process dog detections
                        if class_id == self.dog_class_id and confidence >= self.confidence_threshold:
                            # Convert to bbox format (x, y, w, h)
                            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            
                            detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'center': center,
                                'class_id': class_id,
                                'class_name': 'dog'
                            })
            
            detection_time = time.time() - start_time
            
            # Debug output
            print(f"YOLO Debug: Total detections: {len(detections)}, Dog detections: {len(detections)}, Confidence threshold: {self.confidence_threshold}")
            
            return detections, detection_time
            
        except Exception as e:
            print(f"Error in dog detection: {e}")
            return [], 0
    
    def track_dogs(self, detections, frame, prev_frame=None):
        """
        Track dogs across frames
        
        Args:
            detections: List of dog detections
            frame: Current frame
            prev_frame: Previous frame (optional)
            
        Returns:
            tuple: (current_dogs, dog_tracks)
        """
        current_dogs = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            
            # Try to match with existing tracks
            dog_id = self.match_or_create_track(bbox, center, confidence)
            
            # Calculate movement if we have previous frame
            movement_speed = 0
            if prev_frame is not None:
                movement_speed = self.calculate_movement_speed(bbox, prev_frame, frame)
            
            current_dogs.append({
                'id': dog_id,
                'rect': bbox,
                'center': center,
                'confidence': confidence,
                'class_name': 'dog',
                'movement_speed': movement_speed,
                'is_active': movement_speed > 5,  # Consider dog active if moving
                'activity_type': 'PLAYING' if movement_speed > 10 else 'RESTING' if movement_speed < 2 else 'WALKING'
            })
            
            # Update track history
            if dog_id not in self.track_history:
                self.track_history[dog_id] = []
            self.track_history[dog_id].append(center)
            
            # Keep only recent history
            if len(self.track_history[dog_id]) > 30:
                self.track_history[dog_id] = self.track_history[dog_id][-30:]
        
        # Update tracks
        self.update_tracks(current_dogs)
        
        return current_dogs, self.track_history
    
    def match_or_create_track(self, bbox, center, confidence):
        """Match detection to existing track or create new one"""
        best_match_id = None
        best_distance = float('inf')
        
        x, y, w, h = bbox
        detection_center = center
        
        for dog_id, track in self.dog_tracks.items():
            if track['last_seen'] < 10:  # Only consider recently seen dogs
                track_center = track['last_center']
                distance = np.sqrt((detection_center[0] - track_center[0])**2 + 
                                 (detection_center[1] - track_center[1])**2)
                
                if distance < best_distance and distance < 100:  # Max distance threshold
                    best_distance = distance
                    best_match_id = dog_id
        
        if best_match_id is not None:
            # Update existing track
            self.dog_tracks[best_match_id]['last_bbox'] = bbox
            self.dog_tracks[best_match_id]['last_center'] = center
            self.dog_tracks[best_match_id]['last_seen'] = 0
            self.dog_tracks[best_match_id]['confidence'] = confidence
            return best_match_id
        else:
            # Create new track
            dog_id = self.next_dog_id
            self.next_dog_id += 1
            self.dog_tracks[dog_id] = {
                'last_bbox': bbox,
                'last_center': center,
                'last_seen': 0,
                'confidence': confidence
            }
            return dog_id
    
    def calculate_movement_speed(self, bbox, prev_frame, current_frame):
        """Calculate movement speed of dog"""
        try:
            x, y, w, h = bbox
            
            # Extract region of interest
            roi_current = current_frame[y:y+h, x:x+w]
            roi_prev = prev_frame[y:y+h, x:x+w]
            
            if roi_current.size == 0 or roi_prev.size == 0:
                return 0
            
            # Convert to grayscale
            gray_current = cv2.cvtColor(roi_current, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(gray_prev, gray_current, 
                                          np.array([[x+w//2, y+h//2]], dtype=np.float32), 
                                          None)[0]
            
            if flow is not None and len(flow) > 0:
                # Calculate movement magnitude
                movement = np.sqrt((flow[0][0] - (x+w//2))**2 + (flow[0][1] - (y+h//2))**2)
                return movement
            else:
                return 0
                
        except Exception as e:
            print(f"Error calculating movement speed: {e}")
            return 0
    
    def update_tracks(self, current_dogs):
        """Update dog tracks and remove old ones"""
        # Increment last_seen for all tracks
        for dog_id in self.dog_tracks:
            self.dog_tracks[dog_id]['last_seen'] += 1
        
        # Remove tracks that haven't been seen for too long
        to_remove = [did for did, track in self.dog_tracks.items() if track['last_seen'] > 15]
        for did in to_remove:
            del self.dog_tracks[did]
            if did in self.track_history:
                del self.track_history[did]
    
    def get_dog_analysis(self, current_dogs):
        """Get analysis summary of detected dogs"""
        if not current_dogs:
            return {
                'total_dogs': 0,
                'active_dogs': 0,
                'resting_dogs': 0,
                'playing_dogs': 0,
                'average_confidence': 0,
                'average_movement': 0
            }
        
        active_dogs = sum(1 for dog in current_dogs if dog['is_active'])
        playing_dogs = sum(1 for dog in current_dogs if dog['activity_type'] == 'PLAYING')
        resting_dogs = sum(1 for dog in current_dogs if dog['activity_type'] == 'RESTING')
        
        avg_confidence = sum(dog['confidence'] for dog in current_dogs) / len(current_dogs)
        avg_movement = sum(dog['movement_speed'] for dog in current_dogs) / len(current_dogs)
        
        return {
            'total_dogs': len(current_dogs),
            'active_dogs': active_dogs,
            'resting_dogs': resting_dogs,
            'playing_dogs': playing_dogs,
            'average_confidence': avg_confidence,
            'average_movement': avg_movement
        }


