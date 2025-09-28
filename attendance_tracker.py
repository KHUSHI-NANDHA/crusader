"""
Attendance Tracking System
==========================

Tracks individual people across multiple lectures and maintains attendance records.
Assigns dummy names (Person A, Person B, etc.) to each individual.
"""

import json
import os
import time
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import cv2

class AttendanceTracker:
    def __init__(self, attendance_file="attendance_records.json"):
        self.attendance_file = attendance_file
        self.attendance_records = self.load_attendance_records()
        self.person_database = {}  # person_id -> person_info
        self.next_person_id = 1
        self.current_session_id = None
        self.session_start_time = None
        
        # Person appearance tracking
        self.person_appearances = defaultdict(list)  # person_id -> [session_ids]
        self.session_people = defaultdict(set)  # session_id -> {person_ids}
        
        # Dummy name assignment
        self.dummy_names = []
        self.generate_dummy_names()
        
    def generate_dummy_names(self):
        """Generate dummy names for people"""
        # Generate names like Person A, Person B, Person C, etc.
        for i in range(100):  # Support up to 100 people
            if i < 26:
                name = f"Person {chr(65 + i)}"  # A, B, C, ..., Z
            else:
                name = f"Person {chr(65 + (i % 26))}{i // 26}"  # AA, AB, AC, etc.
            self.dummy_names.append(name)
    
    def load_attendance_records(self):
        """Load attendance records from file"""
        if os.path.exists(self.attendance_file):
            try:
                with open(self.attendance_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_attendance_records(self):
        """Save attendance records to file"""
        try:
            # Convert all sets to lists for JSON serialization
            records_to_save = {}
            for session_id, session_info in self.attendance_records.items():
                records_to_save[session_id] = session_info.copy()
                if 'people_present' in records_to_save[session_id]:
                    if isinstance(records_to_save[session_id]['people_present'], set):
                        records_to_save[session_id]['people_present'] = list(records_to_save[session_id]['people_present'])
            
            with open(self.attendance_file, 'w') as f:
                json.dump(records_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving attendance records: {e}")
    
    def clear_all_data(self):
        """Clear all attendance data for a fresh start"""
        self.attendance_records = {}
        self.person_database = {}
        self.current_session_id = None
        self.session_start_time = None
        print("🗑️ All attendance data cleared for fresh start")
    
    def start_new_session(self, video_path):
        """Start tracking a new session"""
        # Use timestamp with microseconds for better uniqueness
        self.current_session_id = f"session_{int(time.time() * 1000000)}"
        self.session_start_time = datetime.now()
        
        # Generate generic session name based on session count
        session_number = len([s for s in self.attendance_records.values() if len(s['people_present']) > 0]) + 1
        session_name = f"Session {session_number}"
        
        # Initialize session record
        self.attendance_records[self.current_session_id] = {
            'session_name': session_name,
            'video_path': video_path,
            'start_time': self.session_start_time.isoformat(),
            'people_present': set(),
            'attendance_duration': {},
            'total_people': 0
        }
        
        print(f"📚 Started new session: {session_name}")
        print(f"📝 Session ID: {self.current_session_id}")
        
    def end_session(self):
        """End the current session and save records"""
        if self.current_session_id:
            end_time = datetime.now()
            duration = (end_time - self.session_start_time).total_seconds()
            
            # Update session record
            self.attendance_records[self.current_session_id]['end_time'] = end_time.isoformat()
            self.attendance_records[self.current_session_id]['duration_minutes'] = duration / 60
            
            # Convert set to list for JSON serialization
            self.attendance_records[self.current_session_id]['people_present'] = list(
                self.attendance_records[self.current_session_id]['people_present']
            )
            
            # Save records
            self.save_attendance_records()
            
            print(f"✅ Session ended: {self.attendance_records[self.current_session_id]['session_name']}")
            print(f"⏱️ Duration: {duration/60:.1f} minutes")
            print(f"👥 People present: {len(self.attendance_records[self.current_session_id]['people_present'])}")
            
            self.current_session_id = None
            self.session_start_time = None
    
    def register_person(self, person_id, bbox, confidence=1.0):
        """Register a person in the current session"""
        if not self.current_session_id:
            print(f"⚠️ No current session ID - cannot register person {person_id}")
            return None
            
        # Check if person already exists in database
        if person_id not in self.person_database:
            # Assign dummy name
            dummy_name = self.dummy_names[len(self.person_database)]
            
            # Create person record
            self.person_database[person_id] = {
                'person_id': person_id,
                'dummy_name': dummy_name,
                'first_seen': datetime.now().isoformat(),
                'total_sessions': 0,
                'session_history': [],
                'appearance_count': 0
            }
            
            print(f"🆕 New person detected: {dummy_name} (ID: {person_id})")
        else:
            dummy_name = self.person_database[person_id]['dummy_name']
        
        # Add to current session
        self.attendance_records[self.current_session_id]['people_present'].add(person_id)
        
        # Update person's session history
        if self.current_session_id not in self.person_database[person_id]['session_history']:
            self.person_database[person_id]['session_history'].append(self.current_session_id)
            self.person_database[person_id]['total_sessions'] += 1
        
        # Update appearance count
        self.person_database[person_id]['appearance_count'] += 1
        
        return self.person_database[person_id]['dummy_name']
    
    def get_person_info(self, person_id):
        """Get information about a person"""
        if person_id in self.person_database:
            return self.person_database[person_id]
        return None
    
    def get_attendance_summary(self):
        """Get summary of all attendance records"""
        # Only count sessions that actually have people
        sessions_with_people = [session_id for session_id, session_info in self.attendance_records.items() 
                               if len(session_info['people_present']) > 0]
        
        summary = {
            'total_sessions': len(sessions_with_people),  # Only count sessions with people
            'total_people': len(self.person_database),
            'people_attendance': {},
            'session_attendance': {}
        }
        
        # Calculate attendance for each person
        for person_id, person_info in self.person_database.items():
            summary['people_attendance'][person_info['dummy_name']] = {
                'total_sessions': person_info['total_sessions'],
                'session_history': person_info['session_history'],
                'first_seen': person_info['first_seen'],
                'appearance_count': person_info['appearance_count']
            }
        
        # Calculate attendance for each session
        for session_id, session_info in self.attendance_records.items():
            summary['session_attendance'][session_id] = {
                'session_name': session_info['session_name'],
                'people_count': len(session_info['people_present']),
                'people_present': [self.person_database[pid]['dummy_name'] for pid in session_info['people_present'] if pid in self.person_database]
            }
        
        return summary
    
    def get_person_attendance_stats(self):
        """Get detailed attendance statistics for each person"""
        stats = []
        
        # Get total sessions that actually have people (not empty sessions)
        total_sessions_with_people = len([session_id for session_id, session_info in self.attendance_records.items() 
                                        if len(session_info['people_present']) > 0])
        
        for person_id, person_info in self.person_database.items():
            # Calculate percentage based on sessions with people, not total sessions
            percentage = (person_info['total_sessions'] / total_sessions_with_people) * 100 if total_sessions_with_people > 0 else 0
            
            stats.append({
                'person_id': person_id,
                'dummy_name': person_info['dummy_name'],
                'total_sessions_attended': person_info['total_sessions'],
                'session_percentage': percentage,
                'first_seen': person_info['first_seen'],
                'session_history': person_info['session_history']
            })
        
        # Sort by total sessions attended
        stats.sort(key=lambda x: x['total_sessions_attended'], reverse=True)
        return stats
    
    def get_current_session_info(self):
        """Get information about the current session"""
        if self.current_session_id:
            return {
                'session_id': self.current_session_id,
                'session_name': self.attendance_records[self.current_session_id]['session_name'],
                'people_present': len(self.attendance_records[self.current_session_id]['people_present']),
                'people_list': [self.person_database[pid]['dummy_name'] for pid in self.attendance_records[self.current_session_id]['people_present'] if pid in self.person_database]
            }
        return None
    
    def export_attendance_report(self, filename="attendance_report.txt"):
        """Export a detailed attendance report"""
        summary = self.get_attendance_summary()
        person_stats = self.get_person_attendance_stats()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("📊 ATTENDANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📚 Total Sessions: {summary['total_sessions']}\n")
            f.write(f"👥 Total People: {summary['total_people']}\n\n")
            
            f.write("👤 PERSON ATTENDANCE:\n")
            f.write("-" * 30 + "\n")
            for stat in person_stats:
                f.write(f"{stat['dummy_name']}: {stat['total_sessions_attended']} sessions ({stat['session_percentage']:.1f}%)\n")
            
            f.write("\n📚 SESSION ATTENDANCE:\n")
            f.write("-" * 30 + "\n")
            for session_id, session_info in summary['session_attendance'].items():
                f.write(f"{session_info['session_name']}: {session_info['people_count']} people\n")
                f.write(f"  People: {', '.join(session_info['people_present'])}\n\n")
        
        print(f"📄 Attendance report exported to: {filename}")
    
    def reset_attendance(self):
        """Reset all attendance records (use with caution)"""
        self.attendance_records = {}
        self.person_database = {}
        self.next_person_id = 1
        self.person_appearances = defaultdict(list)
        self.lecture_people = defaultdict(set)
        
        # Remove attendance file
        if os.path.exists(self.attendance_file):
            os.remove(self.attendance_file)
        
        print("🔄 All attendance records reset")





