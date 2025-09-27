# Student Mood & Chaos Detection System

A real-time video analysis system that detects student moods, counts people, and measures chaos levels in classroom videos.

## Features

- **Mood Detection**: Identifies student emotions (happy, excited, calm, sad, neutral)
- **People Counting**: Counts the number of people in the video
- **Chaos Detection**: Measures activity and movement levels
- **Real-time Processing**: Live analysis of video feeds
- **Simple GUI**: Easy-to-use interface for video selection and results display

## Installation

1. Install Python 3.7 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Click "Browse" to select an MP4 video file
3. Click "Start Processing" to begin analysis
4. View real-time results in the interface

## How It Works

### Mood Detection
- Uses MediaPipe for facial landmark detection
- Analyzes facial geometry to determine emotions
- Tracks mood changes over time

### People Counting
- Detects faces using MediaPipe face detection
- Counts unique faces in each frame
- Provides average count over time

### Chaos Detection
- Measures frame-to-frame movement
- Calculates noise levels in the video
- Combines metrics for chaos score (0-100%)

## System Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- TensorFlow
- Tkinter (usually included with Python)

## File Structure

- `main.py` - Main application entry point
- `video_processor.py` - GUI and video processing logic
- `mood_detector.py` - Core analysis algorithms
- `requirements.txt` - Python dependencies

## Troubleshooting

- Ensure video file is in MP4 format
- Check that all dependencies are installed
- Make sure video file is not corrupted
- For best results, use videos with clear faces and good lighting

## Notes

- The system works best with clear, well-lit videos
- Processing speed depends on video resolution and system performance
- Results improve with longer video sequences as more data is collected
