"""
IceAnalysis API - Figure Skating Video Analysis Backend
Uses MediaPipe Pose for skeleton detection and custom algorithms for element detection.
"""

import os
# Disable GPU for MediaPipe (required for headless server environments)
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import tempfile
import math
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI(
    title="IceAnalysis API",
    description="AI-powered figure skating video analysis",
    version="1.0.0"
)

# CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization for MediaPipe Pose (avoids GPU issues at startup)
mp_pose = mp.solutions.pose
_pose_instance = None

def get_pose():
    """Get or create MediaPipe Pose instance (lazy init for serverless)."""
    global _pose_instance
    if _pose_instance is None:
        _pose_instance = mp_pose.Pose(
            static_image_mode=True,  # True = CPU-friendly, no video tracking state
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _pose_instance

# Response models
class Keypoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: float
    name: str

class FramePose(BaseModel):
    frame: int
    timestamp_ms: float
    keypoints: List[Keypoint]
    
class SkatingElement(BaseModel):
    type: str  # 'jump' or 'spin'
    name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    score: float
    feedback: List[str]

class AnalysisResult(BaseModel):
    success: bool
    total_frames: int
    fps: float
    duration_seconds: float
    elements: List[SkatingElement]
    session_feedback: List[str]
    poses: Optional[List[FramePose]] = None

# Keypoint names from MediaPipe
KEYPOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

class SkatingAnalyzer:
    """Analyzes pose sequences to detect skating elements."""
    
    def __init__(self):
        self.frame_history = []
        
    def reset(self):
        self.frame_history = []
    
    def analyze_frame(self, keypoints: List[Keypoint], frame: int, fps: float):
        """Analyze a single frame's pose data."""
        if len(keypoints) < 33:
            return None
            
        # Get key landmarks
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        
        # Calculate center of mass (hip midpoint)
        center_y = (left_hip.y + right_hip.y) / 2
        center_x = (left_hip.x + right_hip.x) / 2
        
        # Calculate body angle
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        body_angle = math.degrees(math.atan2(
            shoulder_mid_y - center_y,
            shoulder_mid_x - center_x
        ))
        
        # Ankle height (lower y = higher in frame = higher jump)
        ankle_y = min(left_ankle.y, right_ankle.y)
        
        analysis = {
            'frame': frame,
            'center_x': center_x,
            'center_y': center_y,
            'ankle_y': ankle_y,
            'body_angle': body_angle,
            'is_airborne': False,
            'rotation_speed': 0
        }
        
        # Detect if airborne (compare to baseline)
        if len(self.frame_history) >= 10:
            baseline_ankle = sum(f['ankle_y'] for f in self.frame_history[-10:]) / 10
            # If ankle is significantly higher (lower y value) than baseline
            # Lowered threshold from 0.05 to 0.02 for better detection of smaller jumps
            if ankle_y < baseline_ankle - 0.02:  # 2% of frame height (more sensitive)
                analysis['is_airborne'] = True
            # Also check hip position change for jump detection
            if len(self.frame_history) >= 5:
                baseline_hip = sum(f['center_y'] for f in self.frame_history[-5:]) / 5
                if center_y < baseline_hip - 0.03:  # Hip rises during jump
                    analysis['is_airborne'] = True
        
        # Calculate rotation speed
        if len(self.frame_history) >= 1:
            prev = self.frame_history[-1]
            angle_diff = body_angle - prev['body_angle']
            # Handle angle wraparound
            if angle_diff > 180:
                angle_diff -= 360
            if angle_diff < -180:
                angle_diff += 360
            analysis['rotation_speed'] = angle_diff * fps  # degrees per second
        
        self.frame_history.append(analysis)
        return analysis
    
    def detect_elements(self, fps: float) -> List[SkatingElement]:
        """Detect jumps and spins from analyzed frames."""
        elements = []
        
        if len(self.frame_history) < 30:
            return elements
        
        # Detect jumps (airborne sequences)
        jump_start = None
        total_rotation = 0
        
        for i, frame in enumerate(self.frame_history):
            if frame['is_airborne'] and jump_start is None:
                jump_start = i
                total_rotation = 0
            elif frame['is_airborne'] and jump_start is not None:
                total_rotation += abs(frame['rotation_speed'] / fps)
            elif not frame['is_airborne'] and jump_start is not None:
                # Jump ended
                air_frames = i - jump_start
                if air_frames >= 2:  # Minimum 2 frames airborne (was 5 - too strict)
                    jump = self._classify_jump(
                        self.frame_history[jump_start:i],
                        total_rotation,
                        fps
                    )
                    if jump:
                        elements.append(jump)
                jump_start = None
                total_rotation = 0
        
        # Detect spins (high rotation while grounded)
        spin_start = None
        
        for i, frame in enumerate(self.frame_history):
            is_spinning = abs(frame['rotation_speed']) > 180 and not frame['is_airborne']
            
            if is_spinning and spin_start is None:
                spin_start = i
            elif not is_spinning and spin_start is not None:
                spin_frames = i - spin_start
                if spin_frames >= 30:  # Minimum 1 second spin
                    spin = self._classify_spin(
                        self.frame_history[spin_start:i],
                        fps
                    )
                    if spin:
                        elements.append(spin)
                spin_start = None
        
        return elements
    
    def _classify_jump(self, frames: list, total_rotation: float, fps: float) -> Optional[SkatingElement]:
        """Classify a detected jump."""
        rotations = total_rotation / 360
        # Accept even small jumps (waltz jumps are ~0.5 rotation, bunny hops less)
        if rotations < 0.3:
            return None
        
        # Determine rotation count
        rot_count = round(rotations * 2) / 2  # Round to nearest 0.5
        
        # Determine jump name
        if rot_count == 1.5 or rot_count == 2.5 or rot_count == 3.5:
            # Axel (half rotation extra)
            prefix = {1.5: 'Single', 2.5: 'Double', 3.5: 'Triple'}.get(rot_count, '')
            name = f'{prefix} Axel'
        else:
            prefix = {1: 'Single', 2: 'Double', 3: 'Triple', 4: 'Quad'}.get(int(rot_count), '')
            name = f'{prefix} Jump'
        
        # Generate feedback
        feedback = []
        air_time = len(frames) / fps
        
        if air_time > 0.5:
            feedback.append(f'Good air time ({air_time:.2f}s)')
        else:
            feedback.append(f'Work on jump height (air time: {air_time:.2f}s)')
        
        # Check body position
        avg_angle = sum(f['body_angle'] for f in frames) / len(frames)
        if abs(avg_angle) < 15:
            feedback.append('Good vertical axis in air')
        else:
            feedback.append('Try to stay more vertical during rotation')
        
        # Calculate score (simplified)
        base_value = rot_count * 1.5
        height_bonus = 0.3 if air_time > 0.5 else -0.2
        score = round(base_value + height_bonus, 1)
        
        return SkatingElement(
            type='jump',
            name=name,
            start_frame=frames[0]['frame'],
            end_frame=frames[-1]['frame'],
            start_time=frames[0]['frame'] / fps,
            end_time=frames[-1]['frame'] / fps,
            confidence=min(0.95, 0.6 + rot_count * 0.1),
            score=score,
            feedback=feedback
        )
    
    def _classify_spin(self, frames: list, fps: float) -> Optional[SkatingElement]:
        """Classify a detected spin."""
        # Calculate total revolutions
        total_rotation = sum(abs(f['rotation_speed'] / fps) for f in frames)
        revolutions = total_rotation / 360
        
        if revolutions < 2:
            return None
        
        # Determine spin type from body angle
        avg_body_angle = sum(f['body_angle'] for f in frames) / len(frames)
        
        if abs(avg_body_angle) > 60:
            name = 'Camel Spin'
            feedback = ['Good horizontal position']
        elif abs(avg_body_angle) > 30:
            name = 'Sit Spin'
            feedback = ['Good sit position']
        else:
            name = 'Upright Spin'
            feedback = ['Nice vertical alignment']
        
        # Check centering (drift)
        start_x = frames[0]['center_x']
        end_x = frames[-1]['center_x']
        drift = abs(end_x - start_x)
        
        if drift < 0.1:
            feedback.append('Excellent centering - minimal travel')
        else:
            feedback.append('Work on centering - spin is traveling')
        
        feedback.append(f'{int(revolutions)} revolutions')
        
        # Calculate score
        base_value = 2.0
        rev_bonus = 0.5 if revolutions >= 6 else 0
        center_bonus = 0.3 if drift < 0.1 else -0.2
        score = round(base_value + rev_bonus + center_bonus, 1)
        
        return SkatingElement(
            type='spin',
            name=name,
            start_frame=frames[0]['frame'],
            end_frame=frames[-1]['frame'],
            start_time=frames[0]['frame'] / fps,
            end_time=frames[-1]['frame'] / fps,
            confidence=0.85,
            score=score,
            feedback=feedback
        )

def extract_poses_from_video(video_path: str, sample_rate: int = 3) -> tuple:
    """Extract poses from video using MediaPipe."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    poses = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every Nth frame to speed up processing
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe (lazy init)
            results = get_pose().process(rgb_frame)
            
            if results.pose_landmarks:
                keypoints = []
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoints.append(Keypoint(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=landmark.visibility,
                        name=KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f'point_{i}'
                    ))
                
                poses.append(FramePose(
                    frame=frame_idx,
                    timestamp_ms=(frame_idx / fps) * 1000,
                    keypoints=keypoints
                ))
        
        frame_idx += 1
    
    cap.release()
    return poses, fps, total_frames

@app.get("/")
async def root():
    return {
        "service": "IceAnalysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /analyze": "Upload video for skating analysis",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_video(
    video: UploadFile = File(...),
    include_poses: bool = False,
    sample_rate: int = 2  # Changed from 3 to 2 for better detection
):
    """
    Analyze a figure skating video.
    
    - **video**: Video file (mp4, mov, etc.)
    - **include_poses**: Include full pose data in response (larger payload)
    - **sample_rate**: Analyze every Nth frame (default 3, lower = more accurate but slower)
    """
    
    # Validate file type
    if not video.content_type or not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extract poses
        poses, fps, total_frames = extract_poses_from_video(tmp_path, sample_rate)
        
        if len(poses) < 10:
            return AnalysisResult(
                success=False,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=total_frames / fps if fps > 0 else 0,
                elements=[],
                session_feedback=["Could not detect skater in video. Make sure the full body is visible."],
                poses=poses if include_poses else None
            )
        
        # Analyze poses
        analyzer = SkatingAnalyzer()
        for frame_pose in poses:
            analyzer.analyze_frame(frame_pose.keypoints, frame_pose.frame, fps)
        
        # Detect elements
        elements = analyzer.detect_elements(fps)
        
        # Generate session feedback
        jumps = [e for e in elements if e.type == 'jump']
        spins = [e for e in elements if e.type == 'spin']
        
        session_feedback = []
        if jumps:
            avg_score = sum(j.score for j in jumps) / len(jumps)
            session_feedback.append(f"Jumps: {len(jumps)} detected, average score {avg_score:.1f}")
        if spins:
            avg_score = sum(s.score for s in spins) / len(spins)
            session_feedback.append(f"Spins: {len(spins)} detected, average score {avg_score:.1f}")
        
        if not elements:
            session_feedback.append("No elements detected. Try recording a longer clip with jumps or spins.")
        else:
            session_feedback.append(f"Total elements: {len(elements)}")
        
        return AnalysisResult(
            success=True,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=total_frames / fps if fps > 0 else 0,
            elements=elements,
            session_feedback=session_feedback,
            poses=poses if include_poses else None
        )
        
    finally:
        # Cleanup temp file
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
