"""
IceAnalysis API v2 — ONNX Runtime + YOLOv8-Pose Figure Skating Analysis
Lightweight: 13MB ONNX model, no PyTorch dependency.
"""

import os
import tempfile
import math
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import onnxruntime as ort

app = FastAPI(
    title="IceAnalysis API",
    description="AI-powered figure skating video analysis using YOLOv8-Pose (ONNX)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── YOLO-Pose ONNX Model ────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.onnx")
# Fallback: check /app/ (Docker workdir)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/app/yolov8n-pose.onnx"
_session = None

print(f"[STARTUP] Model path: {MODEL_PATH}, exists: {os.path.exists(MODEL_PATH)}")
print(f"[STARTUP] onnxruntime version: {ort.__version__}")
print(f"[STARTUP] Working directory: {os.getcwd()}")
print(f"[STARTUP] Files in dir: {os.listdir(os.path.dirname(os.path.abspath(__file__)))[:10]}")

def get_session():
    global _session
    if _session is None:
        print(f"[MODEL] Loading ONNX model from {MODEL_PATH}...")
        _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"[MODEL] Model loaded successfully")
    return _session

# COCO keypoint indices (17 keypoints from YOLOv8-Pose)
KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
}


# ── Response Models ──────────────────────────────────────────────────

class SkatingElement(BaseModel):
    type: str
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
    poses: Optional[dict] = None


# ── ONNX Inference Helpers ───────────────────────────────────────────

def preprocess_frame(frame, input_size=640):
    """Preprocess frame for YOLOv8: resize, normalize, HWC→CHW, add batch."""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Pad to square
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # BGR→RGB, HWC→CHW, normalize to [0,1], add batch
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    
    return blob, scale, (0, 0)  # (pad_top, pad_left)


def postprocess_pose(output, scale, conf_threshold=0.25, iou_threshold=0.45):
    """
    Parse YOLOv8-Pose ONNX output.
    Output shape: (1, 56, N) where 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints x,y,conf)
    """
    predictions = output[0]  # Shape: (1, 56, N)
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Shape: (56, N)
    predictions = predictions.T  # Shape: (N, 56)
    
    # Filter by confidence
    scores = predictions[:, 4]
    mask = scores > conf_threshold
    predictions = predictions[mask]
    scores = scores[mask]
    
    if len(predictions) == 0:
        return []
    
    # Extract boxes (x_center, y_center, w, h) → (x1, y1, x2, y2)
    boxes = predictions[:, :4].copy()
    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2
    
    # NMS
    indices = nms(boxes, scores, iou_threshold)
    
    results = []
    for idx in indices:
        box = boxes[idx] / scale  # Scale back to original image coords
        score = float(scores[idx])
        
        # Extract 17 keypoints (x, y, conf each)
        kp_raw = predictions[idx, 5:]  # 51 values = 17 * 3
        keypoints = []
        for k in range(17):
            kx = float(kp_raw[k * 3] / scale)
            ky = float(kp_raw[k * 3 + 1] / scale)
            kc = float(kp_raw[k * 3 + 2])
            keypoints.append([kx, ky, kc])
        
        results.append({
            'box': box.tolist(),
            'score': score,
            'keypoints': np.array(keypoints),
        })
    
    return results


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


# ── Analysis Functions ───────────────────────────────────────────────

def smooth(arr, window=5):
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def angle_between(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cos = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def body_orientation(kps):
    ls, rs = kps[KP['left_shoulder']], kps[KP['right_shoulder']]
    dx = rs[0] - ls[0]
    dy = rs[1] - ls[1]
    return math.degrees(math.atan2(dy, dx))


def extract_frame_data(video_path: str):
    """Extract pose data from every frame using YOLOv8-Pose ONNX."""
    session = get_session()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frame_data = []
    frame_idx = 0
    
    input_name = session.get_inputs()[0].name
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        blob, scale, pad = preprocess_frame(frame)
        
        # Run ONNX inference
        outputs = session.run(None, {input_name: blob})
        
        # Parse results
        detections = postprocess_pose(outputs[0], scale)
        
        # Pick largest person (skater)
        skater_kps = None
        if detections:
            # Sort by box area (largest first)
            detections.sort(key=lambda d: (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1]), reverse=True)
            skater_kps = detections[0]['keypoints']
        
        fd = {
            'frame': frame_idx,
            'time_ms': (frame_idx / fps) * 1000 if fps > 0 else 0,
            'has_pose': skater_kps is not None,
        }
        
        if skater_kps is not None:
            lh, rh = skater_kps[KP['left_hip']], skater_kps[KP['right_hip']]
            la, ra = skater_kps[KP['left_ankle']], skater_kps[KP['right_ankle']]
            ls, rs = skater_kps[KP['left_shoulder']], skater_kps[KP['right_shoulder']]
            lw, rw = skater_kps[KP['left_wrist']], skater_kps[KP['right_wrist']]
            
            fd.update({
                'hip_y': float((lh[1] + rh[1]) / 2),
                'hip_x': float((lh[0] + rh[0]) / 2),
                'ankle_y': float(min(la[1], ra[1])),
                'orientation': float(body_orientation(skater_kps)),
                'knee_angle_l': float(angle_between(
                    skater_kps[KP['left_hip']][:2],
                    skater_kps[KP['left_knee']][:2],
                    skater_kps[KP['left_ankle']][:2]
                )),
                'knee_angle_r': float(angle_between(
                    skater_kps[KP['right_hip']][:2],
                    skater_kps[KP['right_knee']][:2],
                    skater_kps[KP['right_ankle']][:2]
                )),
                'shoulder_width': float(math.sqrt((rs[0]-ls[0])**2 + (rs[1]-ls[1])**2)),
                'wrist_spread': float(math.sqrt((rw[0]-lw[0])**2 + (rw[1]-lw[1])**2)),
                'ankle_diff': float(abs(la[1] - ra[1])),
            })
        
        frame_data.append(fd)
        frame_idx += 1
    
    cap.release()
    return frame_data, fps, total_frames, duration


def detect_elements(frame_data, fps):
    """Detect jumps and spins from pose trajectory."""
    elements = []
    if not frame_data or fps <= 0:
        return elements
    
    # ── Jump Detection ───────────────────────────────────────────
    hip_ys = np.array([fd.get('hip_y', 0) for fd in frame_data], dtype=float)
    smoothed = smooth(hip_ys, window=max(3, int(fps/10)))
    
    window = max(3, int(fps * 0.3))
    min_air_frames = max(3, int(fps * 0.12))
    
    jump_candidates = []
    for i in range(window, len(smoothed) - window):
        local_region = smoothed[i-window:i+window+1]
        if smoothed[i] == np.min(local_region):
            baseline_before = np.mean(smoothed[max(0, i-window*2):i-window]) if i > window*2 else smoothed[0]
            baseline_after = np.mean(smoothed[i+window:min(len(smoothed), i+window*2)]) if i+window*2 < len(smoothed) else smoothed[-1]
            baseline = (baseline_before + baseline_after) / 2
            height = baseline - smoothed[i]
            
            if height > 15:
                takeoff = i
                for j in range(i, max(0, i-window*3), -1):
                    if smoothed[j] >= baseline - 3:
                        takeoff = j
                        break
                
                landing = i
                for j in range(i, min(len(smoothed), i+window*3)):
                    if smoothed[j] >= baseline - 3:
                        landing = j
                        break
                
                jump_candidates.append((takeoff, i, landing, height))
    
    for start, apex_frame, end, height in jump_candidates:
        if end - start >= min_air_frames:
            orientations = [fd.get('orientation', 0) for fd in frame_data[start:end+1]]
            total_rotation = 0
            for j in range(1, len(orientations)):
                diff = orientations[j] - orientations[j-1]
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                total_rotation += abs(diff)
            
            rotations = round(total_rotation / 360 * 2) / 2
            jump_type = classify_jump(frame_data, start, apex_frame, end, rotations)
            air_time_ms = ((end - start) / fps) * 1000
            feedback = generate_jump_feedback(frame_data, start, apex_frame, end, height, rotations, jump_type, fps)
            score = calculate_jump_score(jump_type, rotations)
            
            elements.append(SkatingElement(
                type='jump',
                name=jump_type,
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
                confidence=0.85,
                score=score,
                feedback=feedback,
            ))
    
    # ── Spin Detection ───────────────────────────────────────────
    jump_frames = set()
    for e in elements:
        for f in range(e.start_frame, e.end_frame + 1):
            jump_frames.add(f)
    
    orientations = [fd.get('orientation', 0) for fd in frame_data]
    rotation_speeds = [0]
    for i in range(1, len(orientations)):
        diff = orientations[i] - orientations[i-1]
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        rotation_speeds.append(abs(diff) * fps)
    
    smoothed_speed = smooth(np.array(rotation_speeds), window=max(3, int(fps/5)))
    
    spin_threshold = 120
    spin_start = None
    for i in range(len(smoothed_speed)):
        if i in jump_frames:
            spin_start = None
            continue
        
        if smoothed_speed[i] > spin_threshold and spin_start is None:
            spin_start = i
        elif (smoothed_speed[i] < spin_threshold or i == len(smoothed_speed)-1) and spin_start is not None:
            duration_frames = i - spin_start
            if duration_frames > fps * 0.5:
                total_rot = sum(rotation_speeds[spin_start:i])
                revolutions = total_rot / 360
                avg_speed = np.mean(rotation_speeds[spin_start:i])
                spin_pos = classify_spin_position(frame_data, spin_start, i)
                
                spin_feedback = [
                    f"Detected: {spin_pos} Spin",
                    f"{revolutions:.1f} revolutions at {avg_speed:.0f}°/s",
                ]
                if revolutions >= 6:
                    spin_feedback.append("✅ Great revolution count!")
                if avg_speed > 300:
                    spin_feedback.append("✅ Fast rotation speed")
                elif avg_speed < 180:
                    spin_feedback.append("⚠️ Try pulling arms in tighter for more speed")
                
                spin_score = {'Sit': 1.8, 'Camel': 2.0}.get(spin_pos, 1.5)
                if revolutions >= 6: spin_score += 0.5
                if avg_speed > 300: spin_score += 0.3
                
                elements.append(SkatingElement(
                    type='spin',
                    name=f'{spin_pos} Spin',
                    start_frame=spin_start,
                    end_frame=i,
                    start_time=spin_start / fps,
                    end_time=i / fps,
                    confidence=0.80,
                    score=round(spin_score, 2),
                    feedback=spin_feedback,
                ))
            spin_start = None
    
    return elements


def classify_jump(frame_data, start, apex, end, rotations):
    if rotations % 1 == 0.5:
        return f"{rotation_prefix(rotations)} Axel"
    
    takeoff_start = max(start, apex - 5)
    takeoff_frames = frame_data[takeoff_start:apex]
    ankle_diffs = [fd.get('ankle_diff', 0) for fd in takeoff_frames if fd.get('has_pose')]
    avg_ankle_diff = np.mean(ankle_diffs) if ankle_diffs else 0
    
    prefix = rotation_prefix(rotations)
    if avg_ankle_diff > 20:
        return f"{prefix} Toe Jump"
    return f"{prefix} Edge Jump"


def classify_spin_position(frame_data, start, end):
    knee_angles = []
    for fd in frame_data[start:end]:
        if fd.get('has_pose'):
            ka = min(fd.get('knee_angle_l', 180), fd.get('knee_angle_r', 180))
            knee_angles.append(ka)
    avg_knee = np.mean(knee_angles) if knee_angles else 180
    if avg_knee < 110: return 'Sit'
    elif avg_knee < 140: return 'Camel'
    return 'Upright'


def rotation_prefix(rotations):
    r = int(rotations)
    return {1: 'Single', 2: 'Double', 3: 'Triple'}.get(r, 'Quad' if r >= 4 else 'Single')


def calculate_jump_score(jump_type, rotations):
    base_values = {
        1: {'Toe': 0.4, 'Edge': 0.5, 'Axel': 1.1},
        2: {'Toe': 1.3, 'Edge': 1.7, 'Axel': 3.3},
        3: {'Toe': 4.2, 'Edge': 4.9, 'Axel': 8.0},
        4: {'Toe': 9.5, 'Edge': 10.5, 'Axel': 12.5},
    }
    r = max(1, min(4, int(rotations)))
    if 'Axel' in jump_type: return base_values[r]['Axel']
    elif 'Toe' in jump_type: return base_values[r]['Toe']
    return base_values[r]['Edge']


def generate_jump_feedback(frame_data, start, apex, end, height, rotations, jump_type, fps):
    feedback = [f"Detected: {jump_type} ({rotations} rotations)"]
    air_time_s = (end - start) / fps
    feedback.append(f"Air time: ~{air_time_s:.2f}s")
    
    if height > 40:
        feedback.append("✅ Excellent height!")
    elif height > 20:
        feedback.append("✅ Good height")
    else:
        feedback.append("⚠️ Work on getting more height")
    
    landing_frames = frame_data[max(0, end-3):end+1]
    for fd in landing_frames:
        if fd.get('has_pose'):
            knee = min(fd.get('knee_angle_l', 180), fd.get('knee_angle_r', 180))
            if knee < 150:
                feedback.append("✅ Good knee bend on landing")
            else:
                feedback.append("⚠️ Bend knees more on landing")
            break
    
    return feedback


# ── API Endpoints ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Preload model on startup to catch errors early."""
    try:
        session = get_session()
        inp = session.get_inputs()[0]
        print(f"[STARTUP] Model ready: input={inp.name} shape={inp.shape}")
    except Exception as e:
        print(f"[STARTUP] WARNING: Model load failed: {e}")
        # Don't crash — let health check report the issue

@app.get("/")
async def root():
    return {
        "service": "IceAnalysis API",
        "version": "2.0.0",
        "engine": "yolov8-pose-onnx",
        "status": "running",
        "endpoints": {
            "POST /analyze": "Upload video for skating analysis",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "yolov8-pose-onnx"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_video(video: UploadFile = File(...)):
    """Analyze a figure skating video using YOLOv8-Pose (ONNX Runtime)."""
    
    suffix = '.mp4'
    if video.filename and video.filename.lower().endswith('.mov'):
        suffix = '.mov'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        frame_data, fps, total_frames, duration = extract_frame_data(tmp_path)
        poses_detected = sum(1 for fd in frame_data if fd['has_pose'])
        
        if poses_detected < 5:
            return AnalysisResult(
                success=True,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration,
                elements=[],
                session_feedback=[
                    f"Only detected skater in {poses_detected}/{total_frames} frames.",
                    "Make sure the full body is visible with good lighting.",
                ],
                poses={'detection_rate': f"{poses_detected}/{total_frames}"},
            )
        
        elements = detect_elements(frame_data, fps)
        
        session_feedback = []
        for e in elements:
            icon = "🏃" if e.type == 'jump' else "🔄"
            session_feedback.append(f"{icon} {e.name} (score: {e.score})")
        
        if not elements:
            session_feedback.append("No elements detected. Try recording a longer clip with jumps or spins.")
            session_feedback.append(f"(Skater detected in {poses_detected}/{total_frames} frames)")
        else:
            total_score = sum(e.score for e in elements)
            session_feedback.append(f"Total elements: {len(elements)}, Combined score: {total_score:.1f}")
        
        return AnalysisResult(
            success=True,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            elements=elements,
            session_feedback=session_feedback,
            poses={
                'detection_rate': f"{poses_detected}/{total_frames}",
                'engine': 'yolov8-pose-onnx',
            },
        )
    
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
