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
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        print(f"[MODEL] Model loaded successfully (multi-thread mode)")
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
    metrics: Optional[dict] = None

class KeyframeData(BaseModel):
    frame: int
    time_sec: float
    keypoints: List[List[float]]  # [[x, y, confidence], ...] normalized 0-1
    box: List[float]  # [x1, y1, x2, y2] normalized 0-1

class RotationPoint(BaseModel):
    time_sec: float
    velocity_dps: float  # degrees per second

class AnalysisResult(BaseModel):
    success: bool
    total_frames: int
    fps: float
    duration_seconds: float
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    elements: List[SkatingElement]
    session_feedback: List[str]
    poses: Optional[dict] = None
    keyframes: Optional[List[KeyframeData]] = None
    rotation_velocity: Optional[List[RotationPoint]] = None


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
    """Extract pose data from video using YOLOv8-Pose ONNX.
    
    Memory-optimized for free-tier hosting (512MB RAM):
    - Subsamples frames (every Nth) to reduce inference count
    - Uses 320x320 input instead of 640x640 (4x less memory)
    - Explicit memory cleanup between frames
    """
    import gc
    
    session = get_session()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Process every frame for short clips (smooth skeleton overlay).
    # For longer videos, subsample to keep response time reasonable.
    if duration <= 15:
        step = 1  # Every frame for clips under 15s
    elif duration <= 30:
        step = 2  # Every other frame for 15-30s
    else:
        target_frames = 60
        step = max(1, total_frames // target_frames)
    
    # Also cap video duration — reject videos over 60 seconds
    if duration > 60:
        raise HTTPException(status_code=400, detail=f"Video too long ({duration:.0f}s). Please upload clips under 60 seconds.")
    
    print(f"[ANALYZE] {total_frames} frames @ {fps:.1f}fps, step={step}, ~{total_frames//step} will be processed")
    
    frame_data = []
    frame_idx = 0
    
    input_name = session.get_inputs()[0].name
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        
        # Downscale large frames before ONNX preprocessing to save memory
        h, w = frame.shape[:2]
        if max(h, w) > 1080:
            downscale = 1080 / max(h, w)
            frame = cv2.resize(frame, (int(w * downscale), int(h * downscale)))
        
        # Preprocess (model requires 640x640 fixed input)
        blob, scale, pad = preprocess_frame(frame, input_size=640)
        
        # Run ONNX inference
        outputs = session.run(None, {input_name: blob})
        
        # Parse results
        detections = postprocess_pose(outputs[0], scale)
        
        # Free intermediate memory immediately
        del blob, outputs
        
        # Pick largest person (skater)
        skater_kps = None
        skater_box = None
        if detections:
            detections.sort(key=lambda d: (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1]), reverse=True)
            skater_kps = detections[0]['keypoints']
            skater_box = detections[0]['box']
        
        del detections
        
        # Get original frame dimensions for normalization
        frame_h, frame_w = frame.shape[:2]
        
        fd = {
            'frame': frame_idx,
            'time_ms': (frame_idx / fps) * 1000 if fps > 0 else 0,
            'has_pose': skater_kps is not None,
            '_real': True,  # Mark as actually processed (not interpolated)
        }
        
        # Store raw keypoints and box for skeleton overlay (normalized 0-1)
        if skater_kps is not None and skater_box is not None:
            norm_kps = []
            for k in range(17):
                kx = float(skater_kps[k][0]) / frame_w
                ky = float(skater_kps[k][1]) / frame_h
                kc = float(skater_kps[k][2])
                norm_kps.append([round(kx, 4), round(ky, 4), round(kc, 4)])
            norm_box = [
                round(float(skater_box[0]) / frame_w, 4),
                round(float(skater_box[1]) / frame_h, 4),
                round(float(skater_box[2]) / frame_w, 4),
                round(float(skater_box[3]) / frame_h, 4),
            ]
            fd['_raw_keypoints'] = norm_kps
            fd['_raw_box'] = norm_box
        
        if skater_kps is not None:
            lh, rh = skater_kps[KP['left_hip']], skater_kps[KP['right_hip']]
            la, ra = skater_kps[KP['left_ankle']], skater_kps[KP['right_ankle']]
            ls, rs = skater_kps[KP['left_shoulder']], skater_kps[KP['right_shoulder']]
            lw, rw = skater_kps[KP['left_wrist']], skater_kps[KP['right_wrist']]
            nose = skater_kps[KP['nose']]
            
            hip_cx = float((lh[0] + rh[0]) / 2)
            hip_cy = float((lh[1] + rh[1]) / 2)
            
            fd.update({
                'hip_y': hip_cy,
                'hip_x': hip_cx,
                'ankle_y': float(min(la[1], ra[1])),
                'orientation': float(body_orientation(skater_kps)),
                'hip_orientation': float(math.degrees(math.atan2(rh[1]-lh[1], rh[0]-lh[0]))),
                'nose_offset_x': float(nose[0] - hip_cx),
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
        
        del skater_kps
        frame_data.append(fd)
        frame_idx += 1
        
        # Periodic GC to keep memory in check
        if len(frame_data) % 5 == 0:
            gc.collect()
    
    cap.release()
    del cap
    gc.collect()
    
    # Interpolate skipped frames for smoother trajectory analysis
    if step > 1 and len(frame_data) >= 2:
        frame_data = interpolate_frames(frame_data, total_frames, fps)
    
    print(f"[ANALYZE] Processed {len(frame_data)} frames (from {total_frames} total)")
    return frame_data, fps, total_frames, duration, vid_width, vid_height


def interpolate_frames(sampled_data, total_frames, fps):
    """Linearly interpolate between sampled frames for trajectory continuity."""
    if len(sampled_data) < 2:
        return sampled_data
    
    full_data = []
    for i in range(len(sampled_data) - 1):
        cur = sampled_data[i]
        nxt = sampled_data[i + 1]
        full_data.append(cur)
        
        gap = nxt['frame'] - cur['frame']
        if gap <= 1 or not cur.get('has_pose') or not nxt.get('has_pose'):
            continue
        
        # Interpolate intermediate frames
        numeric_keys = ['hip_y', 'hip_x', 'ankle_y', 'orientation', 'hip_orientation',
                       'nose_offset_x', 'knee_angle_l', 'knee_angle_r', 
                       'shoulder_width', 'wrist_spread', 'ankle_diff']
        
        for step in range(1, gap):
            t = step / gap
            interp = {
                'frame': cur['frame'] + step,
                'time_ms': ((cur['frame'] + step) / fps) * 1000 if fps > 0 else 0,
                'has_pose': True,
                '_real': False,  # Interpolated, not from actual ONNX inference
            }
            for key in numeric_keys:
                if key in cur and key in nxt:
                    # Handle angle wrapping for orientation fields
                    if key in ('orientation', 'hip_orientation'):
                        diff = nxt[key] - cur[key]
                        while diff > 180: diff -= 360
                        while diff < -180: diff += 360
                        interp[key] = cur[key] + diff * t
                    else:
                        interp[key] = cur[key] + (nxt[key] - cur[key]) * t
            full_data.append(interp)
    
    full_data.append(sampled_data[-1])
    return full_data


def calculate_jump_metrics(frame_data, start, apex, end, height, rotations, jump_type, fps):
    """Calculate detailed biomechanical metrics for a detected jump."""
    metrics = {}
    
    air_time_s = (end - start) / fps if fps > 0 else 0
    metrics['air_time_s'] = round(air_time_s, 3)
    
    # ── Height relative to standing height ──
    # Estimate standing height from shoulder-to-ankle distance in pre-jump frames
    pre_start = max(0, start - int(fps * 0.5))
    pre_frames = [fd for fd in frame_data[pre_start:start] if fd.get('has_pose')]
    
    # Estimate standing height from pixel-space shoulder-to-ankle distance
    # hip_y, ankle_y are in pixel coordinates; use those consistently
    standing_heights = []
    for fd in pre_frames:
        hip_y = fd.get('hip_y', 0)
        ankle_y = fd.get('ankle_y', 0)
        if hip_y > 0 and ankle_y > 0:
            # hip-to-ankle is roughly 50% of full standing height
            # full standing height ≈ hip_to_ankle * 2
            hip_to_ankle = abs(ankle_y - hip_y)
            if hip_to_ankle > 10:  # sanity check
                standing_heights.append(hip_to_ankle * 2.0)
    
    # If no pre-jump frames, try all frames with pose data
    if not standing_heights:
        for fd in frame_data:
            if fd.get('has_pose'):
                hip_y = fd.get('hip_y', 0)
                ankle_y = fd.get('ankle_y', 0)
                if hip_y > 0 and ankle_y > 0:
                    hip_to_ankle = abs(ankle_y - hip_y)
                    if hip_to_ankle > 10:
                        standing_heights.append(hip_to_ankle * 2.0)
    
    standing_height = np.mean(standing_heights) if standing_heights else 0
    
    if standing_height > 0 and height > 0:
        height_relative = round(height / standing_height, 2)
    else:
        height_relative = 0
    metrics['height_relative'] = height_relative
    
    # ── Rotation speed (degrees per second) ──
    # Target rotation for jump type
    target_rotations_map = {
        'Single': 360, 'Double': 720, 'Triple': 1080, 'Quad': 1440,
    }
    # Axels add 180° (half extra rotation)
    is_axel = 'Axel' in jump_type
    total_target = 360  # default
    for prefix, deg in target_rotations_map.items():
        if prefix in jump_type:
            total_target = deg + (180 if is_axel else 0)
            break
    
    total_rotation_deg = rotations * 360
    rotation_speed_dps = round(total_rotation_deg / air_time_s) if air_time_s > 0 else 0
    metrics['rotation_speed_dps'] = rotation_speed_dps
    
    # ── Peak rotation speed ──
    orientations = [fd.get('orientation', 0) for fd in frame_data[start:end+1]]
    frame_rot_speeds = []
    for j in range(1, len(orientations)):
        diff = orientations[j] - orientations[j-1]
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        speed = abs(diff) * fps  # degrees per second
        frame_rot_speeds.append(speed)
    
    peak_rotation_speed = round(max(frame_rot_speeds)) if frame_rot_speeds else 0
    metrics['peak_rotation_speed_dps'] = peak_rotation_speed
    
    # ── Pre-rotation ──
    pre_rot_window = int(fps * 0.2)  # 0.2s before takeoff
    pre_rot_start = max(0, start - pre_rot_window)
    pre_orientations = [fd.get('orientation', 0) for fd in frame_data[pre_rot_start:start+1]]
    pre_rotation = 0
    for j in range(1, len(pre_orientations)):
        diff = pre_orientations[j] - pre_orientations[j-1]
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        pre_rotation += abs(diff)
    metrics['pre_rotation_deg'] = round(pre_rotation, 1)
    
    # ── Under-rotation ──
    # total_rotation_deg uses the rounded rotations (e.g., 1.5 for Axel = 540°)
    # total_target is the ISU expected rotation (e.g., 540° for Single Axel)
    # If rounded correctly, these should match. Only flag significant mismatches.
    under_rotation = max(0, total_target - total_rotation_deg)
    # Suppress small values — 2D measurement noise is ±90° at minimum
    metrics['under_rotation_deg'] = round(under_rotation, 1) if under_rotation > 45 else 0
    
    # ── Tuck tightness ──
    air_frames = [fd for fd in frame_data[start:end+1] if fd.get('has_pose')]
    wrist_spreads = [fd.get('wrist_spread', 0) for fd in air_frames if fd.get('wrist_spread', 0) > 0]
    shoulder_widths = [fd.get('shoulder_width', 0) for fd in air_frames if fd.get('shoulder_width', 0) > 0]
    
    if wrist_spreads and shoulder_widths:
        avg_wrist = np.mean(wrist_spreads)
        avg_shoulder = np.mean(shoulder_widths)
        tuck = round(avg_wrist / avg_shoulder, 2) if avg_shoulder > 0 else 1.0
    else:
        tuck = 1.0
    metrics['tuck_tightness'] = tuck
    
    # ── Landing knee angle ──
    landing_window = frame_data[max(0, end-3):end+2]
    landing_knee = 180
    for fd in landing_window:
        if fd.get('has_pose'):
            ka = min(fd.get('knee_angle_l', 180), fd.get('knee_angle_r', 180))
            landing_knee = ka
            break
    metrics['landing_knee_angle'] = round(landing_knee, 1)
    
    # ── Entry speed ──
    entry_window = int(fps * 0.3)
    entry_start = max(0, start - entry_window)
    entry_frames = [fd for fd in frame_data[entry_start:start+1] if fd.get('has_pose') and fd.get('hip_x', 0) > 0]
    
    if len(entry_frames) >= 2:
        hip_xs = [fd['hip_x'] for fd in entry_frames]
        total_movement = sum(abs(hip_xs[i] - hip_xs[i-1]) for i in range(1, len(hip_xs)))
        entry_speed_px = round(total_movement / len(hip_xs), 1)
    else:
        entry_speed_px = 0
    metrics['entry_speed_px'] = entry_speed_px
    
    # Classify entry speed qualitatively
    if entry_speed_px > 8:
        metrics['entry_speed'] = 'fast'
    elif entry_speed_px > 4:
        metrics['entry_speed'] = 'moderate'
    else:
        metrics['entry_speed'] = 'slow'
    
    # ── Takeoff lean angle ──
    takeoff_frames = [fd for fd in frame_data[max(0, start-2):start+3] if fd.get('has_pose')]
    takeoff_lean = 0
    if takeoff_frames and '_raw_keypoints' in takeoff_frames[0]:
        kps = takeoff_frames[0]['_raw_keypoints']
        sh_mid_x = (kps[KP['left_shoulder']][0] + kps[KP['right_shoulder']][0]) / 2
        sh_mid_y = (kps[KP['left_shoulder']][1] + kps[KP['right_shoulder']][1]) / 2
        hip_mid_x = (kps[KP['left_hip']][0] + kps[KP['right_hip']][0]) / 2
        hip_mid_y = (kps[KP['left_hip']][1] + kps[KP['right_hip']][1]) / 2
        
        dx = sh_mid_x - hip_mid_x
        dy = sh_mid_y - hip_mid_y  # In image coords, Y increases downward
        # Angle from vertical: vertical is dy < 0 (shoulder above hip)
        if abs(dy) > 0.001:
            takeoff_lean = abs(math.degrees(math.atan2(dx, -dy)))
    elif takeoff_frames:
        # Fallback: use hip_x movement as rough proxy for lean
        fd = takeoff_frames[0]
        hip_x = fd.get('hip_x', 0)
        hip_y = fd.get('hip_y', 0)
        # Very rough estimate
        takeoff_lean = 5  # default small lean
    
    metrics['takeoff_lean_deg'] = round(takeoff_lean, 1)
    
    return metrics


def calculate_spin_metrics(frame_data, start, end, fps):
    """Calculate detailed biomechanical metrics for a detected spin."""
    metrics = {}
    
    duration_s = (end - start) / fps if fps > 0 else 0
    
    # ── Rotation data ──
    orientations = [fd.get('orientation', 0) for fd in frame_data[start:end+1]]
    rotation_speeds = []  # degrees per second per frame
    for j in range(1, len(orientations)):
        diff = orientations[j] - orientations[j-1]
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        speed = abs(diff) * fps
        rotation_speeds.append(speed)
    
    total_rot = sum(abs(d) for d in rotation_speeds) / fps if fps > 0 else 0
    revolutions = total_rot / 360
    metrics['revolutions'] = round(revolutions, 1)
    
    # ── Average RPM ──
    avg_dps = np.mean(rotation_speeds) if rotation_speeds else 0
    avg_rpm = round(avg_dps / 6, 1)  # 360°/s = 60 RPM, so dps/6 = RPM
    metrics['avg_rpm'] = avg_rpm
    
    # ── Peak RPM ──
    if rotation_speeds:
        # Smooth a bit to avoid noise spikes
        smoothed_speeds = smooth(np.array(rotation_speeds), window=max(3, int(fps / 6)))
        peak_dps = float(np.max(smoothed_speeds))
        peak_rpm = round(peak_dps / 6, 1)
    else:
        peak_rpm = 0
    metrics['peak_rpm'] = peak_rpm
    
    # ── Centering ──
    hip_xs = [fd.get('hip_x', 0) for fd in frame_data[start:end+1] if fd.get('has_pose')]
    if hip_xs:
        centering = round(max(hip_xs) - min(hip_xs), 1)
    else:
        centering = 0
    metrics['centering_px'] = centering
    
    # ── Speed profile ──
    if len(rotation_speeds) >= 4:
        mid = len(rotation_speeds) // 2
        first_half_avg = np.mean(rotation_speeds[:mid])
        second_half_avg = np.mean(rotation_speeds[mid:])
        ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
        if ratio > 1.15:
            speed_profile = 'accelerating'
        elif ratio < 0.85:
            speed_profile = 'decelerating'
        else:
            speed_profile = 'steady'
    else:
        speed_profile = 'steady'
    metrics['speed_profile'] = speed_profile
    
    # ── Position consistency ──
    knee_angles = []
    for fd in frame_data[start:end+1]:
        if fd.get('has_pose'):
            ka = min(fd.get('knee_angle_l', 180), fd.get('knee_angle_r', 180))
            knee_angles.append(ka)
    
    if len(knee_angles) >= 3:
        position_consistency = round(float(np.std(knee_angles)), 1)
    else:
        position_consistency = 0
    metrics['position_consistency'] = position_consistency
    
    return metrics


def generate_detailed_feedback(element_type, metrics):
    """Generate specific, actionable coaching tips based on detailed metrics."""
    feedback = []
    
    if element_type == 'jump':
        # ── Height ──
        hr = metrics.get('height_relative', 0)
        if hr >= 1.0:
            feedback.append(f"✅ Excellent height ({hr}x body height)")
        elif hr >= 0.7:
            feedback.append(f"✅ Good height ({hr}x body height)")
        elif hr > 0:
            feedback.append(f"⚠️ Work on jump height — only {hr}x body height")
        
        # ── Rotation speed / tuck ──
        # Tuck feedback only meaningful for doubles and above (singles don't fully tuck)
        rspeed = metrics.get('rotation_speed_dps', 0)
        tuck = metrics.get('tuck_tightness', 1.0)
        # Estimate rotation count from rotation speed and air time
        air_t = metrics.get('air_time_s', 0)
        rot_count = max(1, round(rspeed * air_t / 360)) if air_t > 0 and rspeed > 0 else 1
        if rot_count >= 2:
            if tuck < 0.5 and rspeed > 800:
                feedback.append(f"✅ Tight tuck — rotation speed {rspeed}°/s")
            elif tuck >= 0.8:
                feedback.append(f"⚠️ Arms too wide in air — tuck tighter for faster rotation")
            elif rspeed > 600:
                feedback.append(f"✅ Good rotation speed ({rspeed}°/s)")
        
        # Pre-rotation detection removed — 2D shoulder tracking picks up normal
        # skating movement as false pre-rotation. Would need 3D pose or edge
        # detection (blade angle) to measure this reliably.
        
        # ── Under-rotation ──
        under_rot = metrics.get('under_rotation_deg', 0)
        if under_rot > 90:
            feedback.append(f"⚠️ Under-rotated by ~{under_rot:.0f}° — focus on completing the rotation before landing")
        elif under_rot <= 45:
            feedback.append(f"✅ Fully rotated!")
        
        # ── Landing ──
        knee = metrics.get('landing_knee_angle', 180)
        if knee < 140:
            feedback.append(f"✅ Solid landing — good knee bend at {knee:.0f}°")
        elif knee < 155:
            feedback.append(f"✅ Decent landing at {knee:.0f}°")
        else:
            feedback.append(f"⚠️ Stiff landing at {knee:.0f}° — bend knees more to absorb impact")
        
        # ── Entry speed ──
        entry = metrics.get('entry_speed', 'moderate')
        if entry == 'fast':
            feedback.append("✅ Good entry speed")
        elif entry == 'slow':
            feedback.append("⚠️ Slow approach — more speed helps with height")
        
        # ── Takeoff lean ──
        # 2D lean measurement is heavily affected by camera angle.
        # Only flag extreme lean (>35°) to avoid false warnings.
        lean = metrics.get('takeoff_lean_deg', 0)
        if lean > 35:
            feedback.append(f"⚠️ Significant lean at takeoff ({lean:.0f}°) — try to stay more upright")
        elif 5 < lean < 20:
            feedback.append(f"✅ Good takeoff posture")
    
    elif element_type == 'spin':
        # ── Speed ──
        avg_rpm = metrics.get('avg_rpm', 0)
        peak_rpm = metrics.get('peak_rpm', 0)
        if avg_rpm >= 240:
            feedback.append(f"✅ Fast spin at {avg_rpm:.0f} RPM")
        elif avg_rpm >= 150:
            feedback.append(f"Good spin speed at {avg_rpm:.0f} RPM")
        else:
            feedback.append(f"⚠️ Spin speed dropped to {avg_rpm:.0f} RPM — pull arms in")
        
        if peak_rpm > avg_rpm * 1.3 and peak_rpm > 200:
            feedback.append(f"Peak speed: {peak_rpm:.0f} RPM")
        
        # ── Centering ──
        centering = metrics.get('centering_px', 0)
        if centering < 50:
            feedback.append("✅ Well-centered spin")
        elif centering < 100:
            feedback.append("Decent centering — minor travel")
        else:
            feedback.append("⚠️ Spin traveling — try to stay centered over skating foot")
        
        # ── Speed profile ──
        profile = metrics.get('speed_profile', 'steady')
        if profile == 'accelerating':
            feedback.append("✅ Good acceleration through the spin")
        elif profile == 'decelerating':
            feedback.append("⚠️ Speed drops off — try to maintain or increase speed")
        else:
            feedback.append("Steady speed throughout")
        
        # ── Position consistency ──
        consistency = metrics.get('position_consistency', 0)
        if consistency < 8:
            feedback.append("✅ Stable position throughout")
        elif consistency < 15:
            feedback.append("Mostly stable position")
        else:
            feedback.append("⚠️ Position wavering — focus on holding the position")
    
    return feedback


def count_rotation_robust(frame_data, start, end, fps):
    """Multi-method robust rotation counting for jumps.
    
    Problem: Single-method shoulder atan2 tracking undercounts rotation by
    20-30% due to YOLO keypoint left/right assignment flips during fast
    rotation. This causes misclassification (Triple Axel → Triple Toe Loop).
    
    Solution: Use multiple independent methods and take the maximum.
    
    Method A: Median-cleaned shoulder orientation tracking
      - Replace outlier angular diffs (>2σ from median) with median
      - Handles L/R keypoint swaps that create sudden ~180° jumps
    
    Method B: Directed rotation (dominant direction only)
      - Sum only CW or CCW diffs (whichever dominates)
      - Keypoint swaps appear as opposite-direction diffs → filtered out
    
    Method C: Hip orientation tracking (same as A but from hip keypoints)
      - Independent signal from different body landmarks
    
    Method D: Shoulder-width oscillation counting
      - Shoulder width oscillates 2x per rotation (frontal/profile views)
      - Doesn't depend on left/right labeling at all
    """
    # Extend window slightly to capture rotation at takeoff/landing boundaries
    rot_start = max(0, start - 3)
    rot_end = min(len(frame_data) - 1, end + 3)
    
    # Collect orientation data from real (non-interpolated) frames
    shoulder_angles = []
    hip_angles = []
    shoulder_widths = []
    
    for fd in frame_data[rot_start:rot_end + 1]:
        if fd.get('_real', True) and fd.get('has_pose'):
            shoulder_angles.append(fd.get('orientation', 0))
            hip_angles.append(fd.get('hip_orientation', fd.get('orientation', 0)))
            shoulder_widths.append(fd.get('shoulder_width', 0))
    
    # Fallback: use all frames if not enough real ones
    if len(shoulder_angles) < 3:
        shoulder_angles = [fd.get('orientation', 0) for fd in frame_data[rot_start:rot_end + 1] if fd.get('has_pose')]
        hip_angles = [fd.get('hip_orientation', fd.get('orientation', 0)) for fd in frame_data[rot_start:rot_end + 1] if fd.get('has_pose')]
        shoulder_widths = [fd.get('shoulder_width', 0) for fd in frame_data[rot_start:rot_end + 1] if fd.get('has_pose')]
    
    if len(shoulder_angles) < 3:
        return 0.0, {}
    
    def compute_angle_diffs(angles):
        """Compute wrapped frame-to-frame angular differences."""
        diffs = []
        for j in range(1, len(angles)):
            d = angles[j] - angles[j - 1]
            while d > 180: d -= 360
            while d < -180: d += 360
            diffs.append(d)
        return diffs
    
    def method_cleaned(diffs):
        """Median-cleaned cumulative rotation. Replaces outlier diffs with median."""
        if not diffs:
            return 0.0
        median_d = float(np.median(diffs))
        std_d = float(np.std(diffs)) if len(diffs) > 2 else abs(median_d) * 0.5
        threshold = max(2 * std_d, 40)  # At least 40° deviation to be outlier
        
        total = 0.0
        for d in diffs:
            if abs(d - median_d) > threshold:
                # Outlier — likely a keypoint swap. Use median instead.
                total += abs(median_d)
            else:
                total += abs(d)
        return total / 360
    
    def method_directed(diffs):
        """Directed rotation: only accumulate in dominant direction."""
        if not diffs:
            return 0.0
        total_pos = sum(d for d in diffs if d > 0)
        total_neg = sum(abs(d) for d in diffs if d < 0)
        return max(total_pos, total_neg) / 360
    
    def method_oscillation(widths):
        """Count shoulder-width oscillation peaks. 2 zero-crossings per rotation."""
        if len(widths) < 5:
            return 0.0
        sw = np.array(widths, dtype=float)
        sw_smooth = smooth(sw, window=max(3, len(sw) // 8))
        sw_centered = sw_smooth - np.mean(sw_smooth)
        
        # Count zero crossings
        crossings = 0
        for j in range(1, len(sw_centered)):
            if sw_centered[j] * sw_centered[j - 1] < 0:
                crossings += 1
        return crossings / 2.0
    
    # ── Compute all methods over EXTENDED window (±3 frames) ──
    sh_diffs = compute_angle_diffs(shoulder_angles)
    hp_diffs = compute_angle_diffs(hip_angles)
    
    a_shoulder_cleaned = method_cleaned(sh_diffs)
    b_shoulder_directed = method_directed(sh_diffs)
    c_hip_cleaned = method_cleaned(hp_diffs)
    d_hip_directed = method_directed(hp_diffs)
    e_oscillation = method_oscillation(shoulder_widths)
    simple_abs_ext = sum(abs(d) for d in sh_diffs) / 360
    
    # ── Also compute over ORIGINAL window (no extension) ──
    # The extended window can capture approach/exit counter-rotation that
    # confuses the directed method. Original window = pure airborne phase.
    orig_sh_angles = []
    orig_hp_angles = []
    for fd in frame_data[start:end + 1]:
        if fd.get('_real', True) and fd.get('has_pose'):
            orig_sh_angles.append(fd.get('orientation', 0))
            orig_hp_angles.append(fd.get('hip_orientation', fd.get('orientation', 0)))
    
    if len(orig_sh_angles) >= 3:
        orig_sh_diffs = compute_angle_diffs(orig_sh_angles)
        orig_hp_diffs = compute_angle_diffs(orig_hp_angles)
        f_orig_sh_directed = method_directed(orig_sh_diffs)
        g_orig_hp_directed = method_directed(orig_hp_diffs)
        simple_abs_orig = sum(abs(d) for d in orig_sh_diffs) / 360
    else:
        f_orig_sh_directed = 0
        g_orig_hp_directed = 0
        simple_abs_orig = 0
    
    # ── Choose best rotation estimate ──
    # Use max across multiple methods and windows:
    # - Directed methods: filter L/R keypoint swap noise (best for fast rotation)
    # Use ONLY directed methods — they filter L/R keypoint swap noise.
    # simple_abs methods are excluded: they sum ALL angle changes including
    # YOLO keypoint swap artifacts (~180° fake jumps), overcounting by 1-2 rotations
    # on fast spins (e.g. triple toe loop measured as quad).
    candidates = [b_shoulder_directed, d_hip_directed, 
                  f_orig_sh_directed, g_orig_hp_directed]
    raw_rotation = max(candidates)
    
    debug = {
        'simple_abs_ext': round(simple_abs_ext, 2),
        'simple_abs_orig': round(simple_abs_orig, 2),
        'ext_sh_dir': round(b_shoulder_directed, 2),
        'ext_hip_dir': round(d_hip_directed, 2),
        'orig_sh_dir': round(f_orig_sh_directed, 2),
        'orig_hip_dir': round(g_orig_hp_directed, 2),
        'oscillation': round(e_oscillation, 2),
        'chosen_raw': round(raw_rotation, 2),
        'num_frames_ext': len(shoulder_angles),
        'num_frames_orig': len(orig_sh_angles),
    }
    
    print(f"[ROT] ext[sh_dir={b_shoulder_directed:.2f} hip_dir={d_hip_directed:.2f}] "
          f"orig[sh_dir={f_orig_sh_directed:.2f} hip_dir={g_orig_hp_directed:.2f}] "
          f"abs_ext={simple_abs_ext:.2f} abs_orig={simple_abs_orig:.2f} → "
          f"raw={raw_rotation:.2f} ({len(shoulder_angles)}/{len(orig_sh_angles)} frames)")
    
    return raw_rotation, debug


def detect_elements(frame_data, fps):
    """Detect jumps and spins from pose trajectory.
    
    Jump detection: hip-Y trajectory local minima (= airborne apex).
    Spin detection: sustained high rotation speed while stationary.
    
    Key thresholds tuned against real skating videos:
    - Jumps need significant height (>25px) AND duration (>0.2s)
    - Jumps must have at least 0.5 rotations (filter noise)
    - Spins need sustained rotation (>1.5s) while staying in place
    - Merge nearby jump candidates that are part of the same jump
    """
    elements = []
    if not frame_data or fps <= 0:
        return elements
    
    # ── Jump Detection ───────────────────────────────────────────
    hip_ys = np.array([fd.get('hip_y', 0) for fd in frame_data], dtype=float)
    smoothed = smooth(hip_ys, window=max(3, int(fps / 8)))
    
    window = max(4, int(fps * 0.35))
    min_air_frames = max(4, int(fps * 0.2))  # At least 0.2s airborne
    min_height = 25  # Minimum pixel height delta (was 15 — too low)
    
    jump_candidates = []
    for i in range(window, len(smoothed) - window):
        local_region = smoothed[i - window:i + window + 1]
        if smoothed[i] == np.min(local_region):
            # Use separate baselines for before/after (skater is moving)
            before_start = max(0, i - window * 3)
            before_end = max(0, i - window)
            after_start = min(len(smoothed), i + window)
            after_end = min(len(smoothed), i + window * 3)
            
            baseline_before = np.mean(smoothed[before_start:before_end]) if before_end > before_start else smoothed[0]
            baseline_after = np.mean(smoothed[after_start:after_end]) if after_end > after_start else smoothed[-1]
            height = max(baseline_before, baseline_after) - smoothed[i]
            
            if height > min_height:
                # Find takeoff: go backward from apex, find the local max 
                # (highest hip Y = skater on ice before jumping)
                takeoff = i
                for j in range(i - 1, max(0, i - window * 4), -1):
                    if smoothed[j] >= baseline_before - 10:
                        takeoff = j
                        break
                    # Also stop if hip starts going down again (we passed the takeoff)
                    if j < i - 2 and smoothed[j] < smoothed[j + 1]:
                        takeoff = j + 1
                        break
                
                # Find landing: go forward from apex, find where hip returns 
                # to post-jump baseline
                landing = i
                for j in range(i + 1, min(len(smoothed), i + window * 4)):
                    if smoothed[j] >= baseline_after - 10:
                        landing = j
                        break
                    # Also stop if hip starts going down again (overshot)
                    if j > i + 2 and smoothed[j] > smoothed[j - 1] and smoothed[j] > smoothed[i] + height * 0.6:
                        landing = j
                        break
                
                jump_candidates.append((takeoff, i, landing, height))
    
    # Merge overlapping/nearby jump candidates (same jump detected multiple times)
    merged = []
    for cand in sorted(jump_candidates, key=lambda c: c[1]):
        if merged and cand[0] <= merged[-1][2] + int(fps * 0.15):
            # Overlaps with previous — keep the one with more height
            prev = merged[-1]
            if cand[3] > prev[3]:
                merged[-1] = (min(prev[0], cand[0]), cand[1], max(prev[2], cand[2]), cand[3])
            else:
                merged[-1] = (min(prev[0], cand[0]), prev[1], max(prev[2], cand[2]), prev[3])
        else:
            merged.append(cand)
    
    for start, apex_frame, end, height in merged:
        air_frames = end - start
        if air_frames < min_air_frames:
            print(f"[JUMP] Rejected apex@{apex_frame}: too short ({air_frames} frames)")
            continue
        
        # Count rotation using robust multi-method approach
        # Extended ±3 frame window captures takeoff/landing rotation;
        # directed method filters noise → no correction factor needed.
        raw_rotations, rot_debug = count_rotation_robust(frame_data, start, end, fps)
        
        # Round to nearest half rotation (for Axel detection)
        rotations = round(raw_rotations * 2) / 2
        
        # Filter: must have at least 0.5 rotations to be a real jump
        if rotations < 0.5:
            print(f"[JUMP] Rejected apex@{apex_frame}: only {rotations} rotations")
            continue
        
        jump_type, rotations = classify_jump(frame_data, start, apex_frame, end, rotations)
        score = calculate_jump_score(jump_type, rotations)
        
        air_time_s = air_frames / fps
        confidence = min(0.95, 0.6 + (height / 100) + (rotations * 0.1))
        
        # Calculate detailed metrics
        jump_metrics = calculate_jump_metrics(frame_data, start, apex_frame, end, height, rotations, jump_type, fps)
        
        # Generate detailed coaching feedback from metrics
        feedback = generate_jump_feedback(frame_data, start, apex_frame, end, height, rotations, jump_type, fps)
        detailed_tips = generate_detailed_feedback('jump', jump_metrics)
        # Merge: keep basic feedback first, then add detailed tips that aren't redundant
        existing_text = ' '.join(feedback).lower()
        for tip in detailed_tips:
            tip_lower = tip.lower()
            # Avoid duplicating the same concept
            if ('height' in tip_lower and 'height' in existing_text):
                continue
            if ('knee' in tip_lower and 'knee' in existing_text):
                continue
            feedback.append(tip)
        
        print(f"[JUMP] ✅ {jump_type}: {rotations} rot (raw: {raw_rotations:.2f}), "
              f"height={height:.0f}px, air={air_time_s:.2f}s, frames {start}-{end}, rot_debug={rot_debug}")
        
        elements.append(SkatingElement(
            type='jump',
            name=jump_type,
            start_frame=start,
            end_frame=end,
            start_time=start / fps,
            end_time=end / fps,
            confidence=round(confidence, 2),
            score=score,
            feedback=feedback,
            metrics=jump_metrics,
        ))
    
    # ── Spin Detection ───────────────────────────────────────────
    # Exclude jump frames + buffer around them (jumps have fast rotation too)
    jump_frames = set()
    for e in elements:
        buffer = int(fps * 0.3)  # 0.3s buffer around each jump
        for f in range(max(0, e.start_frame - buffer), min(len(frame_data), e.end_frame + buffer + 1)):
            jump_frames.add(f)
    
    orientations = [fd.get('orientation', 0) for fd in frame_data]
    hip_xs = [fd.get('hip_x', 0) for fd in frame_data]
    
    rotation_speeds = [0.0]
    for i in range(1, len(orientations)):
        diff = orientations[i] - orientations[i - 1]
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        rotation_speeds.append(abs(diff) * fps)
    
    smoothed_speed = smooth(np.array(rotation_speeds), window=max(3, int(fps / 4)))
    
    spin_threshold = 150  # Higher threshold (was 120)
    min_spin_duration = fps * 1.5  # At least 1.5 seconds (was 0.5 — way too low)
    
    spin_start = None
    for i in range(len(smoothed_speed)):
        if i in jump_frames:
            spin_start = None
            continue
        
        if smoothed_speed[i] > spin_threshold and spin_start is None:
            spin_start = i
        elif (smoothed_speed[i] < spin_threshold or i == len(smoothed_speed) - 1) and spin_start is not None:
            duration_frames = i - spin_start
            
            if duration_frames >= min_spin_duration:
                # Check skater stays roughly in place (spin = stationary rotation)
                spin_hip_xs = [hip_xs[f] for f in range(spin_start, i) if f < len(hip_xs)]
                hip_x_range = max(spin_hip_xs) - min(spin_hip_xs) if spin_hip_xs else 999
                
                # If skater moved too much horizontally, it's not a spin
                if hip_x_range > 150:
                    print(f"[SPIN] Rejected frames {spin_start}-{i}: too much lateral movement ({hip_x_range:.0f}px)")
                    spin_start = None
                    continue
                
                total_rot = sum(rotation_speeds[spin_start:i])
                revolutions = total_rot / 360
                avg_speed = np.mean(rotation_speeds[spin_start:i])
                spin_pos = classify_spin_position(frame_data, spin_start, i)
                
                # Calculate detailed spin metrics
                spin_metrics = calculate_spin_metrics(frame_data, spin_start, i, fps)
                
                spin_feedback = [
                    f"Detected: {spin_pos} Spin",
                    f"{spin_metrics.get('revolutions', revolutions):.1f} revolutions at {spin_metrics.get('avg_rpm', 0):.0f} RPM",
                ]
                
                # Generate detailed coaching tips from metrics
                detailed_spin_tips = generate_detailed_feedback('spin', spin_metrics)
                spin_feedback.extend(detailed_spin_tips)
                
                spin_score = calculate_spin_score(spin_pos, revolutions, avg_speed)
                
                confidence = min(0.90, 0.5 + (revolutions / 20) + (0.1 if hip_x_range < 80 else 0))
                
                print(f"[SPIN] ✅ {spin_pos} Spin: {revolutions:.1f} rev, {avg_speed:.0f}°/s, frames {spin_start}-{i}")
                
                elements.append(SkatingElement(
                    type='spin',
                    name=f'{spin_pos} Spin',
                    start_frame=spin_start,
                    end_frame=i,
                    start_time=spin_start / fps,
                    end_time=i / fps,
                    confidence=round(confidence, 2),
                    score=round(spin_score, 2),
                    feedback=spin_feedback,
                    metrics=spin_metrics,
                ))
            else:
                print(f"[SPIN] Rejected frames {spin_start}-{i}: too short ({duration_frames} frames, need {min_spin_duration:.0f})")
            spin_start = None
    
    # Sort by start time
    elements.sort(key=lambda e: e.start_frame)
    return elements


def classify_jump(frame_data, start, apex, end, rotations):
    """Classify jump type using biomechanical analysis of pose keypoints.
    
    Classification hierarchy:
    1. Axel detection — half-rotation count + forward entry
    2. Toe pick detection — ankle height differential at takeoff
    3. Edge jump differentiation (Salchow vs Loop) — knee bend + leg position
    4. Toe jump differentiation (T vs F vs Lz) — counter-rotation detection
    
    Returns: (jump_name: str, adjusted_rotations: float)
    The adjusted_rotations may differ from input if non-Axel rounding is applied.
    """
    reasons = []  # Track classification reasoning
    
    # ── Step 1: Axel Detection ──────────────────────────────────
    # Axels are the ONLY jump with forward takeoff (skater faces travel direction).
    # They have fractional rotation (n + 0.5) due to this forward entry.
    # BOTH conditions must be met: half-rotation AND forward entry evidence.
    # Widen the "half rotation" tolerance — 2D measurement noise of ±0.3 is common.
    # A measured 1.0 could really be 1.5 (Axel) if forward entry is strong.
    fractional = rotations % 1.0
    is_half_rotation = (0.1 <= fractional <= 0.9)
    # "Borderline integer" — close enough to .0 that it COULD be a half-rotation
    # if forward entry is very strong (rotation was undercounted by ~0.5)
    is_borderline_integer = (fractional < 0.1 or fractional > 0.9)
    
    # Check for forward entry: nose/body faces the travel direction at takeoff
    # For backward jumps (all others), the skater's back faces travel direction
    forward_entry = False
    forward_score = 0  # Confidence in forward entry (0-1)
    
    if start > 3:
        pre_takeoff = frame_data[max(0, start - 15):start]
        pre_hip_xs = [fd.get('hip_x', 0) for fd in pre_takeoff if fd.get('has_pose')]
        pre_nose_offsets = [fd.get('nose_offset_x', 0) for fd in pre_takeoff if fd.get('has_pose') and fd.get('nose_offset_x') is not None]
        
        if len(pre_hip_xs) >= 3:
            # Travel direction from hip movement
            hip_dx = pre_hip_xs[-1] - pre_hip_xs[0]
            travel_dir = 1 if hip_dx > 0 else -1  # +1 = moving right, -1 = moving left
            
            # Forward entry check: nose is AHEAD of hips in travel direction
            # For forward skating, nose_offset_x has same sign as travel direction
            # For backward skating, nose is BEHIND (opposite sign or near zero)
            if pre_nose_offsets and abs(hip_dx) > 5:
                avg_nose_offset = np.mean(pre_nose_offsets)
                nose_ahead = (avg_nose_offset * travel_dir) > 0
                nose_magnitude = abs(avg_nose_offset)
                
                # Strong forward signal: nose clearly ahead of hips in travel direction
                if nose_ahead and nose_magnitude > 3:
                    forward_score = min(1.0, nose_magnitude / 20)
                    forward_entry = forward_score > 0.2
                
                reasons.append(f"hip_dx={hip_dx:.1f}, nose_offset={avg_nose_offset:.1f}, fwd_score={forward_score:.2f}")
            else:
                reasons.append(f"hip_dx={hip_dx:.1f}, no nose data")
    
    # Axel requires BOTH half-rotation AND forward entry evidence
    if is_half_rotation and forward_entry:
        axel_rotations = round(rotations * 2) / 2
        if axel_rotations % 1.0 == 0.5:
            reasons.append(f"Axel: half-rot + forward entry → {axel_rotations}")
            print(f"[CLASSIFY] {' | '.join(reasons)}")
            return f"{rotation_prefix(axel_rotations)} Axel", axel_rotations
    
    # Fallback: borderline integer rotation + STRONG forward entry → still Axel
    # This catches cases where 2D measurement undercounted by ~0.5 rotation
    # (e.g., measured 1.0 but was really 1.5 single Axel)
    if is_borderline_integer and forward_entry and forward_score >= 0.4:
        axel_rotations = round(rotations) + 0.5
        reasons.append(f"Borderline Axel: rot={rotations:.2f} but strong fwd entry (score={forward_score:.2f}) → {axel_rotations}")
        print(f"[CLASSIFY] {' | '.join(reasons)}")
        return f"{rotation_prefix(axel_rotations)} Axel", axel_rotations
    
    # If half-rotation detected but NO forward entry → not an Axel.
    # Round to nearest integer instead (measurement noise caused the .5)
    # Use int(x + 0.5) instead of round() to avoid Python banker's rounding
    # (round(2.5) = 2 in Python, but we want 3 — benefit of the doubt for skater)
    if is_half_rotation and not forward_entry:
        rotations = int(rotations + 0.5)
        if rotations < 1: rotations = 1
        reasons.append(f"Half-rot {fractional:.1f} but backward entry → rounded to {rotations}")
    
    # ── Step 2: Toe Pick Detection ──────────────────────────────
    # Toe pick jumps have one ankle significantly higher than the other
    # at takeoff (picking foot plants toe in ice behind).
    # The toe pick happens ON THE GROUND before the skater becomes airborne,
    # so look at frames BEFORE the detected takeoff, not just during flight.
    takeoff_window = max(5, int((apex - start) * 0.8))
    # Extend back BEFORE the detected start to capture the ground-level toe pick
    pre_takeoff_start = max(0, start - 5)  # ~5 frames before detected takeoff
    takeoff_frames = frame_data[pre_takeoff_start:apex]
    
    ankle_diffs = [fd.get('ankle_diff', 0) for fd in takeoff_frames if fd.get('has_pose')]
    avg_ankle_diff = np.mean(ankle_diffs) if ankle_diffs else 0
    max_ankle_diff = max(ankle_diffs) if ankle_diffs else 0
    
    # Lower threshold — 2D camera angle can reduce apparent ankle height diff
    TOE_PICK_THRESHOLD = 15  # pixels — ankle height difference indicating toe assist
    is_toe_jump = avg_ankle_diff > TOE_PICK_THRESHOLD or max_ankle_diff > TOE_PICK_THRESHOLD * 2
    
    reasons.append(f"ankle_diff avg={avg_ankle_diff:.1f} max={max_ankle_diff:.1f} → {'toe' if is_toe_jump else 'edge'}")
    
    prefix = rotation_prefix(rotations)
    
    if not is_toe_jump:
        # ── Step 3: Edge Jump Classification (Salchow vs Loop) ──
        # Loop: crossed legs, deeper knee bend at takeoff
        # Salchow: free leg swings, more upright at takeoff
        knee_angles = []
        ankle_spreads = []
        for fd in takeoff_frames:
            if fd.get('has_pose'):
                ka = min(fd.get('knee_angle_l', 180), fd.get('knee_angle_r', 180))
                knee_angles.append(ka)
                # Check ankle x-spread (Loop has ankles close together = crossed)
                # We approximate from ankle_diff — but really we need x-difference
                # For now, use knee angle as primary differentiator
        
        avg_knee = np.mean(knee_angles) if knee_angles else 180
        
        LOOP_KNEE_THRESHOLD = 130  # degrees — Loop has deeper bend
        
        if avg_knee < LOOP_KNEE_THRESHOLD:
            reasons.append(f"Edge jump: knee_avg={avg_knee:.1f}° < {LOOP_KNEE_THRESHOLD} → Loop")
            print(f"[CLASSIFY] {' | '.join(reasons)}")
            return f"{prefix} Loop", rotations
        else:
            reasons.append(f"Edge jump: knee_avg={avg_knee:.1f}° ≥ {LOOP_KNEE_THRESHOLD} → Salchow")
            print(f"[CLASSIFY] {' | '.join(reasons)}")
            return f"{prefix} Salchow", rotations
    
    # ── Step 4: Toe Jump Classification (Toe Loop vs Flip vs Lutz) ──
    # Key differentiator: Lutz has COUNTER-ROTATION before takeoff
    # (body rotates opposite to jump direction during approach)
    
    # Measure orientation change in approach (pre-takeoff) vs during jump
    pre_window = min(15, max(5, start))  # frames before takeoff to analyze
    pre_start = max(0, start - pre_window)
    
    pre_orientations = [fd.get('orientation', 0) for fd in frame_data[pre_start:start] if fd.get('has_pose')]
    jump_orientations = [fd.get('orientation', 0) for fd in frame_data[start:apex] if fd.get('has_pose')]
    
    counter_rotation = False
    if len(pre_orientations) >= 3 and len(jump_orientations) >= 3:
        # Calculate approach rotation direction
        pre_delta = 0
        for j in range(1, len(pre_orientations)):
            d = pre_orientations[j] - pre_orientations[j-1]
            while d > 180: d -= 360
            while d < -180: d += 360
            pre_delta += d
        
        # Calculate jump rotation direction
        jump_delta = 0
        for j in range(1, len(jump_orientations)):
            d = jump_orientations[j] - jump_orientations[j-1]
            while d > 180: d -= 360
            while d < -180: d += 360
            jump_delta += d
        
        # Counter-rotation: approach rotates opposite to jump
        if abs(pre_delta) > 15 and abs(jump_delta) > 15:
            counter_rotation = (pre_delta * jump_delta < 0)  # opposite signs
        
        reasons.append(f"pre_rot={pre_delta:.1f}° jump_rot={jump_delta:.1f}° counter={counter_rotation}")
    
    if counter_rotation:
        reasons.append("Toe + counter-rotation → Lutz")
        print(f"[CLASSIFY] {' | '.join(reasons)}")
        return f"{prefix} Lutz", rotations
    
    # Distinguish Flip vs Toe Loop by ankle differential magnitude
    # Flip typically has larger ankle differential (more aggressive toe pick)
    FLIP_THRESHOLD = 35  # pixels
    if avg_ankle_diff > FLIP_THRESHOLD:
        reasons.append(f"Toe: ankle_diff={avg_ankle_diff:.1f} > {FLIP_THRESHOLD} → Flip")
        print(f"[CLASSIFY] {' | '.join(reasons)}")
        return f"{prefix} Flip", rotations
    
    reasons.append(f"Toe: default → Toe Loop")
    print(f"[CLASSIFY] {' | '.join(reasons)}")
    return f"{prefix} Toe Loop", rotations


def classify_spin_position(frame_data, start, end):
    """Classify spin position using knee angles, hip position, and body geometry.
    
    Positions detected:
    - Sit: Deep knee bend (skating knee < 110°), hips drop significantly
    - Camel: Free leg extended back, torso horizontal (shoulder Y ≈ hip Y)
    - Layback: Back arched, spine tilts backward (nose behind hips)
    - Upright: Standing position, knee angles > 150°
    - Combination: Significant position changes during the spin
    """
    knee_angles = []
    shoulder_hip_ratios = []  # For camel detection (torso horizontal)
    position_samples = []
    
    # Sample positions throughout spin to detect combinations
    sample_step = max(1, (end - start) // 10)
    
    for i, fd in enumerate(frame_data[start:end]):
        if not fd.get('has_pose'):
            continue
        
        ka_l = fd.get('knee_angle_l', 180)
        ka_r = fd.get('knee_angle_r', 180)
        min_knee = min(ka_l, ka_r)
        knee_angles.append(min_knee)
        
        # Shoulder Y relative to hip Y — for camel detection
        # In camel, shoulders drop toward hip level (torso horizontal)
        hip_y = fd.get('hip_y', 0)
        # shoulder_width can proxy shoulder height change
        sw = fd.get('shoulder_width', 0)
        if hip_y > 0 and sw > 0:
            shoulder_hip_ratios.append(sw)
        
        # Sample positions at intervals for combination detection
        if i % sample_step == 0:
            if min_knee < 110:
                position_samples.append('sit')
            elif min_knee < 140:
                position_samples.append('camel')
            else:
                position_samples.append('upright')
    
    if not knee_angles:
        return 'Upright'
    
    avg_knee = np.mean(knee_angles)
    min_knee_overall = np.min(knee_angles)
    
    # Check for combination spin: position changes during spin
    unique_positions = set(position_samples)
    if len(unique_positions) >= 2 and len(position_samples) >= 4:
        # Count position transitions
        transitions = sum(1 for i in range(1, len(position_samples)) 
                         if position_samples[i] != position_samples[i-1])
        if transitions >= 2:
            print(f"[SPIN-CLASS] Combination: {transitions} transitions, positions: {position_samples}")
            return 'Combination'
    
    # Primary classification by knee angle
    if avg_knee < 110:
        print(f"[SPIN-CLASS] Sit: avg_knee={avg_knee:.1f}°")
        return 'Sit'
    
    # Camel detection: moderate knee angle + reduced shoulder width (torso tilted)
    # When torso goes horizontal, shoulder width in 2D projection decreases
    if avg_knee < 145 and shoulder_hip_ratios:
        avg_sw = np.mean(shoulder_hip_ratios)
        # Shoulder width shrinks when torso tilts forward (foreshortening)
        # Compare with max observed shoulder width
        max_sw = np.max(shoulder_hip_ratios) if shoulder_hip_ratios else avg_sw
        if max_sw > 0 and avg_sw / max_sw < 0.75:
            print(f"[SPIN-CLASS] Camel: avg_knee={avg_knee:.1f}°, sw_ratio={avg_sw/max_sw:.2f}")
            return 'Camel'
    
    # Camel fallback: moderate knee angle range
    if 110 <= avg_knee < 140:
        print(f"[SPIN-CLASS] Camel (by knee): avg_knee={avg_knee:.1f}°")
        return 'Camel'
    
    print(f"[SPIN-CLASS] Upright: avg_knee={avg_knee:.1f}°")
    return 'Upright'


def rotation_prefix(rotations):
    r = int(rotations)
    return {1: 'Single', 2: 'Double', 3: 'Triple'}.get(r, 'Quad' if r >= 4 else 'Single')


def calculate_jump_score(jump_type, rotations):
    """ISU 2024-25 base values for jumps (Communication 2707).
    
    Complete table for all 6 jump types × 4 rotation levels.
    Also includes Euler (single only, used in combinations).
    """
    base_values = {
        1: {
            'Toe Loop': 0.40, 'Salchow': 0.40, 'Loop': 0.50,
            'Flip': 0.50, 'Lutz': 0.60, 'Axel': 1.10, 'Euler': 0.50,
        },
        2: {
            'Toe Loop': 1.30, 'Salchow': 1.30, 'Loop': 1.70,
            'Flip': 1.80, 'Lutz': 2.10, 'Axel': 3.30,
        },
        3: {
            'Toe Loop': 4.20, 'Salchow': 4.30, 'Loop': 4.90,
            'Flip': 5.30, 'Lutz': 5.90, 'Axel': 8.00,
        },
        4: {
            'Toe Loop': 9.50, 'Salchow': 9.70, 'Loop': 10.50,
            'Flip': 11.00, 'Lutz': 11.50, 'Axel': 12.50,
        },
    }
    r = max(1, min(4, int(rotations)))
    # Match jump type to base value table
    for jump_name, bv in base_values[r].items():
        if jump_name in jump_type:
            return bv
    # Fallback to Toe Loop value
    return base_values[r].get('Toe Loop', 0.40)


def calculate_spin_score(spin_position, revolutions, avg_speed):
    """ISU 2024-25 base values for spins.
    
    Estimates level from observable features (revolutions, speed).
    Level determination is approximate without seeing level features
    like difficult variations, change of edge, etc.
    """
    # Base values by spin type and level (B, 1, 2, 3, 4)
    spin_base_values = {
        'Upright':     {'B': 1.00, '1': 1.20, '2': 1.50, '3': 1.90, '4': 2.40},
        'Sit':         {'B': 1.10, '1': 1.30, '2': 1.60, '3': 2.10, '4': 2.50},
        'Camel':       {'B': 1.10, '1': 1.40, '2': 1.80, '3': 2.30, '4': 2.60},
        'Layback':     {'B': 1.20, '1': 1.50, '2': 1.90, '3': 2.40, '4': 2.70},
        'Combination': {'B': 1.70, '1': 2.00, '2': 2.50, '3': 3.00, '4': 3.50},
    }
    
    # Estimate level from observable metrics
    # (Real level requires seeing specific features — this is approximate)
    estimated_level = 'B'
    level_score = 0
    if revolutions >= 4:
        level_score += 1
    if revolutions >= 6:
        level_score += 1
    if avg_speed > 200:
        level_score += 1
    if avg_speed > 300:
        level_score += 1
    
    level_map = {0: 'B', 1: '1', 2: '2', 3: '3', 4: '4'}
    estimated_level = level_map.get(min(level_score, 4), 'B')
    
    values = spin_base_values.get(spin_position, spin_base_values['Upright'])
    return values.get(estimated_level, values['B'])


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


BUILD_HASH = "d117bfe-fix3"  # Update with each deploy to verify Render version

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "yolov8-pose-onnx", "build": BUILD_HASH}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_video(video: UploadFile = File(...)):
    """Analyze a figure skating video using YOLOv8-Pose (ONNX Runtime)."""
    import gc
    
    suffix = '.mp4'
    if video.filename and video.filename.lower().endswith('.mov'):
        suffix = '.mov'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        print(f"[UPLOAD] File: {video.filename}, size: {file_size_mb:.1f}MB")
        
        # Reject files > 50MB to avoid OOM
        if file_size_mb > 50:
            raise HTTPException(status_code=413, detail="Video too large. Max 50MB.")
        
        tmp.write(content)
        tmp_path = tmp.name
        del content  # Free upload buffer immediately
        gc.collect()
    
    try:
        frame_data, fps, total_frames, duration, vid_w, vid_h = extract_frame_data(tmp_path)
        poses_detected = sum(1 for fd in frame_data if fd['has_pose'])
        
        if poses_detected < 5:
            # Still extract what keyframes we can
            few_keyframes = []
            for fd in frame_data:
                if fd.get('_real') and fd.get('has_pose') and '_raw_keypoints' in fd:
                    few_keyframes.append(KeyframeData(
                        frame=fd['frame'],
                        time_sec=fd['time_ms'] / 1000.0,
                        keypoints=fd['_raw_keypoints'],
                        box=fd['_raw_box'],
                    ))
            return AnalysisResult(
                success=True,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration,
                video_width=vid_w,
                video_height=vid_h,
                elements=[],
                session_feedback=[
                    f"Only detected skater in {poses_detected}/{total_frames} frames.",
                    "Make sure the full body is visible with good lighting.",
                ],
                poses={'detection_rate': f"{poses_detected}/{total_frames}"},
                keyframes=few_keyframes if few_keyframes else None,
            )
        
        elements = detect_elements(frame_data, fps)
        
        # Build keyframes list from real (non-interpolated) frames with pose data
        keyframes_list = []
        for fd in frame_data:
            if fd.get('_real') and fd.get('has_pose') and '_raw_keypoints' in fd:
                keyframes_list.append(KeyframeData(
                    frame=fd['frame'],
                    time_sec=fd['time_ms'] / 1000.0,
                    keypoints=fd['_raw_keypoints'],
                    box=fd['_raw_box'],
                ))
        
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
        
        # Build rotation velocity timeseries (smoothed angular velocity per frame)
        # Use median-cleaned orientation to suppress YOLO keypoint swap artifacts:
        # When left/right shoulders swap during fast rotation, orientation jumps ~180°
        # which creates fake velocity spikes of 5000+°/s. Cap frame-to-frame diff
        # and use 5-point median smoothing to remove these outliers.
        rot_velocity = []
        orientations = [fd.get('orientation', 0) for fd in frame_data]
        
        # Step 1: Compute raw frame-to-frame angular diffs
        raw_diffs = []
        for i in range(1, len(orientations)):
            diff = orientations[i] - orientations[i - 1]
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            raw_diffs.append(diff)
        
        # Step 2: Median-filter the diffs to kill swap spikes (5-frame window)
        filtered_diffs = list(raw_diffs)
        if len(raw_diffs) >= 5:
            for i in range(len(raw_diffs)):
                window = raw_diffs[max(0, i-2):i+3]
                filtered_diffs[i] = float(np.median(window))
        
        # Step 3: Cap maximum plausible rotation speed
        # World-class triple axel peaks ~1800-2200°/s
        # Cap at ~100° per frame at 30fps = 3000°/s (generous upper bound)
        MAX_DEG_PER_FRAME = 100
        for i in range(len(filtered_diffs)):
            if abs(filtered_diffs[i]) > MAX_DEG_PER_FRAME:
                filtered_diffs[i] = MAX_DEG_PER_FRAME * (1 if filtered_diffs[i] > 0 else -1)
        
        for i in range(len(filtered_diffs)):
            vel = abs(filtered_diffs[i]) * fps  # degrees per second
            t = frame_data[i + 1].get('time_ms', 0) / 1000.0
            rot_velocity.append(RotationPoint(time_sec=round(t, 3), velocity_dps=round(vel, 1)))
        
        # Step 4: 3-point moving average for visual smoothness
        if len(rot_velocity) >= 3:
            smoothed = []
            vels = [p.velocity_dps for p in rot_velocity]
            for i in range(len(vels)):
                window = vels[max(0, i-1):i+2]
                avg = sum(window) / len(window)
                smoothed.append(RotationPoint(
                    time_sec=rot_velocity[i].time_sec,
                    velocity_dps=round(avg, 1)
                ))
            rot_velocity = smoothed
        
        return AnalysisResult(
            success=True,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            video_width=vid_w,
            video_height=vid_h,
            elements=elements,
            session_feedback=session_feedback,
            poses={
                'detection_rate': f"{poses_detected}/{total_frames}",
                'engine': 'yolov8-pose-onnx',
            },
            keyframes=keyframes_list if keyframes_list else None,
            rotation_velocity=rot_velocity if rot_velocity else None,
        )
    
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
