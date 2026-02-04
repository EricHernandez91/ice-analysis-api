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

class AnalysisResult(BaseModel):
    success: bool
    total_frames: int
    fps: float
    duration_seconds: float
    elements: List[SkatingElement]
    session_feedback: List[str]
    poses: Optional[dict] = None
    keyframes: Optional[List[KeyframeData]] = None


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
    duration = total_frames / fps if fps > 0 else 0
    
    # Process enough frames for accurate rotation counting.
    # Triple axels spin ~1880°/s — need <90° between samples to avoid aliasing.
    # At 30fps: step=1 → 12°/frame (perfect), step=2 → 24° (good), step=3 → 36° (OK for doubles)
    # For paid tier, process generously. Target ~40 frames for short clips.
    target_frames = 40
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
    return frame_data, fps, total_frames, duration


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
        numeric_keys = ['hip_y', 'hip_x', 'ankle_y', 'orientation', 
                       'knee_angle_l', 'knee_angle_r', 'shoulder_width', 
                       'wrist_spread', 'ankle_diff']
        
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
                    # Handle angle wrapping for orientation
                    if key == 'orientation':
                        diff = nxt[key] - cur[key]
                        while diff > 180: diff -= 360
                        while diff < -180: diff += 360
                        interp[key] = cur[key] + diff * t
                    else:
                        interp[key] = cur[key] + (nxt[key] - cur[key]) * t
            full_data.append(interp)
    
    full_data.append(sampled_data[-1])
    return full_data


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
        
        # Count rotation during the detected jump window only
        rot_start = start
        rot_end = end
        
        # Use ONLY real (non-interpolated) frames for rotation counting.
        # Interpolated frames smooth out rapid rotation, undercounting it.
        real_orientations = []
        for fd in frame_data[rot_start:rot_end + 1]:
            if fd.get('_real', True):  # Default True for non-interpolated
                real_orientations.append(fd.get('orientation', 0))
        
        # If we don't have real frame markers, use all frames
        if len(real_orientations) < 3:
            real_orientations = [fd.get('orientation', 0) for fd in frame_data[rot_start:rot_end + 1]]
        
        total_rotation = 0
        for j in range(1, len(real_orientations)):
            diff = real_orientations[j] - real_orientations[j - 1]
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            total_rotation += abs(diff)
        
        raw_rotations = total_rotation / 360
        
        # Shoulder-based rotation tracking undercounts by ~15-20% during 
        # fast rotation (body tucks, keypoints become ambiguous).
        # Apply calibrated correction. Validated against known jumps:
        # - Single Axel (1.5): raw ~1.41 → corrected ~1.62 → rounds to 1.5 ✓
        # - Triple Axel (3.5): raw ~2.94 → corrected ~3.38 → rounds to 3.5 ✓
        corrected = raw_rotations * 1.15
        rotations = round(corrected * 2) / 2
        
        # Filter: must have at least 0.5 rotations to be a real jump
        if rotations < 0.5:
            print(f"[JUMP] Rejected apex@{apex_frame}: only {rotations} rotations")
            continue
        
        jump_type = classify_jump(frame_data, start, apex_frame, end, rotations)
        feedback = generate_jump_feedback(frame_data, start, apex_frame, end, height, rotations, jump_type, fps)
        score = calculate_jump_score(jump_type, rotations)
        
        air_time_s = air_frames / fps
        confidence = min(0.95, 0.6 + (height / 100) + (rotations * 0.1))
        
        print(f"[JUMP] ✅ {jump_type}: {rotations} rot (raw: {raw_rotations:.2f}, corrected: {corrected:.2f}), height={height:.0f}px, air={air_time_s:.2f}s, frames {start}-{end}")
        
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
    
    See SKATING_CLASSIFICATION.md for full research notes.
    """
    reasons = []  # Track classification reasoning
    
    # ── Step 1: Axel Detection ──────────────────────────────────
    # Axels have fractional rotation (n + 0.5) due to forward takeoff
    fractional = rotations % 1.0
    is_half_rotation = (0.1 <= fractional <= 0.9)
    
    # Also check for forward entry: body faces travel direction at takeoff
    # Forward entry means hip_x is increasing (or decreasing) in the same
    # direction the shoulders face
    forward_entry = False
    if start > 5:
        pre_takeoff = frame_data[max(0, start - 10):start]
        pre_orientations = [fd.get('orientation', 0) for fd in pre_takeoff if fd.get('has_pose')]
        pre_hip_xs = [fd.get('hip_x', 0) for fd in pre_takeoff if fd.get('has_pose')]
        
        if len(pre_hip_xs) >= 3:
            # Travel direction from hip movement
            hip_dx = pre_hip_xs[-1] - pre_hip_xs[0]
            # Shoulder direction from orientation
            avg_orient = np.mean(pre_orientations) if pre_orientations else 0
            # Forward entry: shoulder vector roughly aligns with travel direction
            # (orientation ~0° or ~180° means shoulders face sideways/forward)
            # In 2D, forward skating has shoulders perpendicular to travel
            # Not a perfect signal, but combined with half-rotation it's strong
            reasons.append(f"hip_dx={hip_dx:.1f}, orient={avg_orient:.1f}")
    
    if is_half_rotation:
        axel_rotations = round(rotations * 2) / 2
        if axel_rotations % 1.0 == 0.5:
            reasons.append(f"Axel: fractional rotation {rotations} → {axel_rotations}")
            print(f"[CLASSIFY] {' | '.join(reasons)}")
            return f"{rotation_prefix(axel_rotations)} Axel"
    
    # ── Step 2: Toe Pick Detection ──────────────────────────────
    # Toe pick jumps have one ankle significantly higher than the other
    # at takeoff (picking foot plants toe in ice behind)
    takeoff_window = max(3, int((apex - start) * 0.6))
    takeoff_start = max(start, apex - takeoff_window)
    takeoff_frames = frame_data[takeoff_start:apex]
    
    ankle_diffs = [fd.get('ankle_diff', 0) for fd in takeoff_frames if fd.get('has_pose')]
    avg_ankle_diff = np.mean(ankle_diffs) if ankle_diffs else 0
    max_ankle_diff = max(ankle_diffs) if ankle_diffs else 0
    
    TOE_PICK_THRESHOLD = 25  # pixels — ankle height difference indicating toe assist
    is_toe_jump = avg_ankle_diff > TOE_PICK_THRESHOLD or max_ankle_diff > TOE_PICK_THRESHOLD * 1.5
    
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
            return f"{prefix} Loop"
        else:
            reasons.append(f"Edge jump: knee_avg={avg_knee:.1f}° ≥ {LOOP_KNEE_THRESHOLD} → Salchow")
            print(f"[CLASSIFY] {' | '.join(reasons)}")
            return f"{prefix} Salchow"
    
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
        return f"{prefix} Lutz"
    
    # Distinguish Flip vs Toe Loop by ankle differential magnitude
    # Flip typically has larger ankle differential (more aggressive toe pick)
    FLIP_THRESHOLD = 35  # pixels
    if avg_ankle_diff > FLIP_THRESHOLD:
        reasons.append(f"Toe: ankle_diff={avg_ankle_diff:.1f} > {FLIP_THRESHOLD} → Flip")
        print(f"[CLASSIFY] {' | '.join(reasons)}")
        return f"{prefix} Flip"
    
    reasons.append(f"Toe: default → Toe Loop")
    print(f"[CLASSIFY] {' | '.join(reasons)}")
    return f"{prefix} Toe Loop"


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


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "yolov8-pose-onnx"}


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
        frame_data, fps, total_frames, duration = extract_frame_data(tmp_path)
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
            keyframes=keyframes_list if keyframes_list else None,
        )
    
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
