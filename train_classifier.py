"""
Train a figure skating jump classifier using FS-Jump3D dataset.
V2: Uses 3D view-invariant features + augmentation with noise.

Key insight: 3D joint angles, vertical positions, and rotation around the 
vertical axis are view-invariant. We can approximate these from 2D at inference.
"""

import os
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import math

H36M = {
    'pelvis': 0, 'r_hip': 1, 'r_knee': 2, 'r_ankle': 3,
    'l_hip': 4, 'l_knee': 5, 'l_ankle': 6,
    'spine': 7, 'neck': 8, 'chin': 9, 'head': 10,
    'l_shoulder': 11, 'l_elbow': 12, 'l_wrist': 13,
    'r_shoulder': 14, 'r_elbow': 15, 'r_wrist': 16,
}


def angle_3pts_3d(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 in 3D, degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def angle_3pts_2d(p1, p2, p3):
    """Angle at p2 in 2D."""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def vertical_rotation(poses_3d):
    """Compute rotation around vertical (Z) axis from shoulder orientation.
    FS-Jump3D: X=lateral, Y=depth(along ice), Z=up.
    """
    angles = []
    for frame in poses_3d:
        ls = frame[H36M['l_shoulder']]
        rs = frame[H36M['r_shoulder']]
        dx = rs[0] - ls[0]  # X
        dy = rs[1] - ls[1]  # Y
        angles.append(math.degrees(math.atan2(dy, dx)))
    return np.array(angles)


def hip_vertical_rotation(poses_3d):
    """Rotation around Z axis from hip orientation."""
    angles = []
    for frame in poses_3d:
        lh = frame[H36M['l_hip']]
        rh = frame[H36M['r_hip']]
        dx = rh[0] - lh[0]
        dy = rh[1] - lh[1]
        angles.append(math.degrees(math.atan2(dy, dx)))
    return np.array(angles)


def angular_velocity(angles):
    """Wrapped angular velocity."""
    diffs = []
    for i in range(1, len(angles)):
        d = angles[i] - angles[i-1]
        while d > 180: d -= 360
        while d < -180: d += 360
        diffs.append(d)
    return np.array(diffs)


def detect_phases_3d(poses_3d):
    """Detect jump phases from hip Z (vertical) trajectory.
    Z is up in FS-Jump3D, so apex = max Z.
    """
    n = poses_3d.shape[0]
    hip_z = (poses_3d[:, H36M['l_hip'], 2] + poses_3d[:, H36M['r_hip'], 2]) / 2
    
    if len(hip_z) > 5:
        kernel = np.ones(5) / 5
        hip_z_smooth = np.convolve(hip_z, kernel, mode='same')
    else:
        hip_z_smooth = hip_z
    
    apex_idx = np.argmax(hip_z_smooth)
    
    # Takeoff: where hip starts rising
    takeoff_idx = 0
    baseline = np.percentile(hip_z_smooth[:max(1, apex_idx)], 25)
    for i in range(apex_idx - 1, -1, -1):
        if hip_z_smooth[i] <= baseline:
            takeoff_idx = i
            break
    
    # Landing: where hip stops falling  
    landing_idx = n - 1
    post_baseline = np.percentile(hip_z_smooth[apex_idx:], 25)
    for i in range(apex_idx + 1, n):
        if hip_z_smooth[i] <= post_baseline:
            landing_idx = i
            break
    
    return takeoff_idx, apex_idx, landing_idx


def extract_features_3d(poses_3d, add_noise=False, noise_scale=0.0):
    """Extract view-invariant features from 3D pose sequence.
    
    All features are either:
    - 3D joint angles (view-invariant, extractable from 2D approximately)
    - Vertical positions (Z-axis, maps to Y in 2D image coords)
    - Rotation around vertical axis (somewhat extractable from 2D shoulder width oscillation)
    - Relative positions (view-invariant when normalized)
    """
    n_frames = poses_3d.shape[0]
    
    # Optionally add noise to simulate 2D keypoint errors
    if add_noise and noise_scale > 0:
        noise = np.random.randn(*poses_3d.shape) * noise_scale
        poses_3d = poses_3d + noise
    
    # Normalize: center on pelvis
    pelvis = poses_3d[:, H36M['pelvis'], :].copy()
    poses_centered = poses_3d - pelvis[:, np.newaxis, :]
    
    # Scale by body height (pelvis to head)
    head_heights = np.linalg.norm(poses_centered[:, H36M['head'], :], axis=1)
    avg_height = np.mean(head_heights[head_heights > 0]) if np.any(head_heights > 0) else 1.0
    poses_norm = poses_centered / max(avg_height, 1.0)
    
    # Detect phases
    takeoff_idx, apex_idx, landing_idx = detect_phases_3d(poses_3d)
    
    features = {}
    
    # ═══ 1. ROTATION (around vertical Z axis) ═══
    sh_angles = vertical_rotation(poses_3d)
    hp_angles = hip_vertical_rotation(poses_3d)
    sh_vel = angular_velocity(sh_angles)
    hp_vel = angular_velocity(hp_angles)
    
    # Flight rotation
    fstart = max(takeoff_idx, 0)
    fend = min(landing_idx, n_frames - 1)
    flight_sh = sh_vel[fstart:fend] if fend > fstart else sh_vel
    flight_hp = hp_vel[fstart:fend] if fend > fstart else hp_vel
    
    sh_pos = np.sum(flight_sh[flight_sh > 0]) if len(flight_sh) > 0 else 0
    sh_neg = np.sum(np.abs(flight_sh[flight_sh < 0])) if len(flight_sh) > 0 else 0
    hp_pos = np.sum(flight_hp[flight_hp > 0]) if len(flight_hp) > 0 else 0
    hp_neg = np.sum(np.abs(flight_hp[flight_hp < 0])) if len(flight_hp) > 0 else 0
    
    total_rot_sh = max(sh_pos, sh_neg)
    total_rot_hp = max(hp_pos, hp_neg)
    
    features['rotation_count'] = max(total_rot_sh, total_rot_hp) / 360
    features['total_rotation'] = max(total_rot_sh, total_rot_hp)
    
    if len(flight_sh) > 3:
        features['peak_rot_speed'] = np.max(np.abs(flight_sh))
        features['mean_rot_speed'] = np.mean(np.abs(flight_sh))
        features['rot_speed_std'] = np.std(np.abs(flight_sh))
        mid = len(flight_sh) // 2
        features['rot_accel'] = np.mean(np.abs(flight_sh[mid:])) / (np.mean(np.abs(flight_sh[:mid])) + 1e-8)
    else:
        features['peak_rot_speed'] = 0
        features['mean_rot_speed'] = 0
        features['rot_speed_std'] = 0
        features['rot_accel'] = 1.0
    
    # ═══ 2. ENTRY/APPROACH FEATURES ═══
    pre_window = min(20, takeoff_idx)
    pre_start = max(0, takeoff_idx - pre_window)
    
    # Travel direction in XY plane (horizontal)
    pelvis_pre = poses_3d[pre_start:takeoff_idx+1, H36M['pelvis'], :2]
    if len(pelvis_pre) >= 3:
        travel_vec = pelvis_pre[-1] - pelvis_pre[0]
        travel_angle = math.degrees(math.atan2(travel_vec[1], travel_vec[0]))
        features['travel_speed'] = np.linalg.norm(travel_vec) / max(len(pelvis_pre), 1)
        
        # Body facing at takeoff (from shoulder line)
        facing_angle = sh_angles[takeoff_idx] if takeoff_idx < len(sh_angles) else 0
        entry_diff = facing_angle - travel_angle
        while entry_diff > 180: entry_diff -= 360
        while entry_diff < -180: entry_diff += 360
        features['entry_angle_diff'] = abs(entry_diff)
        features['is_forward_entry'] = 1.0 if abs(entry_diff) < 45 or abs(entry_diff) > 135 else 0.0
    else:
        features['travel_speed'] = 0
        features['entry_angle_diff'] = 90
        features['is_forward_entry'] = 0.0
    
    # Pre-takeoff rotation direction vs flight direction (counter-rotation = Lutz)
    pre_sh = sh_vel[pre_start:takeoff_idx] if takeoff_idx > pre_start else np.array([0])
    pre_rot_total = np.sum(pre_sh)
    flight_rot_dir = np.sign(sh_pos - sh_neg)
    pre_rot_dir = np.sign(pre_rot_total)
    
    features['counter_rotation'] = 1.0 if (pre_rot_dir * flight_rot_dir < 0 and abs(pre_rot_total) > 10) else 0.0
    features['pre_rotation_amount'] = abs(pre_rot_total)
    
    # ═══ 3. TAKEOFF FEATURES (3D joint angles - view invariant!) ═══
    to_start = max(0, takeoff_idx - 3)
    to_end = min(n_frames, takeoff_idx + 3)
    takeoff_poses = poses_3d[to_start:to_end]
    
    if len(takeoff_poses) > 0:
        # 3D ankle height difference (Z axis = vertical)
        l_ankle_z = takeoff_poses[:, H36M['l_ankle'], 2]
        r_ankle_z = takeoff_poses[:, H36M['r_ankle'], 2]
        features['ankle_z_diff_mean'] = np.mean(np.abs(l_ankle_z - r_ankle_z))
        features['ankle_z_diff_max'] = np.max(np.abs(l_ankle_z - r_ankle_z))
        
        # Normalized by body height
        features['ankle_z_diff_norm'] = features['ankle_z_diff_mean'] / max(avg_height, 1.0)
        
        # 3D ankle horizontal spread 
        l_ankle_xy = takeoff_poses[:, H36M['l_ankle'], :2]
        r_ankle_xy = takeoff_poses[:, H36M['r_ankle'], :2]
        features['ankle_xy_spread_takeoff'] = np.mean(np.linalg.norm(l_ankle_xy - r_ankle_xy, axis=1))
        
        # 3D knee angles at takeoff (fully view-invariant!)
        l_knee_angles = []
        r_knee_angles = []
        for p in takeoff_poses:
            lka = angle_3pts_3d(p[H36M['l_hip']], p[H36M['l_knee']], p[H36M['l_ankle']])
            rka = angle_3pts_3d(p[H36M['r_hip']], p[H36M['r_knee']], p[H36M['r_ankle']])
            l_knee_angles.append(lka)
            r_knee_angles.append(rka)
        
        features['l_knee_takeoff'] = np.mean(l_knee_angles)
        features['r_knee_takeoff'] = np.mean(r_knee_angles)
        features['knee_diff_takeoff'] = abs(features['l_knee_takeoff'] - features['r_knee_takeoff'])
        features['knee_min_takeoff'] = min(features['l_knee_takeoff'], features['r_knee_takeoff'])
        
        # 3D hip angles at takeoff
        l_hip_angles = []
        r_hip_angles = []
        for p in takeoff_poses:
            lha = angle_3pts_3d(p[H36M['spine']], p[H36M['l_hip']], p[H36M['l_knee']])
            rha = angle_3pts_3d(p[H36M['spine']], p[H36M['r_hip']], p[H36M['r_knee']])
            l_hip_angles.append(lha)
            r_hip_angles.append(rha)
        
        features['l_hip_angle_takeoff'] = np.mean(l_hip_angles)
        features['r_hip_angle_takeoff'] = np.mean(r_hip_angles)
        features['hip_angle_diff_takeoff'] = abs(features['l_hip_angle_takeoff'] - features['r_hip_angle_takeoff'])
        
        # Toe height relative to ankle (toe pick detection)
        # In H36M we don't have toe directly, but ankle Z relative to pelvis Z indicates lift
        pelvis_z_takeoff = takeoff_poses[:, H36M['pelvis'], 2]
        l_ankle_rel_z = l_ankle_z - pelvis_z_takeoff
        r_ankle_rel_z = r_ankle_z - pelvis_z_takeoff
        features['l_ankle_rel_z_takeoff'] = np.mean(l_ankle_rel_z)
        features['r_ankle_rel_z_takeoff'] = np.mean(r_ankle_rel_z)
        features['ankle_rel_z_diff'] = abs(np.mean(l_ankle_rel_z) - np.mean(r_ankle_rel_z))
    else:
        for k in ['ankle_z_diff_mean', 'ankle_z_diff_max', 'ankle_z_diff_norm',
                   'ankle_xy_spread_takeoff', 'l_knee_takeoff', 'r_knee_takeoff',
                   'knee_diff_takeoff', 'knee_min_takeoff',
                   'l_hip_angle_takeoff', 'r_hip_angle_takeoff', 'hip_angle_diff_takeoff',
                   'l_ankle_rel_z_takeoff', 'r_ankle_rel_z_takeoff', 'ankle_rel_z_diff']:
            features[k] = 0
    
    # ═══ 4. FLIGHT FEATURES ═══
    flight_poses = poses_3d[takeoff_idx:landing_idx+1]
    flight_norm = poses_norm[takeoff_idx:landing_idx+1]
    
    if len(flight_poses) > 0:
        features['air_frames'] = len(flight_poses)
        features['air_frac'] = len(flight_poses) / n_frames
        
        # Jump height (vertical)
        hip_z = (flight_poses[:, H36M['l_hip'], 2] + flight_poses[:, H36M['r_hip'], 2]) / 2
        pre_hip_z = (poses_3d[max(0,takeoff_idx-5):takeoff_idx+1, H36M['l_hip'], 2] + 
                     poses_3d[max(0,takeoff_idx-5):takeoff_idx+1, H36M['r_hip'], 2]) / 2
        baseline_z = np.mean(pre_hip_z) if len(pre_hip_z) > 0 else hip_z[0]
        features['jump_height'] = (np.max(hip_z) - baseline_z) / max(avg_height, 1.0)
        
        # 3D arm tuck
        l_wrist = flight_norm[:, H36M['l_wrist'], :]
        r_wrist = flight_norm[:, H36M['r_wrist'], :]
        wrist_dist = np.mean(np.linalg.norm(l_wrist - r_wrist, axis=1))
        
        l_sh = flight_norm[:, H36M['l_shoulder'], :]
        r_sh = flight_norm[:, H36M['r_shoulder'], :]
        sh_width = np.mean(np.linalg.norm(l_sh - r_sh, axis=1))
        
        features['tuck_ratio'] = wrist_dist / (sh_width + 1e-8)
        
        # Leg position during flight
        l_ankle_flight = flight_norm[:, H36M['l_ankle'], :]
        r_ankle_flight = flight_norm[:, H36M['r_ankle'], :]
        features['ankle_spread_flight'] = np.mean(np.linalg.norm(l_ankle_flight - r_ankle_flight, axis=1))
        
        # Knee angles during flight (3D)
        l_knee_flight = []
        r_knee_flight = []
        for p in flight_poses:
            lka = angle_3pts_3d(p[H36M['l_hip']], p[H36M['l_knee']], p[H36M['l_ankle']])
            rka = angle_3pts_3d(p[H36M['r_hip']], p[H36M['r_knee']], p[H36M['r_ankle']])
            l_knee_flight.append(lka)
            r_knee_flight.append(rka)
        
        features['l_knee_flight'] = np.mean(l_knee_flight)
        features['r_knee_flight'] = np.mean(r_knee_flight)
        features['knee_diff_flight'] = abs(features['l_knee_flight'] - features['r_knee_flight'])
    else:
        for k in ['air_frames', 'air_frac', 'jump_height', 'tuck_ratio',
                   'ankle_spread_flight', 'l_knee_flight', 'r_knee_flight', 'knee_diff_flight']:
            features[k] = 0
    
    # ═══ 5. LANDING FEATURES ═══
    land_start = max(0, landing_idx - 2)
    land_end = min(n_frames, landing_idx + 5)
    landing_poses = poses_3d[land_start:land_end]
    
    if len(landing_poses) > 0:
        l_knee_land = []
        r_knee_land = []
        for p in landing_poses:
            lka = angle_3pts_3d(p[H36M['l_hip']], p[H36M['l_knee']], p[H36M['l_ankle']])
            rka = angle_3pts_3d(p[H36M['r_hip']], p[H36M['r_knee']], p[H36M['r_ankle']])
            l_knee_land.append(lka)
            r_knee_land.append(rka)
        features['knee_min_landing'] = min(np.mean(l_knee_land), np.mean(r_knee_land))
        
        # Ankle spread at landing (free leg extension)
        l_ankle_land = landing_poses[:, H36M['l_ankle'], :]
        r_ankle_land = landing_poses[:, H36M['r_ankle'], :]
        features['ankle_spread_landing'] = np.mean(np.linalg.norm(l_ankle_land - r_ankle_land, axis=1)) / max(avg_height, 1.0)
    else:
        features['knee_min_landing'] = 180
        features['ankle_spread_landing'] = 0
    
    # ═══ 6. BODY SHAPE (3D view-invariant) ═══
    # Spine angle
    spine_angles = []
    for p in poses_3d:
        sa = angle_3pts_3d(p[H36M['pelvis']], p[H36M['spine']], p[H36M['neck']])
        spine_angles.append(sa)
    features['spine_mean'] = np.mean(spine_angles)
    features['spine_std'] = np.std(spine_angles)
    
    # Phase timing
    features['takeoff_frac'] = takeoff_idx / n_frames
    features['apex_frac'] = apex_idx / n_frames
    features['landing_frac'] = landing_idx / n_frames
    
    return features


def load_dataset():
    """Load all FS-Jump3D data and extract features with augmentation."""
    base = Path('/tmp/FS-Jump3D/data/npy')
    
    all_features = []
    all_labels = []
    all_skaters = []
    
    jump_types = ['Axel', 'Flip', 'Loop', 'Lutz', 'Salchow', 'Toeloop']
    
    # Noise levels for augmentation (simulate 2D keypoint noise)
    noise_augments = [0, 5, 10, 15, 20, 25, 30, 40, 50]  # mm noise
    
    for skater_dir in sorted(base.iterdir()):
        if not skater_dir.is_dir(): continue
        skater_name = skater_dir.name
        
        for jump_dir in sorted(skater_dir.iterdir()):
            if not jump_dir.is_dir(): continue
            jump_type = jump_dir.name
            if jump_type not in jump_types: continue
            
            for npy_file in sorted(jump_dir.glob('*.npy')):
                try:
                    pose_3d = np.load(npy_file)
                    if pose_3d.shape[1] != 17 or pose_3d.shape[2] != 3: continue
                    
                    for noise in noise_augments:
                        feats = extract_features_3d(pose_3d, add_noise=(noise > 0), noise_scale=noise)
                        all_features.append(feats)
                        all_labels.append(jump_type)
                        all_skaters.append(skater_name)
                        
                except Exception as e:
                    print(f"Error: {npy_file}: {e}")
    
    return all_features, all_labels, all_skaters


def features_to_array(features_list):
    if not features_list:
        return np.array([]), []
    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    return X, feature_names


def main():
    print("=" * 60)
    print("FS-Jump3D Jump Classifier Training V2 (3D View-Invariant)")
    print("=" * 60)
    
    print("\n[1/4] Loading dataset...")
    features_list, labels, skaters = load_dataset()
    X, feature_names = features_to_array(features_list)
    y = np.array(labels)
    groups = np.array(skaters)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"  Samples: {len(X)}, Features: {len(feature_names)}")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Per class: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    print("\n[2/4] Cross-validation...")
    
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
        ))
    ])
    
    # LOSO
    logo = LeaveOneGroupOut()
    logo_scores = cross_val_score(clf, X, y_enc, cv=logo, groups=groups, scoring='accuracy')
    print(f"  LOSO accuracy: {logo_scores.mean():.3f} ± {logo_scores.std():.3f}")
    for skater, score in zip(np.unique(groups), logo_scores):
        print(f"    {skater}: {score:.3f}")
    
    # 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_scores = cross_val_score(clf, X, y_enc, cv=skf, scoring='accuracy')
    print(f"  5-fold accuracy: {skf_scores.mean():.3f} ± {skf_scores.std():.3f}")
    
    print("\n[3/4] Training final model...")
    clf.fit(X, y_enc)
    
    importances = clf.named_steps['clf'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  Top 15 features:")
    for i in range(min(15, len(feature_names))):
        idx = sorted_idx[i]
        print(f"    {feature_names[idx]:35s} {importances[idx]:.4f}")
    
    y_pred = clf.predict(X)
    print(f"\n  Training accuracy: {np.mean(y_pred == y_enc):.3f}")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))
    
    # LOSO detailed: train on 3, predict on 1, show confusion per skater
    print("\n  LOSO detailed per skater:")
    for test_skater in np.unique(groups):
        train_mask = groups != test_skater
        test_mask = groups == test_skater
        clf_temp = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42))
        ])
        clf_temp.fit(X[train_mask], y_enc[train_mask])
        y_test_pred = clf_temp.predict(X[test_mask])
        y_test_true = y_enc[test_mask]
        print(f"\n  {test_skater}:")
        print(classification_report(y_test_true, y_test_pred, target_names=le.classes_, zero_division=0))
    
    print("\n[4/4] Saving...")
    model_data = {
        'pipeline': clf,
        'label_encoder': le,
        'feature_names': feature_names,
        'version': '2.0',
        'classes': list(le.classes_),
        'loso_accuracy': float(logo_scores.mean()),
    }
    
    output_path = '/tmp/ice-analysis-api/jump_classifier.joblib'
    joblib.dump(model_data, output_path, compress=3)
    size = os.path.getsize(output_path) / 1024
    print(f"  Saved: {output_path} ({size:.1f} KB)")
    print(f"\n  LOSO: {logo_scores.mean():.1%} | 5-fold: {skf_scores.mean():.1%}")


if __name__ == '__main__':
    main()
