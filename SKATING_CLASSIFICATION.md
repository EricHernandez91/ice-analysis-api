# Figure Skating Classification from Pose Data

## Research Notes — Biomechanical Signatures for YOLOv8-Pose (17 COCO Keypoints)

### COCO Keypoint Reference
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

**Limitation:** No foot/blade keypoints — we cannot directly see edge direction. 
We infer edge from ankle position relative to knee and hip.

---

## 6 Standard Jumps — Biomechanical Signatures

### 1. Toe Loop (T)
- **Takeoff:** Back outside edge + toe pick (right foot picks, left foot takes off for CCW skaters)
- **Landing:** Back outside edge (right foot for CCW)
- **Pose signature:**
  - Toe pick visible as **ankle height differential** at takeoff — picking foot's ankle is HIGHER (toe is planted in ice behind)
  - Entry is backward (shoulders face away from travel direction)
  - Moderate ankle differential (15-40px typically)

### 2. Salchow (S)
- **Takeoff:** Back inside edge, NO toe pick
- **Landing:** Back outside edge (opposite foot)
- **Pose signature:**
  - **Minimal ankle height difference** at takeoff — both feet roughly level since no toe assist
  - Free leg swings forward and up to generate lift
  - Entry is backward
  - Free leg (usually right for CCW) swings across body at takeoff

### 3. Loop (Lo)
- **Takeoff:** Back outside edge, NO toe pick
- **Landing:** Back outside edge (same foot)
- **Pose signature:**
  - **Minimal ankle height difference** (edge jump)
  - Legs are **crossed/close together** at takeoff — distinctive "sitting" position
  - Knee angles are more bent than Salchow at takeoff
  - Both ankles very close in x-position

### 4. Flip (F)
- **Takeoff:** Back inside edge + toe pick
- **Landing:** Back outside edge
- **Pose signature:**
  - **Significant ankle height differential** (toe pick assist)
  - Entry is backward
  - Shoulder rotation matches hip rotation (natural rotation direction)
  - Entry curve goes into the rotation direction

### 5. Lutz (Lz)
- **Takeoff:** Back outside edge + toe pick (COUNTER-rotation entry)
- **Landing:** Back outside edge
- **Pose signature:**
  - **Significant ankle height differential** (toe pick assist)
  - **COUNTER-ROTATION** — the key differentiator from Flip:
    - Long backward glide on outside edge
    - Shoulders/hips rotate OPPOSITE to eventual jump rotation before takeoff
    - At the moment of toe pick, the body "switches" rotation direction
  - Typically has longer straight entry (less curved approach)
  - Shoulder angle changes direction between pre-takeoff and takeoff

### 6. Axel (A)
- **Takeoff:** Forward outside edge (the ONLY forward-entry jump)
- **Landing:** Back outside edge
- **Always has +0.5 rotation** (1→1.5, 2→2.5, 3→3.5)
- **Pose signature:**
  - **Forward entry** — shoulders/hips face travel direction at takeoff
  - This is the strongest signal: hip_x velocity at takeoff matches shoulder facing direction
  - Distinctive "step up" motion — free leg kicks forward
  - No toe pick (edge jump)
  - Fractional rotation count (n + 0.5) is the primary detection method

---

## Classification Algorithm (from Pose Data)

### Step 1: Detect if Axel
- **Primary signal:** Rotation count is near n.5 (1.5, 2.5, 3.5)
- **Secondary signal:** Forward entry — at takeoff frame, body faces travel direction
- **Confidence boost:** Both signals agree → high confidence Axel

### Step 2: Detect Toe Pick (Toe vs Edge Jump)
- Measure `ankle_diff` = |left_ankle_y - right_ankle_y| at takeoff
- Toe pick jumps have one foot planted behind → ankle height difference > threshold
- **Toe pick threshold:** > 25px average over takeoff frames
  - Above: Toe jump (T, F, or Lz)
  - Below: Edge jump (S or Lo)

### Step 3: Classify Edge Jumps (S vs Lo)
- **Salchow:** Free leg swings — wider wrist/arm spread, less knee bend
- **Loop:** Crossed legs — both ankles close in x-position, deeper knee bend
- Measure: average of min(knee_angle_l, knee_angle_r) at takeoff
  - < 130°: likely Loop (crossed/sitting position)
  - ≥ 130°: likely Salchow (more upright)

### Step 4: Classify Toe Jumps (T vs F vs Lz)
- **Counter-rotation detection** (Lutz identifier):
  - Look at orientation change in 10-15 frames before takeoff
  - Lutz has counter-rotation: orientation moves opposite to jump rotation, then reverses
  - Measure: orientation_delta_pre (10 frames before) vs orientation_delta_during (during jump)
  - If signs are opposite → Lutz
  - If signs are same → Flip or Toe Loop
- **Toe Loop vs Flip:**
  - Toe Loop: typically smaller ankle differential, simpler entry
  - Flip: larger ankle differential, back inside edge (hard to distinguish from Toe Loop with pose only)
  - Heuristic: ankle_diff > 35px and no counter-rotation → Flip; < 35px → Toe Loop

---

## Spin Classification from Pose Data

### Upright Spin (USp)
- Both knee angles > 150°
- Torso roughly vertical
- Hip-shoulder line roughly vertical

### Sit Spin (SSp)
- Skating knee angle < 110° (deep bend)
- Free leg extended forward
- Hip Y significantly lower than standing

### Camel Spin (CSp)
- Free leg extended behind → hip angle opens
- Torso tilts forward
- Key: measure angle at hip joint between torso and free leg
- Shoulder Y close to hip Y (horizontal body)
- Free leg ankle higher than knee

### Layback Spin (LSp)
- Back arched → nose/shoulder Y higher relative to hip
- Spine tilts backward
- Both knees relatively straight (> 140°)
- Distinctive: nose behind hip center in x-direction

### Combination Spin (CCoSp)
- Position changes during spin → detect if min knee angle varies significantly
- Multiple position phases detected

---

## ISU 2024-25 Base Values

### Jumps (Single → Quad)
| Jump       | Single | Double | Triple | Quad   |
|------------|--------|--------|--------|--------|
| Toe Loop   | 0.40   | 1.30   | 4.20   | 9.50   |
| Salchow    | 0.40   | 1.30   | 4.30   | 9.70   |
| Loop       | 0.50   | 1.70   | 4.90   | 10.50  |
| Flip       | 0.50   | 1.80   | 5.30   | 11.00  |
| Lutz       | 0.60   | 2.10   | 5.90   | 11.50  |
| Axel       | 1.10   | 3.30   | 8.00   | 12.50  |
| Euler      | 0.50   | —      | —      | —      |

### Spins (Level B → 4)
| Spin   | Base | Level 1 | Level 2 | Level 3 | Level 4 |
|--------|------|---------|---------|---------|---------|
| USp    | 1.00 | 1.20    | 1.50    | 1.90    | 2.40    |
| SSp    | 1.10 | 1.30    | 1.60    | 2.10    | 2.50    |
| CSp    | 1.10 | 1.40    | 1.80    | 2.30    | 2.60    |
| LSp    | 1.20 | 1.50    | 1.90    | 2.40    | 2.70    |
| FSSp   | 1.70 | 2.00    | 2.30    | 2.60    | 3.00    |
| FCSp   | 1.70 | 2.00    | 2.30    | 2.60    | 3.00    |
| CCoSp  | 1.70 | 2.00    | 2.50    | 3.00    | 3.50    |
| CoSp   | 1.50 | 1.70    | 2.00    | 2.50    | 3.00    |

---

## Limitations of Pose-Only Classification

1. **No blade contact data** — Cannot directly see inside/outside edge
2. **17 keypoints only** — No foot/toe keypoints in COCO format
3. **2D projection** — Depth information lost in single-camera video
4. **Frame rate** — Fast rotations may alias at low FPS
5. **Occlusion** — Tightly tucked arms/legs during rotation reduce keypoint accuracy

### Confidence Levels
- **Axel:** HIGH — forward entry + half-rotation is very distinctive
- **Toe vs Edge:** MEDIUM — ankle height differential is visible but noisy
- **Lutz vs Flip:** LOW-MEDIUM — counter-rotation is subtle in 2D
- **Salchow vs Loop:** LOW — body position differences are small
- **Toe Loop vs others:** LOW — simplest jump, default when uncertain
