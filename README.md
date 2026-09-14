# AnimalDetection
Detection and tracking of animals (chickens) in drone (UAV) footage, with a focus on recovering the **real-world (lat/lon) coordinates** of each detection rather than just its pixel position.

The project runs entirely on Kaggle notebooks and is organized into three stages, one per subfolder under `notebooks/`: **labeling → tracking → geolocation**.

## Pipeline

### 1. `notebooks/labeling/` — building the detector

- **`chicken_label_project.ipynb` / `chicken_label_project_v2.ipynb`** — trains a YOLOv8 (Ultralytics) object detector on manually labeled drone frames, with augmentations tuned for small objects (mosaic, scale, copy-paste, flips, HSV jitter). `v2` adds a fine-tuning stage on an expanded dataset, boosts frames where extra boxes were added for dense clusters, and compares model versions on held-out frames before packaging the final weights.
- **`chicken_label_auto_project.ipynb`** — uses the trained model with [SAHI](https://github.com/obss/sahi) (slicing-aided inference) to auto-label a new batch of frames, exporting both YOLO-format `.txt` labels and a CVAT XML file, so new frames can be reviewed/corrected instead of labeled from scratch.

### 2. `notebooks/tracking/` — detecting and tracking across the full video

- **`chicken_track_drone_v1.ipynb`** — runs SAHI + YOLOv8 detection over a dense set of ~25k frames (with checkpointing to resume long runs), then feeds detections into [ByteTrack](https://github.com/ifzhang/ByteTrack) (via `supervision`) to produce consistent per-animal tracks. Includes diagnostics (track length distribution, frame-to-frame displacement via the Hungarian algorithm), exports tracks as a CVAT XML (`tracks_dense.xml`), and renders an annotated preview video.

### 3. `notebooks/geolocation/` — projecting detections to GPS coordinates

This folder documents the evolution of the geo-referencing approach, from an early ground-control-point (GCP) method to the final shelter-height calibration used in production:

- **`geolocation_pipeline_ipynb.ipynb`** — first attempts: identifying the right telemetry file via MSE matching against 4 manually marked ground control points, then a full camera-tilt model and multi-frame bundle adjustment.
- **`geolocation_pipeline_ipynb_v2.ipynb`** — refines the GCP-based approach (corners named by azimuth from the shelter).
- **`geolocation_pipeline_ipynb_v3.ipynb`** — experiments with optical-flow-based frame-to-frame homography ("islands") to refine positions between telemetry samples.
- **`geolocation_pipeline_ipynb_v4.ipynb`** — combines a fixed focal length (derived from an assumed horizontal FOV), optical-flow drift detection, and piecewise-linear pitch calibration at anomaly-dense "knot" frames.
- **`chicken-positions-v7.ipynb`** — the final pipeline. Instead of GCPs, it calibrates using the **known height of the shelters** visible in the footage (1.6 m): a ray from a shelter's base, extended 1.6 m vertically, should reproject onto the shelter's marked top pixel. From a set of manually marked (base, top) pixel pairs across many frames, it fits:
  - one shared **focal length** for the whole video (fixed lens), and
  - a **per-frame gimbal pitch** (via least-squares on pixel-space reprojection error).

  It then:
  - interpolates pitch between calibrated frames and drops detections where pitch/altitude make the geometry unreliable,
  - includes an optional video/telemetry time-offset search to verify sync,
  - projects every detection from `tracks_dense.xml` to lat/lon with sanity filters (height, pitch, max plausible ground distance),
  - smooths each track with a median filter and splits it wherever the implied speed exceeds what an animal can physically achieve (catching occlusion jumps, ID switches, or calibration noise),
  - produces a static map of high-confidence trajectories, plus interactive HTML/JS widgets (a timeline slider, and a combined map + video-thumbnail viewer) to visually sanity-check the calibration against the original footage.

## Requirements

- Python 3
- `numpy`, `pandas`, `matplotlib`, `scipy`, `opencv-python`
- `ultralytics` (YOLOv8), `sahi`, `supervision` (ByteTrack)
- A Jupyter/Kaggle environment with GPU

## Inputs

- Manually labeled drone frames (for training) and raw drone frame sequences (for inference)
- Drone flight telemetry (CSV: time, lat, lng, height AGL, yaw)
- A small set of manually marked shelter base/top pixel pairs, used only for camera calibration

## Output

- A trained YOLOv8 chicken detector
- `tracks_dense.xml` — per-frame bounding boxes with track IDs (CVAT format)
- A CSV of geotagged detections per track (`track_id`, `frame`, `time_s`, `east`, `north`, `lat`, `lon`, `segment_id` after speed-based splitting), plus static and interactive visualizations of the resulting trajectories

## Notes

- Calibration uses pixel-space residuals (reprojected vs. marked shelter top) rather than world-space error, which avoids bias from annotation noise.
- Frames near takeoff/landing or with very shallow camera pitch are excluded, since the projection geometry becomes degenerate at low pitch/altitude.
