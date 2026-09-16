# AnimalDetection

Detection and tracking of animals (chickens) in drone (UAV) footage, with a focus on recovering the real-world (lat/lon) coordinates of each detection.

The project runs entirely on Kaggle notebooks and is organized into three main stages under `notebooks/`: **labeling**, **tracking**, and **geolocation**.

### Pipeline

#### 1. `notebooks: labeling`

* **`chicken_label_project.ipynb` / `chicken_label_project_v2.ipynb`**  
  Trains a YOLOv8 (Ultralytics) object detector on manually labeled drone frames, with augmentations tuned for small objects (mosaic, scale, copy-paste, flips, HSV jitter). The `v2` notebook adds fine-tuning on an expanded dataset, boosts frames where extra boxes were added for dense clusters, and compares model versions on held-out frames before saving final weights.

* **`chicken_label_auto_project.ipynb`**  
  Uses the trained model with SAHI (slicing-aided inference) to auto-label a new batch of frames, exporting both YOLO-format `.txt` labels and a CVAT XML file so new frames can be reviewed and corrected instead of labeled from scratch.

#### 2. `notebooks: tracking`

* **`chicken_track_drone_v1.ipynb`**  
  Runs SAHI + YOLOv8 detection over a dense set of ~25k frames (with checkpointing to resume long runs), then feeds detections into ByteTrack (via `supervision`) to produce consistent per-animal tracks. Includes diagnostics (track length distribution, frame-to-frame displacement via the Hungarian algorithm), exports tracks as a CVAT XML (`tracks_dense.xml`), and renders an annotated preview video.

#### 3. `notebooks: geolocation`

This folder documents the evolution of the georeferencing approach, moving from an early ground control point (GCP) method to the final shelter-height calibration used in production:

* **`geolocation_pipeline_ipynb.ipynb`**  
  Initial attempts: identifying the correct telemetry file via MSE matching against 4 manually marked ground control points, followed by a full camera tilt model and multi-frame bundle adjustment.

* **`geolocation_pipeline_ipynb_v2.ipynb`**  
  Refines the GCP-based approach with corners named by azimuth relative to the shelter.

* **`geolocation_pipeline_ipynb_v3.ipynb`**  
  Experiments with optical-flow-based frame-to-frame homography to refine positions between telemetry samples.

* **`geolocation_pipeline_ipynb_v4.ipynb`**  
  Combines a fixed focal length, optical flow drift detection, and piecewise-linear pitch calibration at anomaly-dense knot frames.

* **`chicken-positions-v7.ipynb` (Final Pipeline)**  
  Instead of GCPs, this approach calibrates using the known height of the shelters visible in the footage (1.6 m). A ray projected from a shelter's base, extended 1.6 m vertically, should reproject onto the shelter's marked top pixel. From a set of manually marked (base, top) pixel pairs across multiple frames, it fits:
  1. A single shared focal length for the entire video (fixed lens).
  2. A per-frame gimbal pitch angle (via least-squares optimization on pixel-space reprojection error).

  **Final processing steps:**
  * Interpolates pitch between calibrated frames and drops detections where pitch or altitude make the geometry unreliable.
  * Runs an optional video/telemetry time offset search to verify synchronization.
  * Projects every detection from `tracks_dense.xml` to lat/lon using sanity filters (height, pitch, maximum plausible ground distance).
  * Smooths each track with a median filter and splits tracks whenever implied speed exceeds realistic animal movement (catching occlusion jumps, ID switches, or calibration noise).
  * Generates a static map of high-confidence trajectories alongside interactive HTML/JS widgets (a timeline slider and a combined map with video thumbnail viewer) to visually verify calibration against original footage.

### Requirements

* Python 3
* `numpy`, `pandas`, `matplotlib`, `scipy`, `opencv-python`
* `ultralytics` (YOLOv8), `sahi`, `supervision` (ByteTrack)
* A Jupyter or Kaggle environment with GPU support

### Inputs

* Manually labeled drone frames (for training) and raw frame sequences (for inference)
* Drone flight telemetry (CSV format: timestamp, lat, lng, height AGL, yaw)
* A small set of manually marked shelter base/top pixel pairs for camera calibration

### Outputs

* Trained YOLOv8 chicken detection model weights
* `tracks_dense.xml`:  per-frame bounding boxes with track IDs (CVAT format)
* CSV of geotagged detections per track (`track_id`, `frame`, `time_s`, `east`, `north`, `lat`, `lon`, `segment_id`)
* Static and interactive trajectory visualization plots

### Notes

* Calibration relies on pixel-space residuals (reprojected vs. marked shelter top) rather than world-space error to avoid bias from annotation noise.
* Frames near takeoff or landing and those captured at very shallow camera pitch angles are excluded, as projection geometry becomes degenerate under low pitch/altitude conditions.
