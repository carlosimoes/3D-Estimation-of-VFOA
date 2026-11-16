# 3D-Estimation-of-VFOA

The dissertation project is named 3D Estimation of VFOA, where Attention was calculated based on the person's Gaze and Head Pose. 

Humanoid and social robots may provide valuable resources to society in the most diverse and complex activities and challenges, thanks to their increasing mechanical and decisionmaking abilities. However, robots must comprehend and acquire information about their surroundings for proper interaction with humans. The VFOA can be used as the primary conversational cue. To tackle these challenges, we develop a novel approach that estimates and tracks the VFOA.
The proposed model stems from the consideration that the eye gaze and head pose carry information about actions and interactions. The proposed formulation leads to a 3D algorithm that considers: 
(i) A bounding box of every object in the field of view of the robot’s camera;
(ii) A ray casting algorithm that considers head and gaze directions;
(iii) A Kalman filter performs the tracking of the gaze.
(iv) Finally, the VFOA algorithm estimates the object of attention based on a weighted sum of gaze and head pose information. 
We study the parameters of 3D VFOA algorithm, running simulated scenarios for a selection of the most adequate parameters. The novel approach is validated, tested and benchmarked on the public MPI Sintel dataset containing animated real-world interactions.


Index Terms — VFOA, Eye Gaze, Head Pose, Object Detection, Human-Robot Interaction

---

## What the code does

For every frame of a video (or a stereo pair, an image directory, a camera, or a
synthetic test scene) the pipeline estimates, per person, **which object the
person is looking at** (Visual Focus of Attention, VFOA):

1. **depth**: metric depth from stereo block matching, ground-truth depth (Sintel) or a constant;
2. **persons**: person detection with head boxes and COCO-17 keypoints;
3. **tracking**: persistent person ids (ByteTrack-style Hungarian matching);
4. **headpose** and **gaze**: 3D head and gaze directions per person;
5. **segmentation**: object and person segments;
6. **scene**: metric 3D objects (segments + depth), the 3D eye position of each person, and the segment of the person themselves (which they cannot attend to);
7. **smoothing**: Kalman/EMA smoothing of the directions;
8. **attention**: an attention score per (person, object) pair, with occlusion handling, and a temporally smoothed attention target per person.

Two implementations live in this repository:

- `thesis_final/`: the original offline thesis scripts. Their public names are kept,
  but they are now thin wrappers over the new package (one implementation).
- `vfoa/`: the streaming package. Frames are processed one at a time, each model is
  loaded once, and slow stages can run at a lower rate than the frame rate. Every
  stage has pluggable **backends**, chosen by configuration:

| Stage | `fast` (default, near real time) | `accurate` / `thesis` (thesis models) | `synthetic` (no models) |
|---|---|---|---|
| depth | `constant` (or `sintel_gt`, `stereo_bm`) | `stereo_bm` (real disparity, optional WLS) | `synthetic` |
| persons | `yolo_pose` (YOLO11n-pose) | `openpose_densepose` | `synthetic` |
| headpose | `keypoints` (PnP on face keypoints) | `fsanet` | `synthetic` |
| gaze | `l2cs` (MobileGaze ONNX, every 2nd frame), falls back to `head_proxy` | `gaze360` (temporal window) | `synthetic` |
| segmentation | `yolo_seg` (YOLO11n-seg, every 10th frame) | `panoptic_fpn` (every frame) | `synthetic` (every 5th frame) |

- **accurate**: thesis models + the improved algorithm (angular attention scoring,
  confidence fusion, depth occlusion, ByteTrack, Kalman and HMM smoothing).
- **thesis**: thesis models + the thesis-faithful, bug-fixed algorithm (greedy IoU
  tracking, static Gaze360 window, radius outlier filter, `wall` ignored, ray-cone
  attention with the 45-degree agreement fusion, no occlusion, no smoothing).
- **synthetic**: a deterministic scripted scene with fake backends and known ground
  truth. It needs no model and is used by the tests and the benchmark.

## Install

Python >= 3.10. From the repository root:

```bash
pip install -r requirements.txt            # core: numpy, scipy, opencv-python-headless, pyyaml, pillow
pip install -r requirements-fast.txt       # fast profile: torch, ultralytics, onnxruntime
pip install -r requirements-accurate.txt   # accurate/thesis profiles: torch, torchvision, onnxruntime, open3d, ...
pip install -r requirements-dev.txt        # tests: pytest + CPU torch and onnxruntime
```

- **ultralytics is an optional dependency licensed AGPL-3.0.** It is listed only in
  `requirements-fast.txt` and imported lazily, only in `vfoa/backends/fast/_ultra.py`
  when a YOLO backend is loaded. Every other part of the package (and the
  `synthetic`, `accurate` and `thesis` profiles) works without it.
- All heavy packages (torch, onnxruntime, open3d, ultralytics, detectron2, ...) are
  imported lazily inside the backends. A missing package or missing weights raise
  `BackendUnavailable` with a message that names the package and the config key.
- **Weights.** YOLO11 and MobileGaze weights download on first use into
  `models.weights_dir` (default `weights/`, git-ignored) or `$VFOA_WEIGHTS_DIR`.
  Set `models.allow_download=false` for offline machines.
- **Accurate backends.** detectron2 (with `projects/DensePose`), Pose_Estimation
  (PyTorch OpenPose), gaze360 and headpose-fsanet-pytorch are source checkouts, not
  PyPI packages. Point `models.third_party_root` or `$VFOA_THIRD_PARTY` at a
  detectron2 checkout that also holds the other three (the thesis layout); relative
  weight paths resolve against it. The WLS disparity filter needs
  `opencv-contrib-python-headless` (it replaces `opencv-python-headless`).

## Quick start

All commands run from the repository root. Each run writes `results.jsonl`,
`config.yaml` (the resolved configuration) and `timing.json` to `--output-dir`, and
prints the per-stage timing table.

```bash
# synthetic profile: no models, no weights
python -m vfoa run --profile synthetic --max-frames 100 --output-dir outputs/synth

# fast profile on a monocular video (constant depth), with an overlay video
python -m vfoa run --profile fast --input Final_videos/got.mp4 --output-dir outputs/got --video

# fast profile on the Sintel clip with ground-truth depth
python -m vfoa run --config configs/sintel_gt.yaml --output-dir outputs/sintel --max-frames 50

# accurate profile (thesis models, improved algorithm) on a rectified stereo pair
export VFOA_THIRD_PARTY=/path/to/detectron2
python -m vfoa run --config configs/accurate.yaml --input LEFT.mp4 --right RIGHT.mp4 \
    --camera-dir CAMS/ --output-dir outputs/accurate

# thesis profile (thesis models, thesis-faithful bug-fixed algorithm)
python -m vfoa run --config configs/thesis.yaml --input LEFT.mp4 --right RIGHT.mp4 --output-dir outputs/thesis

# legacy entry point (defaults to the thesis profile; accepts --profile/--set/--max-frames)
python thesis_final/FinalTest.py -d VIDEO.mp4 -od outputs/legacy
```

Other `run` flags: `--profile fast|accurate|thesis|synthetic`, `--config FILE`,
`--input PATH|CAMERA_INDEX|synthetic`, `--right PATH`, `--camera-dir D` (Sintel
`.cam` intrinsics), `--depth-dir D` (Sintel `.dpt` depth), `--max-frames N`,
`--set section.key=value` (repeatable), `--realtime`, `--video`, `--show`, `-v`.
`--set runtime.lookahead_frames=9` gives Gaze360 a centred temporal window
(>= 3 x `gaze.window_step`) at the cost of 9 frames of latency.

Tests (models are always faked; network tests are opt-in):

```bash
python -m pytest -q
VFOA_RUN_NETWORK=1 python -m pytest -q -m network   # downloads YOLO / L2CS weights
```

## Conventions

- **One 3D frame**: the OpenCV pinhole frame of the left camera, right-handed, origin
  at the optical centre, +X right, +Y down, +Z forward, in **metres**.
  `u = fx X / Z + cx`, `v = fy Y / Z + cy`.
- **Depth** is Z > 0 (float32 metres). Invalid depth is 0 with `valid = False`; there
  are no inf/10000 sentinels and no negative Z.
- **Disparity** is real disparity in pixels, `d = fx B / Z` (Sintel baseline B = 0.1 m).
  A per-frame 0-255 normalised disparity is visualisation only (#1).
- **Images** are uint8 BGR everywhere; each backend converts to what its model expects (#9).
- **Boxes** are `[x1, y1, x2, y2]` pixels; **keypoints** are COCO-17 `(17, 3)` x, y, confidence.
- **Directions** (gaze, head) are unit vectors in the camera frame from the eyes towards
  what is looked at; a person looking at the camera has direction (0, 0, -1).
- **Head pose** (yaw, pitch, roll in degrees) -> forward =
  (-sin yaw, -cos yaw sin pitch, -cos yaw cos pitch) (#2); +yaw turns towards image
  left, +pitch looks up.
- **Gaze360** (yaw, pitch in radians) is converted by `vfoa.geometry.gaze360_to_camera`,
  with an optional eye-ray correction (`gaze.eye_frame_correction`).
- **Time**: `Frame.index` is the absolute 0-based source index; `t = index / fps` seconds.

## Configuration

`load_config` merges `DEFAULTS` (in `vfoa/config.py`) <- profile <- YAML file <-
`--set` overrides. `configs/{fast,accurate,thesis,synthetic}.yaml` mirror the
profiles and list the most useful knobs as comments; `configs/sintel_gt.yaml` is
the fast profile on the repository's Sintel clip. Every key read by the code is in
`DEFAULTS` (checked by `tests/test_benchmark.py`). Sections:

| Section | Main keys |
|---|---|
| `input` | `source` (auto/video/images/camera/synthetic), `path`, `right_path`, `camera_dir`, `depth_dir`, `start`, `max_frames`, `fps`, `realtime`, `pace` |
| `camera` | `fx`, `fy`, `cx`, `cy` (null -> from `hfov_deg`), `hfov_deg`, `baseline_m` |
| `depth` | `backend`, `stride`, `phase`, `scale` (stereo matching resolution), `num_disparities`, `block_size`, `wls`, `wls_lambda`, `wls_sigma`, `min_depth_m`, `max_depth_m`, `constant_depth_m`, `speckle_window`, `speckle_range` |
| `persons` | `backend`, `stride`, `phase`, `score_thr`, `kpt_thr`, `model`, `imgsz`, `head_scale`, `match_keypoints`; OpenPose/DensePose paths, `densepose_thr`, `upper_body_only`, `match_min_inside`, `keep_headless` |
| `tracking` | `type` (byte/greedy_iou), `box` (head/body), `iou_thr`, `max_age`, `high_thr`, `low_thr`, `motion`, `max_predict_frames` |
| `headpose` | `backend`, `stride`, `phase`, `max_hold_frames`, FSA-Net `models`; keypoints backend: `relaxed_kpt_thr`, `fallback_confidence`, `max_reproj_frac` |
| `gaze` | `backend`, `fallback`, `stride` (fast: 2), `phase`, `max_hold_frames`; Gaze360: `window`, `window_step`, `uncertainty_thr`, `eye_frame_correction`, `weights`, `batch_size`; `proxy_confidence` (head_proxy); `l2cs.{model, path, url, sha256, download, input_size, kpt_thr, crop_margin, perspective_correction, providers}` |
| `segmentation` | `backend`, `stride`, `phase`, `async`, `score_thr`, `model`, `imgsz`, Panoptic FPN paths, `panoptic_thr` |
| `scene` | `point_stride`, `max_points_per_object`, `min_points`, `outlier` (percentile/radius/none), `outlier_percentiles`, `outlier_nb_points`, `outlier_radius_m`, `outlier_radius_scale`, `ignore_labels`, `person_labels`, `self_min_overlap`, `self_aabb_margin_m`, `eye_patch_px`, `match_dist_m`, `object_max_age` |
| `attention` | `method` (angular/rays), `sigma_gaze_deg`, `sigma_head_deg`, `fusion` (fixed/agreement/confidence), `w_gaze`, `w_head`, `agreement_deg`, `target` (surface/centroid), `occlusion` (depth/aabb/none), `occlusion_margin_m`, `occlusion_samples`, `min_target_distance_m`, `distance_full_m`, `distance_decay_m`, `min_score`, `rays.{n, half_angles_gaze_deg, half_angles_head_deg}` |
| `smoothing` | `direction` (kalman/ema/none), `ema_alpha`, `kalman_q`, `kalman_r`, `max_hold_s`, `jump_deg`, `target` (hmm/hysteresis/none), `hmm_stay_prob`, `hysteresis_frames` |
| `cache` | `enabled`, `dir`, `mode` (readwrite/read/write), `stages`, `chunk_size` |
| `output` | `dir`, `jsonl`, `csv`, `video`, `show` |
| `models` | `device` (auto/cpu/cuda), `weights_dir`, `allow_download`, `third_party_root` |
| `runtime` | `lookahead_frames`, `num_threads`, `seed`, `log_every` |
| `timing` | `enabled` |
| `synthetic` | `width`, `height`, `fps`, `n_frames`, `n_persons`, `n_objects`, `seed`, `noise_deg`, `occluder`, `cost_ms`, `sway_m` |

**Multi-rate scheduling.** A model stage S runs on frame i when
`(i - S.phase) % S.stride == 0` (or when it has no result yet); otherwise its last
result is held (head/gaze estimates for at most `max_hold_frames`). Tracking runs
whenever persons runs; scene, smoothing and attention run on every frame.

## L2CS gaze backend (fast profile)

`gaze.backend: l2cs` (`vfoa/backends/fast/l2cs_gaze.py`) is an L2CS-style binned
yaw/pitch gaze model run with ONNX Runtime. The default model is MobileGaze
MobileOne-S0 at 448 px, on a square face crop gated on the nose plus one eye.

- **Fallback.** The fast profile sets `gaze.fallback: head_proxy`: when onnxruntime or
  the weights are unavailable, the head direction is used as gaze. The switch is
  logged and counted (`gaze_fallback` in `timing.json`); the benchmark prints a NOTE.
- **Weights download.** With `models.allow_download` and `gaze.l2cs.download` (both
  default true) `mobileone_s0_gaze.onnx` is downloaded from the MobileGaze GitHub
  release (`yakhyo/gaze-estimation`, tag `weights`) into `models.weights_dir` (or
  `$VFOA_WEIGHTS_DIR`) and its sha256 is checked. Other models: `gaze.l2cs.model`
  (`resnet18`, `resnet34`, `resnet50`, `mobilenetv2`; no pinned hash) or an explicit
  `gaze.l2cs.path`.
- **Licence.** The MobileGaze code is MIT, but the weights are trained on Gaze360,
  whose terms allow **research / non-commercial use only**. The weights are never
  committed (`weights/` is git-ignored); do not redistribute them.
- **Known limitations.** The yaw/pitch-to-camera sign convention was derived from the
  code: it broadly agrees with the keypoint head direction on the real clips, but
  differs by 25-60 degrees on the Sintel CG clip, so it is plausible but **not
  confirmed**. The entropy-based confidence is low (about 0.2-0.26), so
  `attention.fusion: confidence` under-weights gaze. The released ONNX files have a
  static batch of 1 (one call per person, about 20-35 ms per person on this CPU),
  hence `gaze.stride: 2` in the fast profile. Persons seen from behind get no L2CS
  gaze.

## Cache and re-scoring without models

With `cache.enabled` every model stage (depth, persons, headpose, gaze,
segmentation) stores its per-frame output under `<output.dir>/cache` (`.npz` chunks
+ `meta.json`). Cache keys depend only on the input and the stage's own settings, so
attention, smoothing and scene settings can be changed and re-scored from the cache
without loading any model:

```bash
python -m vfoa run --config configs/sintel_gt.yaml --output-dir outputs/sintel --set cache.enabled=true
python -m vfoa run --config configs/sintel_gt.yaml --output-dir outputs/sintel \
    --set cache.enabled=true --set cache.mode=read \
    --set attention.sigma_gaze_deg=6 --set attention.fusion=fixed
```

## Real-time mode

```bash
python -m vfoa run --profile fast --input 0 --realtime --set segmentation.async=true --show
```

- `--realtime` (`input.realtime`) wraps the source in a latest-frame reader: frames
  the pipeline cannot keep up with are dropped instead of queued. A file source is
  replayed at its native fps (`input.pace`).
- `--set segmentation.async=true` runs segmentation on a worker thread; the frame loop
  never waits for it and uses the previous segmentation meanwhile.
- `--set runtime.lookahead_frames=0` (the default) gives one result per input frame
  with no added latency.
- `timing.json` holds the per-stage statistics (`stages`: runs, mean/p50/p95/max ms,
  amortised `per_frame_ms`, `share`), `fps`, `setup_ms` (model load times),
  `counters` (`dropped_frames`, `captured_frames`, `gaze_fallback`,
  `segmentation_async_*`), `latency` (capture-to-result ms), `async` (worker
  statistics), `cache`, `backends` and `wall_s`.
- CPU preset (see `configs/fast.yaml`): `--set persons.stride=2 --set gaze.stride=4
  --set segmentation.stride=30 --set segmentation.async=true`.

## Benchmark

`benchmarks/bench_pipeline.py` runs N frames after a warm-up and prints the per-stage
table (runs, mean/p50/p95/max ms, amortised ms per frame, share, FPS), and writes a
JSON result (default `benchmarks/results/<profile>-<input>.json`, git-ignored):

```bash
python benchmarks/bench_pipeline.py --profile synthetic --frames 60 --warmup 5
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/museum.mp4 --frames 30 --warmup 5
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/mpi_sintel.mp4 \
    --camera-dir Final_videos/stereo/camdata_left/bandage_2 \
    --depth-dir Final_videos/stereo/training/bandage_2_depth --set depth.backend=sintel_gt --frames 30
python benchmarks/bench_pipeline.py --profile synthetic --input Final_videos/mpi_sintel.mp4 \
    --camera-dir Final_videos/stereo/camdata_left/bandage_2 \
    --depth-dir Final_videos/stereo/training/bandage_2_depth \
    --stereo-synth --set depth.backend=stereo_bm --set depth.scale=0.5 --frames 20
python benchmarks/bench_pipeline.py --frames 0 --micro     # #32 micro-benchmarks only
```

`--set` overrides any key (strides, `segmentation.async`, `input.realtime`, ...);
`--stereo-synth` renders the right view from ground-truth depth so `stereo_bm` can be
measured without stereo footage; `--micro` adds the micro-benchmarks of the #32 fixes;
`--markdown` also prints Markdown tables. When a backend or its weights are
unavailable the benchmark prints `SKIPPED: <reason>`, writes `status: skipped` and
exits 0. Measured CPU numbers are in [docs/performance.md](docs/performance.md);
GPU numbers are to be measured with the same commands.

## Fixed issues

Implemented and unit-tested on the local branch `realtime-integration` (CPU-only;
the accurate backends were not run with the real thesis models, which were not
available on the development machine).

| Issue | Fix |
|---|---|
| #1 | Real disparity in pixels and metric depth (`d = fx B / Z`); no 0-255 normalisation before reprojection |
| #2 | FSA-Net Euler angles converted to a unit forward vector (`geometry.euler_to_forward`) |
| #3 | Gaze/head scores normalised first, then a real weighted sum (`fixed`, `agreement`, `confidence` fusion) |
| #4 | `calc_ratio` uses `bbox[0] - bbox[1]` |
| #5 | Ray-box slab test requires `t >= 0`: objects behind the viewer are not hit |
| #6 | Each person's own segment is excluded from their attention targets |
| #7 | Gaze360 receives a real temporal window of distinct frames |
| #8 | FSA-Net input normalisation `(x - 127.5) / 128` |
| #9 | BGR/RGB order handled inside each backend |
| #12, #15 | Streaming pipeline replaces the broken multi-batch path; legacy entry points rewritten over it |
| #14 | Missing gaze, head pose or head box never raises (the cue's weight goes to the other cue) |
| #16 | Eye depth from a robust patch near borders; cone basis valid for any direction; no duplicated ray |
| #17 | Python >= 3.10 and NumPy 2 compatible (no float16 timestamps, no removed aliases) |
| #19 | Angular (probabilistic) attention scoring, the default `attention.method` |
| #20 | Occlusion: depth-based (default) or AABB-based; occluded pairs score 0 |
| #23 | Kalman/EMA direction smoothing and an HMM/hysteresis attention-target smoother |
| #28 | Per-stage output cache and re-scoring without models |
| #30 | Unit tests (`tests/`, models faked) and a CI workflow (`.github/workflows/tests.yml`) |
| #32 | No PLY round-trip, a single (optionally scaled) StereoBM, models loaded once, streamed video (see `--micro`) |
| #35 | Fast backends, multi-rate stages, real-time mode, asynchronous segmentation, benchmark |
| optional | #10 detectors see the original image, #11 replaced by the scene builder, #13 single device, no process pools, #18 ByteTrack-style tracking, #21 confidence-weighted fusion, #31 structured JSONL/CSV outputs with frame timestamps |

Note: adding the reconciled config keys to `DEFAULTS` changes the stage-cache keys
once, so caches written before this change are recomputed.
