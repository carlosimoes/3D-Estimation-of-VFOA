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
- **Gaze360 / L2CS** (yaw t, pitch p in radians) use the Gaze360 dataset eye frame:
  right-handed, +x points to the image left, +y up, +z away from the camera, with
  g = (cos p sin t, sin p, -cos p cos t). `vfoa.geometry.gaze360_to_camera` converts it
  to the camera frame as (-g.x, -g.y, g.z) (a rotation, not a mirror): t = p = 0 looks
  into the camera, +t turns towards the image left, +p looks up.
- **Eye-ray correction** (`vfoa.geometry.ray_aligned`): crop-based models predict a
  direction relative to the camera -> eye ray, so the direction is rotated into that
  ray's frame. It applies to gaze360 (`gaze.eye_frame_correction`), l2cs
  (`gaze.l2cs.perspective_correction`, default true) and FSA-Net
  (`headpose.eye_frame_correction`); all three default to true.
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
| `headpose` | `backend`, `stride`, `phase`, `max_hold_frames`, FSA-Net `models`, `eye_frame_correction`; keypoints backend: `relaxed_kpt_thr`, `fallback_confidence`, `max_reproj_frac` |
| `gaze` | `backend`, `fallback`, `stride` (fast: 2), `phase`, `max_hold_frames`; Gaze360: `window`, `window_step`, `uncertainty_thr`, `eye_frame_correction`, `weights`, `batch_size`; `proxy_confidence` (head_proxy); `l2cs.{model, path, url, sha256, download, input_size, kpt_thr, crop_margin, perspective_correction, confidence_ref, providers}` |
| `segmentation` | `backend`, `stride`, `phase`, `async`, `score_thr`, `model`, `imgsz`, Panoptic FPN paths, `panoptic_thr` |
| `scene` | `point_stride`, `max_points_per_object`, `min_points`, `outlier` (percentile/radius/none), `outlier_percentiles`, `outlier_nb_points`, `outlier_radius_m`, `outlier_radius_scale`, `ignore_labels`, `person_labels`, `self_min_overlap`, `self_aabb_margin_m`, `eye_patch_px`, `match_dist_m`, `object_max_age` |
| `attention` | `method` (angular/rays), `sigma_gaze_deg`, `sigma_head_deg`, `fusion` (fixed/agreement/confidence), `w_gaze`, `w_head`, `agreement_deg`, `target` (surface/centroid), `occlusion` (depth/aabb/none), `occlusion_margin_m`, `occlusion_samples`, `min_target_distance_m`, `distance_full_m`, `distance_decay_m`, `min_score`, `rays.{n, half_angles_gaze_deg, half_angles_head_deg}` |
| `smoothing` | `direction` (kalman/ema/none), `ema_alpha`, `kalman_q`, `kalman_r`, `max_hold_s`, `jump_deg`, `target` (hmm/hysteresis/none), `hmm_stay_prob`, `hysteresis_frames` |
| `cache` | `enabled`, `dir`, `mode` (readwrite/read/write), `stages`, `chunk_size` |
| `output` | `dir`, `jsonl`, `csv`, `video`, `show` |
| `models` | `device` (auto/cpu/cuda), `weights_dir`, `allow_download`, `third_party_root` |
| `runtime` | `lookahead_frames`, `num_threads` (null: library defaults, except that on CPU the YOLO backends cap torch at half the logical CPUs, a quarter with `segmentation.async`; an integer sets the torch threads of the YOLO models and `intra_op_num_threads` of the ONNX Runtime sessions, l2cs and FSA-Net), `seed`, `log_every` |
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
- **Sign convention (verified).** The model outputs Gaze360-convention yaw/pitch (see
  Conventions), converted by `vfoa.geometry.gaze360_to_camera`. Sources: the output
  order and `draw_gaze` of the upstream MobileGaze `onnx_inference.py`; the Gaze360
  dataset README (eye frame with +x to the image left, +y up, +z away from the camera);
  and, on the repository clips got.mp4, museum.mp4 and video.mp4, agreement with the
  nose-offset head-turn cue (correlation +0.63, 89% sign agreement; verification,
  2026-11-16).
- **Sintel.** On the Sintel CG clip L2CS and the keypoint head direction still differ
  by 25-60 degrees. This is not a sign error: the character's head is turned left
  while the eyes look down-right, and on a CG face the model output is unstable from
  frame to frame.
- **Eye-ray correction.** `gaze.l2cs.perspective_correction` (default true) rotates the
  gaze into the frame of the camera -> eye ray (the eye pixels, else the crop centre).
- **Confidence.** The raw entropy certainty of the 4-degree bins sits in a narrow band
  (about 0.18-0.37, median 0.24 on the repository clips), so it is rescaled:
  confidence = clip(raw / `gaze.l2cs.confidence_ref`, 0, 1) x face visibility, with
  `confidence_ref` 0.25 by default; null or <= 0 keeps the raw value (which makes
  `attention.fusion: confidence` under-weight gaze).
- **Cost.** The released ONNX files have a static batch of 1 (one call per person). On
  the 8-vCPU test machine, when it was quiet, L2CS cost about 8-15 ms per gaze run on
  museum.mp4 (benchmark-docs, 2026-11-16), hence `gaze.stride: 2` in the fast
  profile. The session disables ONNX Runtime's intra-op spin-waiting so its threads
  do not compete with torch (see
  [docs/performance.md](docs/performance.md#cpu-threading)).
- A person who does not pass the face gate (nose plus one eye, `gaze.l2cs.kpt_thr`) gets
  no L2CS gaze; attention then uses the head direction alone when a head pose exists.
  Seen from behind, the keypoint head pose of the fast profile has no estimate either
  (it needs the nose plus three more face keypoints), so that person is not scored.

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
- Live camera input (`--input 0`) is monocular, so depth is `constant` (see Known
  limitations). Stereo depth needs files (`--input LEFT --right RIGHT`) or a custom
  `FrameSource` that fills `Frame.right` with the rectified right image.
- Strides are counted in source frames, not in processed frames: when frames are
  dropped, `segmentation.stride: 10` means every 10 / fps seconds, so on a slow CPU
  segmentation (and gaze, stride 2) can be due on almost every processed frame. Raise
  the strides when the processing rate is far below the source rate.
- `--set segmentation.async=true` runs segmentation on a worker thread; the frame loop
  never waits for it and uses the previous segmentation meanwhile.
- `--set runtime.lookahead_frames=0` (the default) gives one result per input frame
  with no added latency.
- `timing.json` holds the per-stage statistics (`stages`: runs, mean/p50/p95/max ms,
  amortised `per_frame_ms`, `share`), `fps`, `setup_ms` (model load times),
  `counters` (`dropped_frames`, `captured_frames`, `gaze_fallback`,
  `segmentation_async_*`), `latency` (capture-to-result ms), `async` (worker
  statistics), `cache`, `backends` and `wall_s`.
- Speed on the 8-vCPU CPU-only test machine (museum.mp4, 1280x720, fast profile,
  default strides, not real-time mode): 18.5 FPS on a quiet machine (13-19 FPS over
  repeated runs), but 3-8 FPS when other CPU-bound jobs share the machine. See
  [docs/performance.md](docs/performance.md#run-to-run-variance).
- Real-time mode on the same machine (`input.realtime=true`, `segmentation.async=true`,
  museum.mp4 replayed at 23.98 fps): 6.0-8.3 FPS processed with the current thread
  default, against 0.8-1.3 FPS with the old 4-thread default (one run at a time,
  2026-11-16; see [docs/performance.md](docs/performance.md#real-time-mode)).
- CPU preset (see `configs/fast.yaml`): `--set persons.stride=2 --set gaze.stride=4
  --set segmentation.stride=30 --set segmentation.async=true`: 17-18 FPS on museum.mp4
  with the current thread default (19-24 FPS with `OMP_WAIT_POLICY=PASSIVE`), without
  the persons spikes seen before the async thread cap, so PASSIVE is no longer needed
  to avoid them. In the two pairs measured it was still 40% and 2% faster with async
  segmentation; with the default strides (synchronous segmentation) it was slower
  (11.99 against 18.50 FPS). See
  [docs/performance.md](docs/performance.md#cpu-preset).
- CPU threads: on CPU the YOLO backends cap torch at half the logical CPUs, or a
  quarter when `segmentation.async` is true, because the frame loop and the
  segmentation worker then run two torch thread pools (ultralytics would otherwise use
  `cpu_count - 1`, which spin-waits and was 25x slower on a shared VM).
  `runtime.num_threads` overrides this; **when the machine is shared, use
  `--set runtime.num_threads=2`** (under load it beat the 4-thread default: 7.9 against
  5.1-5.8 FPS, verification). See [docs/performance.md](docs/performance.md#cpu-threading).

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

## Known limitations

Numbers marked "(verification, 2026-11-16)" come from the verification run of the
`realtime-integration` branch on the 8-vCPU CPU-only test machine.

- **Constant depth gives little 3D attention.** With `depth.backend=constant` (the
  default of the fast profile on monocular video) the eyes and every object lie in one
  fronto-parallel plane, so a target is found for only 9-28% of the person-frames on
  the repository clips (verification, 2026-11-16). Meaningful 3D attention needs real
  depth: `stereo_bm` (a stereo pair) or `sintel_gt` (Sintel ground truth). A monocular
  depth backend does not exist yet.
- **The thesis ray method misses small objects.** `attention.method=rays` (the thesis
  profile) casts rays on rings at 11.3 and 21.8 degrees around the gaze (12.5 and 24.0
  around the head direction), with no centre ray, as in the thesis. An object that
  subtends less than the inner ring and is looked at directly gets no hits: 0 of 240 targets on the synthetic scene
  (verification, 2026-11-16). Use angular scoring (`attention.method: angular`, the
  default of the other profiles).
- **The accurate backends were never run with the real thesis models** (detectron2,
  DensePose, OpenPose, Gaze360, FSA-Net are not on the development machine); they are
  tested with fakes only.
- **YOLO11n-seg only knows the COCO classes**, so objects outside the 80 COCO classes
  are not segmented and cannot be attention targets in the fast profile.
- **Head pose yaw.** The keypoint head pose seems to overestimate yaw compared with
  l2cs gaze (regression slope of l2cs yaw on head yaw 0.34-0.57 on the repository
  clips; verification, 2026-11-16). This is not validated against ground truth.
- **Fallback naming.** After a gaze fallback (for example l2cs -> head_proxy) the
  written `config.yaml` still names the requested backend; the `backends` entry of
  `timing.json` names the backend that actually ran (and `gaze_fallback` counts the
  switch).
- **Timing of the `output` stage.** The `output` stage shows N-1 runs for N frames,
  because the last frame's output time is added after its record is closed.

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
| #2, #7 (follow-up) | Gaze360 axes: the eye frame has +x to the image left, so the camera vector is (-g.x, -g.y, g.z) (a rotation; it was mirrored left-right); gaze360, l2cs and FSA-Net share one eye-ray correction (`geometry.ray_aligned`) |
| #35, #32 (follow-up) | CPU torch thread cap: a quarter of the logical CPUs with `segmentation.async` (two torch pools), half otherwise; real-time mode on museum.mp4 went from 0.8-1.3 to 6.0-8.3 FPS in this update's runs (see [docs/performance.md](docs/performance.md#real-time-mode)) |
| optional | #10 detectors see the original image, #11 replaced by the scene builder, #13 single device, no process pools, #18 ByteTrack-style tracking, #21 confidence-weighted fusion, #31 structured JSONL/CSV outputs with frame timestamps |

Note: adding the reconciled config keys to `DEFAULTS` changes the stage-cache keys
once, so caches written before this change are recomputed.
