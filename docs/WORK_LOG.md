# Work log: code review and real-time refactor

This log records the work done on the thesis repository `3D-Estimation-of-VFOA`
on 2026-11-13 and 2026-11-14: the code review, the GitHub issues, the comparison
with BrandPose, the latency analysis, the multi-agent refactor into the `vfoa`
package, the gaze-model research, the new gaze backend and the CPU measurements.
Umbrella tracking issue: #37.

## 1. Overview

**Goal.** Fix the correctness bugs found in the original thesis code
(`thesis_final/`) and turn the offline pipeline into a streaming, near-real-time
one. The thesis models stay available, and fast models can be swapped in.

**Current state**

1. 14 of the 15 planned tasks, plus a later gaze-backend task, are merged into the local branch `realtime-integration`.
2. The new `vfoa` package (about 19,400 added lines with tests) streams frames through pluggable backends: `fast`, `accurate`/`thesis` and `synthetic`.
3. The test suite passes on this CPU-only machine: 615 passed, 2 skipped (network tests).
4. The last planned task (`benchmark-docs`), the final verification and the final report did not run, because the Fable spend limit was reached.
5. Latency has been measured on CPU only. The accurate (thesis-model) backends have not been run with the real models.

**Where the code lives**

- Branch: `realtime-integration`, tip `909decc`. It is 36 commits ahead of `main`: 99 files changed, 21,945 insertions and 3,303 deletions.
- Integration worktree: `/mnt/c/Users/CSIMOES/orca/3D-Estimation-of-VFOA-integration`.
- Per-task branches: `task/*`, one per task (section 7).
- This log is on the local branch `docs/work-log`, which is based on `realtime-integration`.

**Nothing has been pushed.** All branches are local. The workflows were not allowed to create, edit or comment on GitHub issues or PRs, and all 37 issues are still open.

## 2. Timeline

Times are UTC (the same as Europe/London in November) and come from the
workflow run metadata where available.

**2026-11-13**

- **Code review** of `thesis_final/` (about 4,800 lines) at commit `8356bf8`. Several findings were checked numerically (section 3).
- **37 GitHub issues** filed: 36 findings plus the umbrella issue #37.
- **BrandPose comparison** with `escudero/brandpose` (section 5).
- **Latency analysis** of the offline design. Main costs:
  - the whole video is read into RAM;
  - models are reloaded on every call;
  - a PLY file round-trip;
  - Gaze360's centred window needs future frames.

  Recommended changes: streaming, loading each model once, multi-rate stages, vectorised geometry, faster backends and TensorRT.
- **User decisions** for the refactor:
  - Scope: correctness plus real-time, that is issues #1-#9, #12, #14-#17, #19, #20, #23, #28, #30, #32 and #35. Optional: #10, #11, #13, #18, #21 and #31.
  - Pluggable backends: fast models plus the accurate thesis models.
  - Local branches only; nothing is pushed.
  - Work runs as a background workflow.
- **17:54: the orchestrated refactor starts** (run `wf_0d7c4b6c-dc3`). Fable 5.2 plans the task graph. The `foundation` task is planned, built and merged.
- The run is paused by a usage limit ("resets at 7pm") and later by a second one ("resets at 12am"). The machine also slept twice during the run.

**2026-11-14**

- The run continues. The remaining tasks up to `legacy-entry` are reviewed and merged; the last merge is `6cfbad2`.
- **About 11:37:** the Fable spend limit is reached. Planning for `benchmark-docs`, the verification step and the final report all fail. The Fable limit resets on 2026-11-17 at 13:00 Europe/London, and usage-credit requests are turned off for the organisation.
- **11:38-11:39:** the main session measures CPU timing for the synthetic profile and for the fast profile on the Sintel clip (section 8).
- **12:17-12:29: gaze research run** (`wf_4de5793e-5e0`, 14 agents). Its report recommends an L2CS-style backend with MobileGaze weights.
- **14:32-14:52: gaze backend run** (`wf_ddf48012-ef5`, Opus only). The `l2cs` backend is merged as `909decc`.
- **14:46-14:56:** real-model checks: L2CS on the Sintel clip, the fallback path, and sign checks on `got`, `museum` and `video`.
- **14:57-15:04: failed resume** of the main run. Fable was still limited, and the cache missed for the parallel steps, so implementation steps re-ran and moved the `task/*` branches. The main session restored every branch (section 7.5).
- This work log is written.

## 3. Code review findings

The original code is in `thesis_final/`, about 4,800 lines, reviewed at commit `8356bf8`.

| Area | Finding | Issue |
|---|---|---|
| Depth | Disparity normalised to 0-255 per frame before 3D reprojection | #1 |
| Head pose | Euler angles (yaw, pitch, roll in degrees) used as a 3D direction vector | #2 |
| Attention | Gaze and head weights cancel out in the normalisation, so there is no real weighted sum. Verified: 0.5385 vs 0.5385. | #3 |
| Attention | `calc_ratio` typo (`[0]-bbox[1]`) makes the normalisation negative. Verified: about -142.5. | #4 |
| Attention | The ray-box test counts objects behind the viewer (no `t >= 0` check). Verified: a box behind the origin is hit. | #5 |
| Attention | Every person attends to themselves (their own segment is never excluded) | #6 |
| Models | Gaze360 is fed the same frame 7 times, so the temporal model gets no temporal context | #7 |
| Models | FSA-Net input normalisation `(x-447.5)/448` instead of 127.5/128 | #8 |
| Models | BGR/RGB mix-ups: RGB fed to the BGR detectron2 models, BGR crops fed to Gaze360 | #9 |
| Depth/models | Detectors run on frames with invalid-disparity pixels blacked out | #10 |
| Depth | Invalid-bbox filter uses `and` instead of `or` | #11 |
| Pipeline | The multi-batch path is broken for videos longer than `-b` frames | #12 |
| Pipeline | The multi-GPU path in the DensePose and Panoptic predictors cannot work | #13 |
| Attention | KeyError when a person has no gaze, head pose or head box | #14 |
| Pipeline | Crash and edge-case bugs in VideoProcessor, Reconstruction and FinalTest | #15 |
| Geometry | Eyes near the image border; cone parallel to the x-axis; the first and last cone rays are duplicates (verified) | #16 |
| Engineering | Python >= 3.10 and NumPy 2 problems; timestamps become `inf` under NumPy 2 | #17 |
| Tracking | IDs are matched to the previous frame only; collisions drop detections | #18 |
| Attention | The Kalman filter is implemented but never used | #23 |
| Engineering | Hard-coded `/home` paths, no requirements file, no config file | #29 |
| Engineering | No unit tests | #30 |
| Performance | PLY round-trip, duplicate StereoBM, model reloads, whole video in RAM | #32 |

The other issues (#19-#22, #24-#28, #31, #33-#36) are improvements, evaluation work and repository hygiene (section 4).

## 4. GitHub issues

All 37 issues are open on GitHub. Their permalinks point to commit `8356bf8`.

"Status" is the state on `realtime-integration`, based on the task-to-issue
mapping in the plan and the task reports. **Every status below is per task
reports and not yet verified by Fable**: the verification step did not run. All
code was written against fakes on a CPU-only machine. "Resolved (code)" means
the fix is implemented and unit-tested but has not been run with the real
thesis models.

| # | Title | Priority | Area | Scope | Status on realtime-integration | Tasks |
|---|---|---|---|---|---|---|
| 1 | Depth: 3D reprojection uses min-max normalized 0-255 disparity instead of real disparity | P1 | depth | in | Resolved | geometry, depth, scene |
| 2 | Head pose: yaw/pitch/roll angles (degrees) are used as a 3D direction vector | P1 | attention, models | in | Resolved | geometry, accurate-gaze-headpose |
| 3 | Attention: gaze/head weights cancel out during normalization (no weighted sum) | P1 | attention | in | Resolved | attention |
| 4 | Attention: calc_ratio uses `[0]-bbox[1]` instead of `bbox[0]-bbox[1]` | P1 | attention | in | Resolved | attention |
| 5 | Attention: ray-box test counts objects behind the viewer (no t >= 0 check) | P1 | attention | in | Resolved | geometry, attention |
| 6 | Attention: every person attends to themselves (own segment is never excluded) | P1 | attention | in | Resolved | attention, scene |
| 7 | Gaze360 receives the same frame 7 times (temporal model gets no temporal context) | P1 | models | in | Resolved (code) | accurate-gaze-headpose |
| 8 | FSA-Net input normalization `(x-447.5)/448` looks wrong (reference uses 127.5/128) | P1 | models | in | Resolved (code) | accurate-gaze-headpose |
| 9 | Color channel order: RGB fed to BGR detectron2 models; BGR crops fed to Gaze360 | P1 | models | in | Resolved (code) | accurate-detectron, accurate-gaze-headpose |
| 10 | Detectors run on frames where pixels with invalid disparity are blacked out | P2 | depth, models | optional | Resolved | depth |
| 11 | Reconstruction: invalid-bbox filter uses `and` (needs both corners invalid) instead of `or` | P2 | depth | optional | Resolved (replaced by the scene builder) | scene |
| 12 | Multi-batch processing path is broken (videos longer than `-b` frames) | P2 | pipeline | in | Resolved (streaming replaces batches) | tracking-smoothing, io, pipeline, legacy-entry |
| 13 | Multi-GPU path in DensePose/Panoptic predictors cannot work | P2 | pipeline | optional | Resolved (code; single device, no process pools) | accurate-detectron |
| 14 | KeyError when a person has no gaze, head pose or head box | P2 | attention | in | Resolved | attention, scene, pipeline |
| 15 | Assorted crash/edge-case bugs in VideoProcessor, Reconstruction and FinalTest | P2 | pipeline | in | Resolved (legacy entry points rewritten) | io, legacy-entry |
| 16 | Geometry edge cases: eyes near the image border, cone parallel to the x-axis, duplicated ray | P2 | attention | in | Resolved | geometry, scene |
| 17 | Compatibility with Python >=3.10 and NumPy 2 | P2 | engineering | in | Resolved | foundation, tracking-smoothing, depth, io, accurate-gaze-headpose, legacy-entry |
| 18 | Tracking: IDs only matched to the previous frame; collisions drop detections | P2 | tracking | optional | Resolved | tracking-smoothing, accurate-detectron |
| 19 | Attention: replace ray-hit counting with angular (probabilistic) scoring | P2 | attention | in | Resolved | attention |
| 20 | Attention: handle occlusion (only the first surface along the gaze should count) | P2 | attention | in | Resolved (one known gap, section 10) | attention |
| 21 | Attention: confidence-weighted gaze/head fusion using Gaze360 uncertainty | P2 | attention, models | optional | Partial: implemented, but L2CS confidence is low and uncalibrated | attention |
| 22 | Attention: tighter object geometry than axis-aligned bounding boxes | P3 | depth, attention | out | Open. The angular method scores sampled surface points, not boxes, which goes part of the way. | - |
| 23 | Temporal smoothing: integrate the Kalman filter and add a temporal model over the attention target | P2 | attention | in | Resolved | tracking-smoothing, pipeline |
| 24 | Evaluate newer models per stage (stereo/depth, head detection, head pose, gaze target) | P3 | models | out | Open. YOLO11 and L2CS backends were added, but no evaluation was done. | - |
| 25 | Evaluation: create ground-truth attention annotations and report accuracy | P2 | evaluation | out | Open | - |
| 26 | Evaluation: ablation of cues and fusion strategies | P2 | attention, evaluation | out | Open. The config switches exist; no ablation was run. | - |
| 27 | Evaluation: measure depth accuracy against Sintel ground-truth depth | P2 | depth, evaluation | out | Open. Only unit-level checks against the ground truth exist. | - |
| 28 | Cache per-stage outputs so attention can be re-run without the models | P2 | pipeline, engineering | in | Resolved | synthetic-cache, pipeline |
| 29 | Reproducibility: config file, pinned environment, remove hard-coded absolute paths | P2 | engineering | out | Partial (side effect): YAML configs, requirements files and no hard-coded paths; versions are not pinned | foundation |
| 30 | Add unit tests for the geometry and attention code | P1 | attention, engineering | in | Resolved (615 tests). The CI workflow from `benchmark-docs` is not done. | foundation, geometry, synthetic-cache, attention |
| 31 | Structured outputs with frame-based timestamps instead of str(dict) lines and long-form CSV | P3 | pipeline | optional | Resolved | io, legacy-entry |
| 32 | Performance: drop the PLY round-trip, duplicate StereoBM, model reloads and in-RAM video | P3 | engineering | in | Resolved; the performance doc from `benchmark-docs` is not done | depth, io, accurate-detectron, scene, accurate-gaze-headpose, pipeline, legacy-entry |
| 33 | Remove dead and duplicated code | P3 | engineering | out | Open | - |
| 34 | Repo hygiene: keep media/data out of git and normalize line endings | P3 | engineering | out | Open | - |
| 35 | Real-time / on-robot mode | P3 | pipeline | in | Partial: streaming, real-time mode and fast backends exist; GPU latency and the benchmark are not done | foundation, fast-backends, pipeline, realtime, l2cs-gaze |
| 36 | README: align claims with the implementation | P2 | documentation | out | Open. The README is unchanged. | - |
| 37 | Tracking: code review findings and improvement plan (umbrella) | - | documentation | - | Open | - |

## 5. Comparison with BrandPose (escudero/brandpose)

| Aspect | BrandPose | This project (VFOA) |
|---|---|---|
| Purpose | Retail analytics: aggregate seconds of attention per shelf | HRI: the visual focus of attention of each person, for a robot |
| Gaze model | Gaze-LLE gaze-target heatmaps (target-based) | A 3D gaze and head direction per person (direction-based) |
| Depth | Single-image metric depth | Stereo depth |
| Scene | Hand-built store model | Panoptic objects from segmentation plus depth |
| Out-of-frame targets | Not possible: a target-based model can only point inside the image | Possible: a direction can point at the robot or the camera |

The key difference is direction-based versus target-based gaze. Target-based
gaze cannot handle out-of-frame targets such as the robot.

**Ideas borrowed:**

- angle-based scoring (#19);
- occlusion (#20);
- self-exclusion (#6);
- confidence weighting (#21);
- ByteTrack-style tracking plus fragment merging (#18);
- stage caching (#28);
- a pinned environment (#29).

## 6. Architecture of the `vfoa` package

The architecture was written by Fable 5.2 as a shared contract for all tasks
(about 36,000 characters). It covers:

- **Conventions.** There is one 3D frame: the OpenCV left-camera frame (+X right, +Y down, +Z forward), in metres.
- **Depth.** Depth is positive Z. Disparity is real disparity in pixels. Invalid depth is 0 with a mask; there are no `inf` sentinels.
- **Images.** Images are uint8 BGR everywhere, and each backend converts to what its model needs.
- **Directions.** Directions are unit vectors from the eyes. A person looking at the camera has direction (0, 0, -1).
- **Head pose.** FSA-Net Euler angles are converted with `euler_to_forward`.
- **Gaze360.** Gaze360 yaw and pitch are converted with `gaze360_to_camera`, with an optional eye-ray correction.

### 6.1 Modules

| Module | Role |
|---|---|
| `vfoa/types.py` | Dataclasses: `Intrinsics`, `Frame`, `DepthFrame`, `PersonDetection`, `DirectionEstimate`, `Person`, `Segmentation`, `SceneObject`, `AttentionScore`, `FrameResult` |
| `vfoa/config.py` | `Config` with attribute access. Merge order: `DEFAULTS` <- profile <- YAML <- overrides. Also `config_hash` and `dump_config`. |
| `vfoa/timing.py` | `StageTimer`: per-stage mean, p50, p95 and max; amortised ms per frame; setup costs; counters |
| `vfoa/geometry.py` | Vectorised, NaN-free camera math: unprojection, disparity and depth, eye pixel, Euler and gaze conversions, unique cone rays, slab ray-AABB test with `t >= 0` |
| `vfoa/attention.py` | Angular kernel scoring (default) or the bug-fixed thesis ray-cone method. Also fusion (`fixed`, `agreement`, `confidence`), self-exclusion and occlusion (depth or AABB). |
| `vfoa/tracking.py` | `PersonTracker`: two-stage ByteTrack-style Hungarian matching; ids persist and are never reused |
| `vfoa/smoothing.py` | `DirectionSmoother` (Kalman/EMA, based on the corrected thesis `KalmanFilter3D`) and `TargetSmoother` (HMM or hysteresis over the attention target) |
| `vfoa/scene.py` | `SceneBuilder`: metric objects from segmentation plus depth on a strided grid; robust 3D eyes; own-segment matching; persistent object ids (`ObjectRegistry`). No PLY files. |
| `vfoa/cache.py` | Per-stage output cache (`.npz` chunks plus `meta.json`). The keys depend only on the inputs and the stage's own config. |
| `vfoa/pipeline.py` | The streaming `Pipeline` (section 6.3) |
| `vfoa/realtime.py` | `LatestFrameSource` (keeps only the newest frame and counts dropped frames) and `AsyncStageRunner` (runs a slow stage on a worker thread) |
| `vfoa/io/` | `sources.py` (video, image directory, camera; streamed), `outputs.py` (JSONL, CSV, overlay video, mosaic, display), `sintel.py` (Python 3 `.cam`/`.dpt` readers) |
| `vfoa/cli.py`, `__main__.py` | `python -m vfoa run ...` |
| `vfoa/backends/` | Lazy string registry, `create_backend`, `create_backend_with_fallback`, base classes, all backends |

Import hygiene: importing any `vfoa` or `thesis_final` module loads only the
standard library, numpy, scipy, cv2 and yaml. Model libraries load inside
`Backend.load()`. A missing library or missing weights raise
`BackendUnavailable`, with a message that names the config key.

### 6.2 Backends and profiles

| Stage | `fast` (default) | `accurate` / `thesis` (thesis models) | `synthetic` |
|---|---|---|---|
| depth | `constant` (or `sintel_gt`, `stereo_bm`) | `stereo_bm`: real disparity, optional WLS | `synthetic` |
| persons | `yolo_pose`: YOLO11n-pose | `openpose_densepose` | `synthetic` |
| head pose | `keypoints`: PnP on face keypoints | `fsanet`, with corrected preprocessing | `synthetic` |
| gaze | `l2cs` (MobileGaze ONNX), falls back to `head_proxy` | `gaze360` with a real temporal window | `synthetic` |
| segmentation | `yolo_seg`: YOLO11n-seg, every 10th frame | `panoptic_fpn`, every frame | `synthetic`, every 5th frame |

- **`accurate`** uses the thesis models with the improved algorithm: angular attention, confidence fusion, depth occlusion, ByteTrack and smoothing.
- **`thesis`** uses the same models with the thesis-faithful, bug-fixed algorithm: greedy IoU tracking, a static Gaze360 window, radius outliers, `wall` ignored, ray-cone attention with the 45-degree agreement fusion, no occlusion and no smoothing.
- **`synthetic`** is a deterministic scripted scene with fake backends. It needs no models and has known ground truth, and it is used by the tests and the benchmark.
- **YAML files.** `configs/{fast,accurate,thesis,synthetic}.yaml` mirror the profiles. `configs/sintel_gt.yaml` is the fast profile on the repository's Sintel clip with ground-truth depth.
- **Third-party code.** The accurate backends find the third-party checkouts (detectron2 with DensePose, Pose_Estimation, gaze360, headpose-fsanet-pytorch) through `models.third_party_root` or `VFOA_THIRD_PARTY`.
- **Weights.** Weights resolve through `models.weights_dir` or `VFOA_WEIGHTS_DIR`.

### 6.3 Streaming loop and multi-rate scheduling

- **Per-frame loop.** `Pipeline.process(frame)` runs on every frame.
  - It ingests the newest frame: persons, then tracking, then `gaze.observe()` (buffering).
  - It then finalises the frame: depth, head pose, gaze, segmentation, scene, smoothing, attention and the output.
- **Multi-rate.** Stage S runs on frame i when `(i - phase) % stride == 0`, or when it has no result yet; otherwise its last result is held. Held head and gaze estimates expire after `max_hold_frames`. Tracking runs whenever persons runs. Scene, smoothing and attention run on every frame.
- **Lookahead.** `runtime.lookahead_frames = L` delays finalisation by L frames, so a temporal model can see future frames (a centred Gaze360 window). L = 0 is the real-time default.
- **Real-time mode.** `--realtime` wraps the source in `LatestFrameSource`, which drops frames the pipeline cannot keep up with. `segmentation.async=true` runs segmentation on a worker thread. Each model is loaded once, at `open()`.
- **Cache.** With `cache.enabled`, every model stage reads its result from the cache, or computes it and stores it. A rerun with `cache.mode=read` can re-score attention with new settings without loading any model.
- **Timing.** Each stage is timed. `timing.json` holds the per-stage statistics, model load times, counters (`dropped_frames`, `gaze_fallback`), capture-to-result latency and the async worker statistics.
- **Outputs.** The run writes these files to `output.dir`:
  - `results.jsonl`: one object per frame, with persons, objects, scores, targets and timings;
  - `results.csv`, optional;
  - `overlay.mp4`, optional;
  - `config.yaml`, the resolved configuration;
  - `timing.json`.

### 6.4 Legacy entry points

`thesis_final/` keeps its public names as thin wrappers over `vfoa`, so there is one implementation:

- `FinalTest.py` keeps the legacy flags, adds `--config`, `--profile`, `--set` and `--max-frames`, and defaults to the `thesis` profile.
- `VideoProcessor.py` and `Reconstruction_scene.py` run on the streaming pipeline.
- `Attention.py`, `BoxTracker.py`, `KalmanFilter.py`, `Depth_Estimator.py` and the predictor modules wrap the new code.
- `_bootstrap.py` adds the repository root to `sys.path`.

## 7. Orchestration

### 7.1 How the main workflow worked (run `wf_0d7c4b6c-dc3`)

| Role | Model | What it did |
|---|---|---|
| `plan:graph` | Fable 5.2 | Wrote the architecture contract and the 15-task graph with dependencies, owned files, acceptance criteria and issue mapping |
| `plan:<task>` | Fable 5.2 | A detailed plan per task, often after prototyping on the machine |
| `impl:<task>` | Opus 5.5 (medium effort) | Implemented the task in its own git worktree on `task/<id>`, with strict file ownership. The whole suite had to pass. |
| `review:<task>#n` | Fable 5.2 | Reviewed the diff: approved, or listed required changes |
| `fix:<task>#n` | Opus 5.5 | Applied the required changes on the same branch |
| `merge:<task>` | Opus 5.5 | Merged with `--no-ff` into `realtime-integration` in the integration worktree and re-ran the suite. Merges were serialised through a merge queue. |
| `verify:round1`, `report` | Fable 5.2 | Final verification and report. Both failed at the Fable spend limit. |

Tasks ran in parallel wherever the dependency graph allowed. Commits reference
issues with "Refs #N", never "Fixes" or "Closes".

Agent counts: 72 agents (38 Fable, 34 Opus), about 9.27 M tokens and 1,293 tool calls.

### 7.2 Tasks

"Review rounds" counts the Fable reviews per task. Two rounds means the first
review requested changes; the log gives the number of changes requested.

| Task | Depends on | Issues | Review rounds | Merge commit | Branch tip |
|---|---|---|---|---|---|
| foundation | - | #30 #35 #17 | 1 | `43eb07f` | `task/foundation` `63d0924` |
| geometry | foundation | #1 #2 #5 #16 #30 | 2 (2 changes) | `06395ef` | `task/geometry` `7f16785` |
| tracking-smoothing | foundation | #23 #12 #18 #17 | 1 | `df7d830` | `task/tracking-smoothing` `e910cd2` |
| depth | foundation | #1 #32 #17 #10 | 2 (3 changes) | `4c5e71c` | `task/depth` `9e9b7ec` |
| io | foundation | #12 #15 #17 #31 #32 | 1 | `b3a2ef2` | `task/io` `6ce0a04` |
| synthetic-cache | foundation | #28 #30 | 1 | `75be654` | `task/synthetic-cache` `a6aa4f2` |
| fast-backends | foundation | #35 | 2 (2 changes) | `05b42fa` | `task/fast-backends` `9d6d646` |
| accurate-detectron | foundation | #9 #32 #13 #18 | 2 (1 change) | `4c2dd30` | `task/accurate-detectron` `f617b19` |
| attention | geometry | #3 #4 #5 #6 #14 #19 #20 #21 #30 | 2 (3 changes) | `a3d2014` | `task/attention` `a8d8449` |
| scene | geometry | #6 #14 #16 #32 #11 #1 | 1 | `c76d858` | `task/scene` `d3219c0` |
| accurate-gaze-headpose | geometry | #2 #7 #8 #9 #17 #32 | 1 | `f2745fc` | `task/accurate-gaze-headpose` `c849fda` |
| pipeline | attention, scene, tracking-smoothing, io, synthetic-cache, depth | #35 #28 #23 #12 #14 #32 | 2 (2 changes) | `1524c16` | `task/pipeline` `2a849f6` |
| realtime | pipeline | #35 | 1 | `c405ba3` | `task/realtime` `1041191` |
| legacy-entry | pipeline, accurate-detectron, accurate-gaze-headpose | #12 #15 #17 #32 #31 | 1 | `6cfbad2` | `task/legacy-entry` `558ce09` |
| benchmark-docs | pipeline, realtime, fast-backends, legacy-entry | #35 #32 #30 | - (planning failed) | not merged | none |
| l2cs-gaze (separate run) | realtime-integration | #35 | 1 (Opus) | `909decc` | `task/l2cs-gaze` `0c4b6bf` |

Six of the 14 merged tasks needed a fix round before approval: geometry,
depth, fast-backends, accurate-detectron, attention and pipeline. The changes
requested were:

- **geometry:** `gaze360_to_camera` silently ignored a batched eye ray.
- **depth:** the tests depended on whether `cv2.ximgproc` was installed.
- **fast-backends:** keypoints at exactly (0, 0) had to be treated as missing.
- **accurate-detectron:** an unavailability test was not hermetic.
- **attention:** depth occlusion excluded the whole body box.
- **pipeline:** interrupted runs demoted caches that were already complete.

`benchmark-docs` was meant to add a CPU benchmark harness, reconcile the config
defaults, write the README and `docs/performance.md`, and add a CI workflow.

### 7.3 Gaze research run (`wf_4de5793e-5e0`)

- **Agents:** Opus 5.5 (medium) orchestrated Sonnet 5.5 (medium) web researchers: 14 agents in total (3 Opus, 11 Sonnet), about 0.96 M tokens.
- **Method:** the researchers shortlisted five candidates and checked their repositories, papers, licences and release assets. CPU timing was measured with ONNX Runtime; GPU numbers come from third parties.
- **Recommendation:** an L2CS-style binned yaw/pitch backend with MobileGaze weights. The code is MIT; the Gaze360-trained weights are research-only.
- **Runner-up:** GaT / ST-WSGE. It is CC-BY-NC and handles the back of the head.
- **Not chosen as default:** UniGaze (needs facial landmarks, non-commercial licence) and Gaze-LLE (target-based, cannot point out of frame).

### 7.4 Gaze backend run (`wf_ddf48012-ef5`)

This run used Opus only, because Fable was unavailable. Four agents (plan,
implementation, review, merge) used about 0.37 M tokens.

- **Backend:** `vfoa/backends/fast/l2cs_gaze.py`, registry key `"l2cs"` (`BinnedGaze360Backend`). Default model: MobileOne-S0, run with onnxruntime.
- **Gate and crop:** a face gate (nose plus at least one eye) and a square face crop at 448 px.
- **Decoding:** the two outputs are read by position. Confidence = 1 minus the normalised bin entropy, times face visibility. Perspective correction exists but is off by default.
- **Fast profile:** `gaze.backend: l2cs` with `gaze.fallback: head_proxy`. The fallback works through `create_backend_with_fallback` and is recorded in the `gaze_fallback` counter.
- **Weights:** downloaded on first use, with a sha256 check for MobileOne-S0.
- **Tests:** 30 new tests in `tests/test_gaze_l2cs.py`. Network downloads are blocked during tests.
- **Result:** merged as `909decc`; 615 passed, 2 skipped.

### 7.5 Failed resume (2026-11-14, 28 agents)

- **What happened:** resuming the main run failed.
  - Fable was still limited, so all 7 reviews, the verification and the report errored.
  - The workflow cache served 12 earlier steps, but it missed for the parallel implementation steps.
  - 7 Opus implementation agents re-ran, and they moved the `task/*` branches to `909decc`.
- **Repair:** the main session restored every `task/*` branch to its original tip (the second parent of each merge commit) and removed the stray worktrees. The integration branch was unaffected.
- **Lesson:** do not resume that run. Use a new, small "tail" workflow for the remaining steps.

## 8. Results

### 8.1 Tests

`python -m pytest -q` on `realtime-integration` (this branch): **615 passed, 2 skipped** (the opt-in network tests), in about 50 s on CPU.

### 8.2 CPU timing (main session, 2026-11-14)

The machine is a shared, CPU-only WSL machine with 8 cores. Numbers are in
ms per frame, from `timing.json`.

**Synthetic profile** (no models; 100 frames at 640x360): **121 fps**, 8.3 ms mean total pipeline overhead.

| Stage | Mean ms | p50 ms | Runs |
|---|---|---|---|
| read | 2.39 | 2.07 | 100 |
| scene | 1.76 | 1.49 | 100 |
| headpose (fake) | 1.23 | 0.99 | 100 |
| attention | 1.18 | 1.11 | 100 |
| smoothing | 0.59 | 0.51 | 100 |
| tracking | 0.35 | 0.32 | 100 |
| segmentation (fake, stride 5) | 1.54 | 1.56 | 20 |
| total | 8.28 | 7.41 | 100 |

**Fast profile on the Sintel clip** (`configs/sintel_gt.yaml`; YOLO11n on CPU; ground-truth depth; 50 frames at 1024x436). This run was measured before the `l2cs` backend was merged, so it has no gaze stage. Result: **2.4 fps**.

| Stage | Mean ms | p50 ms | Runs | Amortised ms/frame |
|---|---|---|---|---|
| persons (YOLO11n-pose) | 314.4 | 186.6 | 50 | 314.4 |
| segmentation (YOLO11n-seg, every 10th frame) | 498.2 | 269.7 | 5 | 49.8 |
| depth (read Sintel `.dpt`) | 37.9 | 39.8 | 50 | 37.9 |
| read | 9.3 | 8.0 | 50 | 9.3 |
| scene | 2.07 | 1.90 | 50 | 2.07 |
| attention | 0.85 | 0.84 | 50 | 0.85 |
| tracking | 0.34 | 0.26 | 50 | 0.34 |
| headpose (keypoints PnP) | 0.31 | 0.29 | 50 | 0.31 |
| smoothing | 0.20 | 0.19 | 50 | 0.20 |
| total | 415.3 | 250.9 | 50 | 415.3 |

- **Non-model stages** (tracking, head pose, scene, smoothing, attention, output) add up to about 3.5-4 ms per frame, so the CPU cost is almost all model inference.
- **Model loading** happens once: persons 4.3 s, segmentation 2.6 s.

### 8.3 Gaze backend

| Run | Input | Frames | Gaze mean / p50 / p95 ms per frame | Total fps |
|---|---|---|---|---|
| l2cs | Sintel clip, ground-truth depth | 30 | 25.2 / 21.0 / 52.8 | 1.0 |
| sign_video | `Final_videos/video.mp4`, constant depth | 40 | 34.2 / 26.2 / 72.4 | 1.1 |
| sign_museum | `Final_videos/museum.mp4`, constant depth | 40 | 21.7 / 20.2 / 67.8 | 0.9 |
| sign_got | `Final_videos/got.mp4`, constant depth | 40 | 23.2 / 19.3 / 51.1 | 1.2 |
| l2cs_fallback | Sintel clip, no weights, no download | 5 | head_proxy, `gaze_fallback = 1` | 3.9 |

- **Gaze cost on CPU:** about 20-35 ms per person at 448 px. The load time is 0.26-1.1 s, including the download.
- **Persons stage in these runs:** 720-890 ms per frame on the loaded machine, so the total fps is about 1. The persons cost is much higher than in the 2.4 fps run.
- **Research measurements:** MobileOne-S0 took about 37-65 ms per call with batch 1.
- **Output:** on the Sintel clip, gaze vectors are unit length and change smoothly for all 29 person-frames.
- **Confidence:** about 0.20-0.26, which is low.
- **Sign check:** gaze was drawn against the head direction on the real clips (`got`, `museum`, `video`), and the two broadly agree. On the Sintel CG clip they differ by 25-60 degrees: gaze x points image-right while the keypoint head direction points left. The sign convention is therefore plausible but not confirmed.

## 9. How to run

All commands run from the repository root. `python` is any Python >= 3.10
environment with the requirements installed.

```bash
pip install -r requirements.txt -r requirements-dev.txt   # core + tests
pip install -r requirements-fast.txt                      # fast profile (ultralytics, onnxruntime)
pip install -r requirements-accurate.txt                  # accurate/thesis profiles (+ source checkouts)
```

**Tests** (models are always faked; network tests are opt-in):

```bash
python -m pytest -q
VFOA_RUN_NETWORK=1 python -m pytest -q -m network   # downloads YOLO / L2CS weights
```

**Synthetic profile** (no models):

```bash
python -m vfoa run --profile synthetic --max-frames 100 --output-dir outputs/synth
```

**Fast profile** on a video (YOLO11 weights and the L2CS ONNX download on first use):

```bash
python -m vfoa run --profile fast --input Final_videos/got.mp4 --output-dir outputs/got --video
```

**Sintel clip with ground-truth depth:**

```bash
python -m vfoa run --config configs/sintel_gt.yaml --output-dir outputs/sintel --max-frames 50
```

**Real-time mode:** drop frames to keep up, and run segmentation on a worker thread:

```bash
python -m vfoa run --profile fast --input 0 --realtime --set segmentation.async=true --show
```

**Useful overrides:**

- Lower the gaze rate on CPU: `--set gaze.stride=3`.
- Turn gaze off: `--set gaze.backend=none`.
- Use the head direction as gaze: `--set gaze.backend=head_proxy`.
- Re-score from the cache without models: run once with `--set cache.enabled=true`, then again with `--set cache.enabled=true --set cache.mode=read` plus the new attention or smoothing settings.

**Accurate (thesis-model) backends on a GPU machine:**

- **Third-party checkouts.** Prepare the thesis checkouts with the thesis layout: a detectron2 checkout that also holds `projects/DensePose`, `Pose_Estimation`, `gaze360` and `headpose-fsanet-pytorch`, with their weights at the default relative paths in `vfoa/config.py`.
- **Environment.** Set `VFOA_THIRD_PARTY=/path/to/detectron2`, or `models.third_party_root`. Set `VFOA_WEIGHTS_DIR` for the downloadable weights.
- **WLS filter.** Install `opencv-contrib-python-headless` in place of `opencv-python-headless` to get the WLS disparity filter.

```bash
export VFOA_THIRD_PARTY=/path/to/detectron2
python -m vfoa run --config configs/accurate.yaml --input LEFT.mp4 --right RIGHT.mp4 \
    --camera-dir CAMS/ --output-dir outputs/accurate
python -m vfoa run --config configs/thesis.yaml ...               # thesis-faithful algorithm
python -m vfoa run --config configs/accurate.yaml --set runtime.lookahead_frames=9 ...   # centred Gaze360 window (>= 3 * window_step; 9 at 30 fps)
python thesis_final/FinalTest.py -d VIDEO.mp4 -od outputs/legacy  # legacy entry point (thesis profile)
```

`models.device: auto` selects CUDA when it is available.

**L2CS weights and licence**

- **Download.** With `models.allow_download` and `gaze.l2cs.download` set (the defaults), the fast profile downloads `mobileone_s0_gaze.onnx` from the MobileGaze GitHub release (`yakhyo/gaze-estimation`, tag `weights`) into `models.weights_dir`, or into `VFOA_WEIGHTS_DIR`, and checks its sha256.
- **Other models.** Choose another model with `gaze.l2cs.model` (`resnet18`, `resnet34`, `resnet50`, `mobilenetv2`; no pinned hash), or give an explicit file with `gaze.l2cs.path`.
- **L2CS-Net ResNet-50.** It needs a user-made ONNX export, for example from the py-feat safetensors.
- **Licence.** The code is MIT. The weights are trained on Gaze360 and are **research / non-commercial only**. Never commit them.

## 10. Known issues and open questions

Collected from the task `open_problems`, the reviews and the gaze report.

**Gaze**

1. The L2CS axis signs come from the code only. On Sintel, gaze and head direction differ by 25-60 degrees. This needs checking on footage with ground truth, or on clips of a person looking left, right, up and down.
2. L2CS entropy confidence is low (0.2-0.26), so confidence-weighted fusion gives gaze little weight. It needs rescaling or calibration.
3. The released MobileGaze ONNX files have a static batch of 1, so they need one call per person. Batching needs a re-export with `onnx_export.py --dynamic`.
4. CPU gaze cost (about 20-35 ms per person at 448 px) is over a real-time budget. Options: `gaze.stride`, 224 px input (accuracy unknown), or a GPU.
5. Only MobileOne-S0 has a pinned sha256. L2CS-Net ResNet-50 has no built-in download.
6. Perspective correction is implemented but not validated, so it stays off.
7. Back-of-head: L2CS/MobileGaze does not cover it. Those persons fall back to head pose; GaT would be the option.
8. After a fallback, the `config.yaml` written by the CLI still says `l2cs`; the switch shows only in the log and in the `gaze_fallback` counter. The gaze cache key does not hash the weights file contents. Adding `gaze.l2cs`/`gaze.fallback` to the defaults changed every gaze cache key once.

**Config gaps.** Several optional keys are read with `.get` defaults but are
not in `DEFAULTS` or the YAML files:

- `tracking.motion` and `tracking.max_predict_frames`;
- `smoothing.jump_deg`;
- `depth.speckle_window` and `depth.speckle_range`;
- `headpose.relaxed_kpt_thr`, `headpose.fallback_confidence` and `headpose.max_reproj_frac`;
- `persons.upper_body_only`, `persons.match_min_inside` and `persons.keep_headless`;
- `scene.outlier_nb_points`, `scene.outlier_radius_m`, `scene.outlier_radius_scale` and `scene.self_aabb_margin_m`;
- `synthetic.sway_m`;
- `input.pace`.

This reconciliation was part of `benchmark-docs`.

**Pipeline and real time**

1. `output` timing shows N-1 runs, because the last frame's output time is added after its record is closed.
2. `io/sources.py` ignores `input.start` and `input.max_frames` for the synthetic source. The pipeline works around it.
3. In async mode the segmentation worker's own time is not in the stage table; it is reported under `async`.
4. A gaze backend with a fallback is loaded at `open()` even when a complete cache could serve it.
5. The README and docs do not mention `--realtime`, `segmentation.async` or the new `timing.json` keys.

**Backends and geometry**

1. In the fast profile, a person seen from behind gets no head estimate, because PnP needs the nose plus 3 more face keypoints. With `head_proxy`, that person has no gaze either.
2. Reviewer suggestions not done:
   - track on the body box in the fast profile;
   - raise the requirement to `ultralytics>=8.3.160`;
   - make the `yolo_seg` `np.unique` faster;
   - clip the `HeadProxyGaze` confidence.
3. Accurate backends: these have not been run with the real models, because detectron2, DensePose, OpenPose, Gaze360 and FSA-Net are not on this machine.
   - Nits left: a success-path test for the OpenPose runtime, a NaN guard in skeleton-to-head-box matching, and a faster panoptic conversion.
   - Gaze360 confidence comes from its uncertainty; FSA-Net always returns 1.0.
4. Attention: when the person's own segment is unknown, an occluder inside the body box is ignored. With the default rings, the thesis `rays` method misses the small synthetic objects.
5. Legacy: the OpenGL/lucid `renderArrow` path could not be run here. `thesis_final/Attention.py` still has its own copy of `vect2cone`.
6. Depth: consider `wls: false` for the fast profile. A duplicated `frame_intrinsics` helper remains.

**Process**

- The statuses in section 4 come from the task reports. The Fable verification step never ran.

## 11. Remaining work and next steps

1. **Fable tail run after 2026-11-17 13:00 Europe/London.** Start a new, small workflow; do not resume `wf_0d7c4b6c-dc3`. It should:
   - plan and implement `benchmark-docs`: the benchmark harness, config reconciliation, the README and `docs/performance.md`, and the CI workflow;
   - run the verification of `realtime-integration` against the issues;
   - write the final report.
2. **GPU latency.** Measure the fast and accurate profiles on a GPU machine, then try ONNX/TensorRT FP16 (YOLO11, L2CS) and batched gaze.
3. **Accurate backend validation.** Run `accurate` and `thesis` with the original third-party repos and weights, and compare them with the original thesis outputs.
4. **Gaze signs.** Confirm the L2CS sign convention and the perspective correction on footage with ground truth. Calibrate the confidence.
5. **Push or PR decision.** Decide whether to push `realtime-integration` (and this log) and open a PR that references #37. Nothing is pushed yet. GitHub issues should be closed only after verification.
6. **Out-of-scope issues.** Still open:
   - evaluation: #25, #26, #27;
   - tighter object geometry: #22;
   - newer models per stage: #24;
   - pinned environment: #29;
   - dead code: #33;
   - repository hygiene: #34;
   - README: #36.

## 12. Licences to note

| Component | Licence | Consequence |
|---|---|---|
| ultralytics (YOLO11 pose/seg, fast profile) | AGPL-3.0 | Optional dependency. It is imported lazily, only in `vfoa/backends/fast/_ultra.py`, and the rest of the package works without it. |
| MobileGaze / L2CS-Net code | MIT | Fine to use. |
| Gaze360-trained weights (MobileGaze, L2CS, GaT, UniGaze-joint) | Gaze360 terms: research / non-commercial only | Optional download; never commit or redistribute the weights. State this in the README. |
| GaT / ST-WSGE (runner-up gaze model) | CC-BY-NC-4.0 (code and weights) | Research-only. Keep it out of the `vfoa` source tree. |
| UniGaze | ModelGo NC-RAI (non-commercial) | Optional "accurate frontal" option only. |
