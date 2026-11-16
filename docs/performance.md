# Performance (CPU measurements)

Measured with `benchmarks/bench_pipeline.py` on 2026-11-16 (Refs #35, #32).

## Machine and caveats

- 8-core x86-64 CPU, **no GPU**, WSL2 (Linux 6.6), Python 3.12.3, NumPy 2.5.5,
  OpenCV 5.0.0, torch 2.15.0+cpu, ultralytics 8.4.215, onnxruntime 1.32.0.
- **The machine was shared and noisy.** The 1-minute load average was 1.2-4.8 during
  the synthetic runs and 3-10 during the model runs (recorded in each JSON result as
  `meta.loadavg_start` / `meta.loadavg_end`); repeated runs differ by 10-20%.
- CPU thread settings change the model numbers by more than 20x on this machine (see
  "CPU threading" below). Every JSON result records them: `meta.torch_threads` and
  `meta.env` (`OMP_NUM_THREADS`, `OMP_WAIT_POLICY`, `MKL_NUM_THREADS` as set when the
  benchmark started). The fast-profile runs below were made with none of these
  variables set, unless stated.
- **GPU numbers are to be measured by the user**, with exactly the same commands on the
  GPU machine (`models.device: auto` selects CUDA when available).

How to read the tables: "runs" is the number of measured frames on which the stage
ran (strided stages run less often), "mean/p50/p95/max ms" are over those runs,
"ms/frame" is the cost amortised over all measured frames, and "share" is the
fraction of the total. FPS = 1000 / mean total. The total is the sum of the stages
of a frame (frame read included, result writing excluded). Warm-up frames are
excluded; model load times are reported separately as `setup`.

## Synthetic profile: pure pipeline overhead

No models: the synthetic backends serve a scripted scene (2 persons, 4 objects,
segmentation every 5th frame). This measures the frame loop, tracking, scene
building, smoothing and attention.

```bash
python benchmarks/bench_pipeline.py --profile synthetic --frames 60 --warmup 5
```

640x360, 60 frames after 5 warm-up. **111.8 FPS** (load average 4.2).

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 60 | 2.57 | 2.31 | 3.50 | 5.71 | 2.57 | 28.8% |
| depth | 60 | 0.10 | 0.09 | 0.22 | 0.33 | 0.10 | 1.2% |
| persons | 60 | 0.07 | 0.06 | 0.14 | 0.17 | 0.07 | 0.8% |
| tracking | 60 | 0.43 | 0.32 | 1.02 | 1.99 | 0.43 | 4.8% |
| headpose | 60 | 1.33 | 1.17 | 2.29 | 2.84 | 1.33 | 14.9% |
| gaze | 60 | 0.09 | 0.06 | 0.28 | 0.29 | 0.09 | 1.0% |
| segmentation | 12 | 1.97 | 1.85 | 2.75 | 2.97 | 0.39 | 4.4% |
| scene | 60 | 1.95 | 1.66 | 4.21 | 5.58 | 1.95 | 21.8% |
| smoothing | 60 | 0.66 | 0.54 | 1.17 | 1.79 | 0.66 | 7.4% |
| attention | 60 | 1.34 | 1.19 | 2.03 | 2.98 | 1.34 | 15.0% |
| gaze:observe | 60 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 60 | 8.95 | 8.31 | 13.47 | 13.84 | 8.95 | 100.0% |

The same command on a quieter moment (load average 1.2) gave 130.4 FPS (7.67 ms mean
total). The budget of the architecture (>= 30 FPS at 640x360 without models) is met.

Real video decode + synthetic backends (the synthetic world takes the video size):

```bash
python benchmarks/bench_pipeline.py --profile synthetic --input Final_videos/museum.mp4 --frames 30
```

1280x720, 30 frames after 10 warm-up. **47.5 FPS**; total mean 21.07 ms, of which
read 3.12 ms, scene 4.75 ms, headpose (synthetic) 4.76 ms, attention 3.18 ms,
segmentation 11.44 ms on 6 of 30 frames (2.29 ms amortised).

## Stereo depth (`stereo_bm`) with a synthesised right view

There is no stereo footage in the repository, so `--stereo-synth` renders the right
view of each frame from the Sintel ground-truth depth before the timed run.

```bash
python benchmarks/bench_pipeline.py --profile synthetic --input Final_videos/mpi_sintel.mp4 \
    --camera-dir Final_videos/stereo/camdata_left/bandage_2 \
    --depth-dir Final_videos/stereo/training/bandage_2_depth \
    --stereo-synth --set depth.backend=stereo_bm --set depth.scale=0.5 --frames 20
```

1024x436, 20 frames after 10 warm-up, load average 12.5. **44.9 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 20 | 7.65 | 7.52 | 9.77 | 10.65 | 7.65 | 34.4% |
| depth | 20 | 5.15 | 4.87 | 5.86 | 6.15 | 5.15 | 23.1% |
| persons | 20 | 0.09 | 0.08 | 0.11 | 0.12 | 0.09 | 0.4% |
| tracking | 20 | 0.47 | 0.44 | 0.64 | 0.69 | 0.47 | 2.1% |
| headpose | 20 | 2.89 | 2.46 | 4.37 | 6.48 | 2.89 | 13.0% |
| gaze | 20 | 0.14 | 0.07 | 0.52 | 0.87 | 0.14 | 0.6% |
| segmentation | 4 | 4.44 | 4.44 | 5.23 | 5.31 | 0.89 | 4.0% |
| scene | 20 | 2.51 | 2.15 | 4.00 | 5.33 | 2.51 | 11.3% |
| smoothing | 20 | 0.70 | 0.55 | 1.24 | 1.54 | 0.70 | 3.1% |
| attention | 20 | 1.79 | 1.46 | 3.30 | 3.53 | 1.79 | 8.0% |
| gaze:observe | 20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 20 | 22.26 | 20.57 | 27.96 | 30.11 | 22.26 | 100.0% |

StereoBM at half resolution (`depth.scale=0.5`, no WLS: `cv2.ximgproc` is not
installed) costs about 5 ms per 1024x436 frame.

## Fast profile on real video

YOLO11n-pose persons, keypoint head pose, L2CS gaze (MobileOne-S0 ONNX, every 2nd
frame), YOLO11n-seg every 10th frame. Weights were present locally in `weights/`, so
no gaze fallback happened (`meta.fallbacks` is empty). torch ran with 4 intra-op
threads (`meta.torch_threads`, the CPU default described under "CPU threading").

```bash
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/museum.mp4 --frames 30 --warmup 5
```

1280x720, 30 frames after 5 warm-up, load average 4.1 -> 4.3. **18.50 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 30 | 1.15 | 0.86 | 2.33 | 2.83 | 1.15 | 2.1% |
| depth | 30 | 0.33 | 0.26 | 0.59 | 0.95 | 0.33 | 0.6% |
| persons | 30 | 36.41 | 27.32 | 61.82 | 109.00 | 36.41 | 67.4% |
| tracking | 30 | 0.44 | 0.41 | 0.67 | 0.75 | 0.44 | 0.8% |
| headpose | 30 | 0.28 | 0.23 | 0.56 | 0.89 | 0.28 | 0.5% |
| gaze | 15 | 10.51 | 11.23 | 21.75 | 22.43 | 5.25 | 9.7% |
| segmentation | 3 | 52.57 | 50.76 | 63.82 | 65.27 | 5.26 | 9.7% |
| scene | 30 | 3.35 | 3.32 | 4.61 | 4.90 | 3.35 | 6.2% |
| smoothing | 30 | 0.25 | 0.25 | 0.36 | 0.57 | 0.25 | 0.5% |
| attention | 30 | 1.33 | 1.32 | 1.61 | 3.30 | 1.33 | 2.5% |
| gaze:observe | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 30 | 54.05 | 43.50 | 116.22 | 132.68 | 54.05 | 100.0% |

setup: load:persons 7618 ms, load:segmentation 3002 ms, load:gaze 217 ms.
Capture-to-result latency: mean 55.6 ms, p95 125.1 ms
(over all 35 frames, warm-up included).

Sintel clip with ground-truth depth:

```bash
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/mpi_sintel.mp4 \
    --camera-dir Final_videos/stereo/camdata_left/bandage_2 \
    --depth-dir Final_videos/stereo/training/bandage_2_depth --set depth.backend=sintel_gt \
    --frames 30 --warmup 5
```

1024x436, 30 frames after 5 warm-up, load average 7.0 -> 7.8. **16.54 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 30 | 5.77 | 5.27 | 8.40 | 11.41 | 5.77 | 9.5% |
| depth | 30 | 14.79 | 13.33 | 23.44 | 31.88 | 14.79 | 24.5% |
| persons | 30 | 26.76 | 22.51 | 41.89 | 43.80 | 26.76 | 44.3% |
| tracking | 30 | 0.29 | 0.26 | 0.44 | 0.60 | 0.29 | 0.5% |
| headpose | 30 | 0.28 | 0.27 | 0.38 | 0.52 | 0.28 | 0.5% |
| gaze | 15 | 11.96 | 11.43 | 14.61 | 14.94 | 5.98 | 9.9% |
| segmentation | 3 | 33.66 | 33.39 | 36.98 | 37.38 | 3.37 | 5.6% |
| scene | 30 | 2.12 | 2.01 | 2.95 | 3.56 | 2.12 | 3.5% |
| smoothing | 30 | 0.26 | 0.26 | 0.40 | 0.52 | 0.26 | 0.4% |
| attention | 30 | 0.84 | 0.86 | 1.05 | 1.43 | 0.84 | 1.4% |
| gaze:observe | 30 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.0% |
| total | 30 | 60.45 | 57.46 | 86.71 | 91.24 | 60.45 | 100.0% |

The first run of this command, with a cold file cache, gave **3.15 FPS**: depth
147.5 ms and read 34.7 ms per frame, because the `.dpt` depth files
(about 1.8 MB each) and the video are read from the Windows drive (`/mnt/c`) through
WSL2. The table above is the repeated run (warm cache). Depth and read are file I/O
here, not computation.

### CPU threading

**Cause.** ultralytics' `select_device()` calls
`torch.set_num_threads(min(8, cpu_count - 1))` at the first predict, i.e. 7 torch
threads on this 8-vCPU WSL2 VM. OpenMP's default policy is to spin-wait between
parallel regions, and the l2cs ONNX Runtime session spins its own intra-op threads
too. On a shared VM these busy-waiting pools starve each other: each YOLO11n call
took 2-3 s instead of tens of milliseconds.

**What vfoa does by default.**

- `vfoa/backends/fast/_ultra.py::load_yolo` sets torch's intra-op threads after the
  warm-up predict: `runtime.num_threads` when it is set; otherwise, when the model runs
  on CPU, `min(current, cpu_count // 2)` (4 here); on a GPU device it leaves torch
  alone. The chosen value is logged at INFO.
- `vfoa/backends/fast/l2cs_gaze.py` creates its ONNX Runtime session with
  `session.intra_op.allow_spinning = 0`.

**Knobs.**

- `runtime.num_threads` (default null): an integer sets torch's threads for the YOLO
  models and `intra_op_num_threads` for the ONNX Runtime sessions.
- `export OMP_WAIT_POLICY=PASSIVE` before starting Python (it must be set before torch
  is imported, so setting it from vfoa code is too late). It mainly helps when
  `segmentation.async` overlaps the persons stage; with the default strides it was
  slower here (OpenMP threads sleep and wake up between layers).

**Before / after** (fast profile, museum.mp4, 30 frames after 5 warm-up):

| variant | torch threads | persons mean ms | persons p95 ms | FPS |
|---|---:|---:|---:|---:|
| before (commit 4cedb24: 7 torch threads, ORT spinning) | 7 | 2888.89 | 3852.02 | 0.30 |
| ORT spinning off, `--set runtime.num_threads=7` (7 torch and 7 ORT threads) | 7 | 2109.49 | 2474.74 | 0.42 |
| after: default (4 torch threads, ORT spinning off) | 4 | 36.41 | 61.82 | 18.50 |
| after + `OMP_WAIT_POLICY=PASSIVE` | 4 | 58.82 | 104.39 | 11.99 |
| CPU preset, after | 4 | 78.43 | 283.43 | 20.89 |
| CPU preset, after + `OMP_WAIT_POLICY=PASSIVE` | 4 | 77.37 | 134.39 | 18.05 |

The "before" row is the measurement made with commit 4cedb24 on the same day. The
second row shows that the torch thread count alone is enough to fall back to under
1 FPS. The PASSIVE runs used `OMP_WAIT_POLICY=PASSIVE python benchmarks/bench_pipeline.py ...`
with otherwise identical commands.

### Gaze stride

`--set gaze.stride=1` (L2CS on every frame), same museum command, load average 4.3 -> 5.0: gaze ran
on 30 of 30 frames, mean 9.86 ms, **9.86 ms per frame** amortised and
16.74 FPS, versus 5.25 ms per frame and 18.50 FPS with the default
`gaze.stride: 2` (15 runs of 10.51 ms). L2CS is the only per-person model cost
(the released ONNX files are batch 1, one call per person), so its cost grows with the
number of people. Held estimates are smoothed by the Kalman
filter, so a stride of 2 costs little accuracy at 25-30 fps. Persons and
segmentation strides are unchanged in the profile (1 and 10).

### CPU preset

```bash
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/museum.mp4 --frames 30 --warmup 5 \
    --set persons.stride=2 --set gaze.stride=4 --set segmentation.stride=30 --set segmentation.async=true
```

1280x720, 30 frames after 5 warm-up, load average 5.0 -> 5.2. **20.89 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 30 | 1.24 | 1.09 | 2.12 | 3.95 | 1.24 | 2.6% |
| depth | 30 | 0.29 | 0.29 | 0.35 | 0.36 | 0.29 | 0.6% |
| persons | 15 | 78.43 | 27.25 | 283.43 | 645.40 | 39.21 | 81.9% |
| tracking | 15 | 0.38 | 0.37 | 0.44 | 0.46 | 0.19 | 0.4% |
| headpose | 30 | 0.25 | 0.24 | 0.41 | 0.45 | 0.25 | 0.5% |
| gaze | 7 | 8.31 | 11.17 | 19.55 | 22.42 | 1.94 | 4.0% |
| segmentation | 1 | 0.04 | 0.04 | 0.04 | 0.04 | 0.00 | 0.0% |
| scene | 30 | 3.21 | 3.12 | 4.16 | 5.53 | 3.21 | 6.7% |
| smoothing | 30 | 0.21 | 0.22 | 0.36 | 0.42 | 0.21 | 0.4% |
| attention | 30 | 1.32 | 1.29 | 2.22 | 2.56 | 1.32 | 2.8% |
| gaze:observe | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 30 | 47.86 | 18.64 | 110.69 | 655.36 | 47.86 | 100.0% |

With `segmentation.async` the segmentation stage in the table is only the hand-off
to the worker; the worker itself ran 2 times with a mean of
586.6 ms off the frame loop. While it runs, the persons calls on the main thread
slow down (p95 283 ms, max 645 ms). With `OMP_WAIT_POLICY=PASSIVE` exported
the same command gave **18.05 FPS** with persons p95 134 ms, max 159 ms and
a worker mean of 99.4 ms; capture-to-result latency p95 156 ms
versus 275 ms.

On this machine the default strides already reach about 18-19 FPS on museum.mp4, so
the preset brings little extra average throughput; it is meant for slower CPUs and
scenes with more people, and is best combined with `OMP_WAIT_POLICY=PASSIVE`.

### Observations

- **Persons dominates.** With the default strides YOLO11n-pose is about 65-70% of the
  frame time on museum.mp4; all non-model stages together (tracking, head pose,
  scene, smoothing, attention) cost about 5-6 ms per frame.
- **Threading matters more than the model.** See "CPU threading": without the cap on
  torch threads and with ONNX Runtime spinning, the same profile ran at 0.3 FPS.

### Accurate profile

```bash
python benchmarks/bench_pipeline.py --profile accurate --input Final_videos/mpi_sintel.mp4 \
    --camera-dir Final_videos/stereo/camdata_left/bandage_2 --frames 10 --warmup 2
```

Recorded result: `SKIPPED: detectron2 is not importable (No module named
'detectron2'); install detectron2 or point the config key models.third_party_root
(or the environment variable VFOA_THIRD_PARTY) to the thesis third-party checkout
holding detectron2` (status skipped, exit 0). The thesis models are not available on
this machine.

## Micro-benchmarks of the #32 fixes

```bash
python benchmarks/bench_pipeline.py --frames 0 --micro
```

Median of 5 repeats, load average 12.2.

| micro-benchmark | legacy | new | legacy ms | new ms | speed-up | results agree |
|---|---|---|---:|---:|---:|---|
| ply_roundtrip | ASCII PLY write (`np.savetxt`) + `open3d.io.read_point_cloud` + AABB, 20,000 points | `geometry.aabb_from_points` | 44.09 | 1.525 | 28.9x | yes (1e-4) |
| slab_test | pure-Python `Attention.intersection` loop, 400 rays x 20 boxes | vectorised `geometry.ray_aabb` | 21.72 | 1.036 | 21.0x | yes (same hit count) |
| stereo_bm | StereoBM computed twice at full resolution (1024x436, 160 disparities) | one StereoBM at scale 0.5 (80 disparities) | 20.65 | 2.924 | 7.1x | yes (median disparity within 10%) |

The stereo pair is Sintel bandage_2 frame 1 with a right view synthesised from its
ground-truth depth. Other #32 items are structural and not micro-benchmarked: each
model is loaded once per run (`setup` above shows the one-off load times) and video
is streamed frame by frame instead of being read into RAM.

## GPU numbers

Not measured here. Run the same commands on the GPU machine (they write
`benchmarks/results/<profile>-<input>.json`; pass `--markdown` to get tables like the
ones above) and add the results to this file.
