# Performance (CPU measurements)

Measured with `benchmarks/bench_pipeline.py` on 2026-11-16 (Refs #35, #32).

## Machine and caveats

- 8-core x86-64 CPU, **no GPU**, WSL2 (Linux 6.6), Python 3.12.3, NumPy 2.5.5,
  OpenCV 5.0.0, torch 2.15.0+cpu, ultralytics 8.4.215, onnxruntime 1.32.0.
- **The machine was shared and noisy.** The 1-minute load average was 1.2-4.8 during
  the synthetic runs and 4-13.5 during the model runs (recorded in each JSON result as
  `meta.loadavg_start` / `meta.loadavg_end`). Treat the model numbers as an upper bound.
- Model cost on this machine also varies from day to day: YOLO11n-pose took
  0.2-0.3 s per frame on 2026-11-14 and 2.8-3.2 s per frame on 2026-11-16 in
  the same configuration (see "Observations" below).
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
no gaze fallback happened (`meta.fallbacks` is empty).

```bash
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/museum.mp4 --frames 30 --warmup 5
```

1280x720, 30 frames after 5 warm-up, load average 4.0 -> 9.8. **0.30 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 30 | 3.91 | 1.93 | 11.55 | 16.69 | 3.91 | 0.1% |
| depth | 30 | 0.67 | 0.31 | 3.02 | 4.15 | 0.67 | 0.0% |
| persons | 30 | 2888.89 | 2801.82 | 3852.02 | 4367.03 | 2888.89 | 87.9% |
| tracking | 30 | 0.54 | 0.49 | 0.94 | 1.20 | 0.54 | 0.0% |
| headpose | 30 | 0.36 | 0.27 | 0.99 | 1.45 | 0.36 | 0.0% |
| gaze | 15 | 51.10 | 51.24 | 129.17 | 149.79 | 25.55 | 0.8% |
| segmentation | 3 | 3550.23 | 3189.05 | 4188.62 | 4299.68 | 355.02 | 10.8% |
| scene | 30 | 6.43 | 4.26 | 17.57 | 39.49 | 6.43 | 0.2% |
| smoothing | 30 | 0.83 | 0.32 | 0.83 | 15.34 | 0.83 | 0.0% |
| attention | 30 | 2.77 | 1.63 | 8.97 | 10.43 | 2.77 | 0.1% |
| gaze:observe | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 30 | 3284.97 | 3010.44 | 5916.32 | 6786.01 | 3284.97 | 100.0% |

setup: load:persons 8949 ms, load:segmentation 3894 ms, load:gaze 313 ms.

Sintel clip with ground-truth depth:

```bash
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/mpi_sintel.mp4 \
    --camera-dir Final_videos/stereo/camdata_left/bandage_2 \
    --depth-dir Final_videos/stereo/training/bandage_2_depth --set depth.backend=sintel_gt \
    --frames 30 --warmup 5
```

1024x436, 30 frames after 5 warm-up, load average 9.8 -> 13.5. **0.30 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 30 | 24.01 | 7.91 | 72.68 | 134.76 | 24.01 | 0.7% |
| depth | 30 | 21.72 | 19.48 | 36.25 | 37.56 | 21.72 | 0.7% |
| persons | 30 | 2884.99 | 2855.08 | 3751.29 | 5144.81 | 2884.99 | 87.7% |
| tracking | 30 | 0.35 | 0.27 | 0.50 | 2.16 | 0.35 | 0.0% |
| headpose | 30 | 0.32 | 0.31 | 0.46 | 0.68 | 0.32 | 0.0% |
| gaze | 15 | 58.90 | 51.75 | 115.41 | 130.49 | 29.45 | 0.9% |
| segmentation | 3 | 3219.92 | 3188.69 | 3452.29 | 3481.58 | 321.99 | 9.8% |
| scene | 30 | 5.46 | 2.87 | 18.33 | 30.41 | 5.46 | 0.2% |
| smoothing | 30 | 0.50 | 0.29 | 1.52 | 4.39 | 0.50 | 0.0% |
| attention | 30 | 1.48 | 0.90 | 2.95 | 13.21 | 1.48 | 0.0% |
| gaze:observe | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 30 | 3290.29 | 3078.58 | 5808.47 | 5943.38 | 3290.29 | 100.0% |

setup: load:persons 7338 ms, load:segmentation 3646 ms, load:gaze 275 ms.

### Gaze stride

`--set gaze.stride=1` (L2CS on every frame), same museum command, load average 11.9:
gaze ran on 30 of 30 frames, mean 42.57 ms, **42.57 ms per frame** amortised, versus
25.55 ms per frame with the default `gaze.stride: 2` (15 runs of 51.10 ms). Total
0.29 FPS: on this machine the gaze saving is invisible next to the persons stage, but
L2CS is the only per-person model cost (the released ONNX files are batch 1, one call
per person), so its cost grows with the number of people. Held estimates are smoothed
by the Kalman filter, so a stride of 2 costs little accuracy at 25-30 fps. Persons and
segmentation strides are unchanged in the profile (1 and 10).

### CPU preset

```bash
python benchmarks/bench_pipeline.py --profile fast --input Final_videos/museum.mp4 --frames 30 --warmup 5 \
    --set persons.stride=2 --set gaze.stride=4 --set segmentation.stride=30 --set segmentation.async=true
```

1280x720, 30 frames after 5 warm-up, load average 11.4 -> 12.2. **4.75 FPS**.

| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| read | 30 | 14.56 | 5.39 | 42.55 | 115.15 | 14.56 | 6.9% |
| depth | 30 | 0.41 | 0.33 | 0.93 | 1.33 | 0.41 | 0.2% |
| persons | 15 | 316.00 | 344.58 | 520.30 | 544.10 | 158.00 | 75.0% |
| tracking | 15 | 0.54 | 0.48 | 0.81 | 1.13 | 0.27 | 0.1% |
| headpose | 30 | 0.75 | 0.29 | 3.76 | 7.04 | 0.75 | 0.4% |
| gaze | 7 | 71.12 | 105.52 | 141.45 | 146.02 | 16.59 | 7.9% |
| segmentation | 1 | 0.02 | 0.02 | 0.02 | 0.02 | 0.00 | 0.0% |
| scene | 30 | 13.08 | 7.04 | 43.05 | 60.26 | 13.08 | 6.2% |
| smoothing | 30 | 1.14 | 0.28 | 7.08 | 14.10 | 1.14 | 0.5% |
| attention | 30 | 5.89 | 2.02 | 15.11 | 29.77 | 5.89 | 2.8% |
| gaze:observe | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| total | 30 | 210.70 | 86.76 | 557.41 | 651.34 | 210.70 | 100.0% |

With `segmentation.async` the segmentation stage in the table is only the hand-off
to the worker; the worker itself ran 2 times with a mean of 336.5 ms (max 472.4 ms)
off the frame loop. Capture-to-result latency: mean 283.3 ms, p50 115.4 ms.

### Observations

- **Persons dominates.** In the default fast profile YOLO11n-pose is 88-89% of the
  frame time; all non-model stages together (tracking, head pose, scene, smoothing,
  attention) cost about 5-15 ms per frame.
- **Unexplained YOLO slowdown in the synchronous runs.** In the CPU preset run the
  persons stage took 0.32 s per call, against 2.8-3.2 s in the default runs. Shorter
  runs on museum.mp4 (10 frames after 3 warm-up) reproduced this: with
  `--set segmentation.async=true` persons took 0.30 s per call (worker segmentation
  0.54 s); with synchronous segmentation 3.23 s (segmentation 3.89 s); with
  `--set segmentation.backend=none` 2.92 s; with `OMP_NUM_THREADS=4` and no
  segmentation 2.77 s. So it is not caused by the segmentation model itself or by the
  OpenMP thread count. It looks like an interaction between ultralytics/torch on the
  main thread and this machine; the cause was not identified. It must be re-checked
  on the target machine before conclusions are drawn from the 0.30 FPS figures; the
  2.4 FPS measured on 2026-11-14 with the same Sintel configuration is the other data
  point.

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
