# LensBeat — Implementation Notes

This implements the approach document's Phases 2, 3, 4, and 7 (the doc's own
"Priority Order if You Run Out of Time" top four: signal processing, temporal
stabilization, evaluation, and ROI/confidence), restructured into the
modular architecture from Section 7.

## What's implemented

| Approach doc section | File(s) |
|---|---|
| Section 7 modular architecture | Full directory structure below |
| Risk 1 / Phase 4 (ROI) | `preprocessing/face.py`, `preprocessing/roi.py` |
| Risk 2 (motion) | `preprocessing/face.py` (`motion_score`), fed into `quality/` |
| Risk 3 / Phase 2 (signal processing) | `signal_processing/filtering.py`, `spectrum.py`, `bpm.py` |
| Risk 4 / Phase 3 (BPM jumping) | `signal_processing/stabilization.py` |
| Risk 5 / Phase 7 (confidence) | `quality/signal_quality.py` |
| Final Architecture (Section 11) | `inference/pipeline.py` orchestrates all of the above |
| Section 5 (metrics) | `evaluation/metrics.py` |
| Section 8 (model versioning) | `evaluation/benchmark.py` (`log_version`, `print_version_table`) |
| Section 4 baseline / Section 6 logging | `benchmark.py` (standalone script) |

```
LensBeat/
├── model/physnet.py              -- unchanged from your original checkpoint's architecture
├── preprocessing/
│   ├── face.py                   -- mediapipe FaceMesh + EMA-smoothed tracking
│   └── roi.py                    -- 4 ROI modes (Versions A-D from Phase 4)
├── signal_processing/            -- named to avoid shadowing Python's stdlib `signal` module
│   ├── filtering.py               -- mean removal, detrend, Butterworth bandpass
│   ├── spectrum.py                -- Welch PSD
│   ├── bpm.py                     -- peak detection, harmonic correction, SNR
│   └── stabilization.py           -- median filter + EMA
├── quality/signal_quality.py     -- confidence scoring (SNR + prominence + temporal + motion)
├── inference/pipeline.py         -- orchestrates the full measurement chain
├── evaluation/
│   ├── metrics.py                 -- MAE, RMSE, Pearson, ±5/±10 BPM
│   └── benchmark.py               -- dashboard printing + version logging
├── benchmark.py                  -- standalone script: run pipeline against labeled test clips
├── app.py                        -- thin Flask/SocketIO wiring (no logic lives here)
├── templates/index.html          -- now shows Signal Quality / Motion / Confidence badges
└── requirements.txt
```

## Naming conflict caught during implementation

The approach doc's Section 7 diagram names the signal folder `signal/`.
**This name was avoided** — a top-level `signal` package shadows Python's
built-in `signal` module, which Flask-SocketIO and other libraries rely on
for OS signal handling (`SIGINT`, etc.). This was caught during testing (it
silently broke `import signal` in the same process) and the folder is named
`signal_processing/` instead. All internal imports use this name.

## What's deferred (not implemented this pass)

- **Phase 5 (PhysNet fine-tuning with augmentation)** and **Phase 6 (Optuna
  hyperparameter search)** are not implemented. These require your labeled
  dataset and Colab training time, distinct from this deployment-side
  restructuring. The training notebook from earlier work
  (`LensBeat_Fusion_TRAINING_FINAL.ipynb`) still applies for fine-tuning;
  it would need augmentation (brightness/contrast/rotation/translation,
  Section "Phase 5") added to match this approach doc's specification.
- **ROI Version comparison (Phase 4's "test A vs B vs C vs D")** — the code
  supports all four modes via `ROIExtractor(mode=...)`, but actually running
  the comparison and applying the "decision gate" requires labeled test data,
  which wasn't available in this session.

## What needs real-data calibration (flagged honestly, not guessed)

Verified with synthetic sine-wave test signals during implementation:

- **Harmonic correction threshold** (`signal_processing/bpm.py`, the `0.6 *
  candidate_power` cutoff) behaves correctly in principle (only corrects when
  the fundamental carries a comparable share of power to the detected peak)
  but the right threshold value depends on how PhysNet's actual predicted
  waveforms distribute power between fundamental and harmonic -- untested
  against real model output.
- **SNR normalization reference** (`quality/signal_quality.py`'s `snr_good=3.0`)
  is a placeholder. Testing showed real SNR values from short (~4.3s) clips
  can be small and don't cleanly separate clean-vs-noisy synthetic signals,
  likely due to `filtfilt` edge-padding distortion on short signals. This
  needs recalibration once you have real confidence-vs-actual-accuracy data
  (Section 6's "log every inference... then analyze errors" is exactly the
  mechanism for this -- run `benchmark.py` against labeled clips and check
  whether high-confidence predictions really do have lower error).
- Consider a **lower Butterworth filter order** (2 instead of 3) if edge
  distortion on 128-frame clips turns out to matter in practice -- flagged
  but not changed, since it's a real trade-off (lower order = less edge
  distortion but weaker out-of-band rejection) that should be decided from
  real data, not guessed.

## Running it

```bash
pip install -r requirements.txt
# copy your fine-tuned checkpoint next to app.py as LensBeat_PhysNet_finetuned.pth
python app.py
# open http://localhost:5000
```

For benchmarking against labeled test clips (Section 4/5):
```bash
python benchmark.py --version v1_baseline --clips-dir /path/to/test_clips \
    --model-path LensBeat_PhysNet_finetuned.pth --roi-mode landmark_multi
```
`clips-dir` expects one subfolder per clip containing `frames.npy` ((T,H,W,3)
uint8 RGB array) and `true_bpm.txt` (ground truth). Pre-extracted frame
arrays are used rather than raw video specifically because OpenCV's video
decoding was the recurring crash source during this project's earlier
training work -- sidestepping it here too.
