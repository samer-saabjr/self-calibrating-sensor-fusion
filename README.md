# Supplementary Material S2 — Code and data

**Manuscript:** *Self-Calibrating Sensor Fusion with Imperfect Supervision: Physical Validation, Expert Withdrawal, and Failure Modes*  
**Journal:** Measurement: Sensors (Elsevier)  
**Authors:** Aya Edris and Samer Saab Jr, Lebanese American University.

**Suggested Elsevier supplementary-file caption:** Supplementary Material S2: Python acquisition and analysis notebooks, standalone reproduction scripts, and time-aligned microphone, detector-output, and annotation-event datasets for binary walking detection and three-level person counting. No raw camera frames are included.

This archive provides the actual synchronized sensor-derived measurements and human annotation events for two continuous experimental sessions, the Raspberry Pi data-acquisition source notebooks, and reproducible analysis implementations for the reported experiments. Raw camera images and video frames are **not** included. The accompanying files are supplementary to the main manuscript.

## Quick start

Use Python 3.13 with the required libraries installed, and run commands from the **root of the extracted ZIP** (the directory containing this README):

```bash
python -m pip install -r requirements-analysis.txt
python code/reproduce\_tables\_1\_3.py
python code/reproduce\_tables\_9\_10.py
python code/generate\_figure\_1\_traces.py
```

The standalone numerical checks run without camera hardware or internet access and may take a few minutes. The first script verifies manuscript Tables 1–3 (dataset counts, camera errors, microphone levels, and offline affine references). The second checks all twelve learning arms in manuscript Tables 9 and 10 against published rounded values and reports the matched frozen controls. The third generates the two acoustic and camera trace panels corresponding to manuscript Figure 1, under `figures/`.

To execute an analysis notebook interactively:

```bash
cd code
jupyter lab analysis\_binary\_supervision.ipynb
```

Notebooks expect the sibling `../data/` directory and use local paths. They save generated plots to the current `code/` directory (unless a notebook cell specifies another location). Notebook outputs may be re-rendered when run; existing plot styling is not an input to the numerical results.

## Directory structure

```text
README.md
requirements-analysis.txt
VALIDATION.md
SHA256SUMS.txt
code/
  acquisition\_binary\_raspberry\_pi.ipynb
  acquisition\_counting\_raspberry\_pi.ipynb
  analysis\_binary\_supervision.ipynb
  analysis\_counting\_supervision.ipynb
  analysis\_binary\_expert\_withdrawal.ipynb
  analysis\_counting\_expert\_withdrawal.ipynb
  analysis\_binary\_initialization.ipynb
  analysis\_counting\_initialization.ipynb
  reproduce\_tables\_1\_3.py
  reproduce\_tables\_9\_10.py
  generate\_figure\_1\_traces.py
data/
  binary\_walking\_1851.csv
  person\_counting\_2163.csv
figures/
  trace\_binary.pdf
  trace\_counting.pdf
```

## Data contents and provenance

The datasets are measurements and event logs recorded at approximately one row per second on the Raspberry Pi, not synthetic examples. The *sensor and event values* in the original two CSV recordings are preserved, with no resampling, normalization, smoothing, or alteration. The source recording filenames used by the acquisition code were `Final\_Stomping\_30.csv` and `Final\_Stomping\_Multiclass\_usb.csv`; for public distribution these are renamed as shown above.

For privacy and cleanliness, the publication CSVs replace the original wall-clock `timestamp` field with `elapsed\_seconds = 0, 1, ...`. This changes only the time coordinate, not any recorded microphone measurements, expert predictions, or human annotation events. The original binary file appended five non-sample logger summary lines; those are excluded from the publication CSV. These are deterministic packaging changes, not analytical transformations. No raw camera images or video are shared.

|File|Valid rows|Task|State labels in manuscript|
|-|-:|-|-|
|`data/binary\_walking\_1851.csv`|1,851|Binary walking detection|-1 not walking; +1 walking|
|`data/person\_counting\_2163.csv`|2,163|Person counting|0, 1, 2 people|

Both datasets provide these columns: `elapsed\_seconds`, `RMS\_mic1`, `dBFS\_mic1`, `RMS\_mic2`, `dBFS\_mic2`, one camera-derived expert field (`human` for binary or `human\_count` for counting), and one annotation event field (`mouse` or `mouse\_event`). RMS is the mean-removed RMS of ADC counts; dBFS equals `20 log10(RMS / 2^14)` up to recorded rounding.

The annotation columns are **event streams, not ground-truth state labels**. For binary walking, a valid button event toggles occupancy; adjacent duplicate event codes are suppressed before parity reconstruction. For counting, event code 1 toggles the one-person class and code 2 toggles the two-person class; this is reconstructed by the notebooks and the standalone reproduction script. Both recordings start in the unoccupied/not-walking state.

Reconstructed class counts: binary (1,168 not-walking / 683 walking); counting (860 / 559 / 744 for classes 0 / 1 / 2). See manuscript Section 3.4.2 and Tables 1–2.

## File-to-manuscript map

|Manuscript result|Reproduction source|
|-|-|
|Tables 1–3; dataset baselines, microphone levels, offline least-squares references|`code/reproduce\_tables\_1\_3.py` plus both CSVs|
|Figure 1: representative microphone and camera traces|`code/generate\_figure\_1\_traces.py` (creates `figures/trace\_binary.pdf` and `figures/trace\_counting.pdf`)|
|Table 4; Figures 2–3: binary supervising trajectories|`analysis\_binary\_supervision.ipynb`|
|Table 5; Figures 4–5: counting supervising trajectories|`analysis\_counting\_supervision.ipynb`|
|Tables 6–8; Figures 6–9: expert withdrawal and cutoff sweeps|`analysis\_binary\_expert\_withdrawal.ipynb`; `analysis\_counting\_expert\_withdrawal.ipynb`|
|Tables 9–10; Figure 10: corrupted initialization and frozen controls|`reproduce\_tables\_9\_10.py`; `analysis\_binary\_initialization.ipynb`; `analysis\_counting\_initialization.ipynb`|
|Figures 11–13: acquisition architecture and room layouts|TikZ figures in the manuscript LaTeX source; no external image data are required|

Some analysis notebooks generate additional diagnostic plots that were not included in the manuscript. Their source code and outputs are retained for auditability. The main numerical experiments are deterministic, given the published datasets and selected parameters.

## Cutoff sweep specification

For expert withdrawal (Section 3.5.4 and Section 4.4), both analyses use nine **irregularly spaced** training cutoffs, not uniform 200-sample increments:

* Binary: 200, 300, 400, 600, 800, 1000, 1250, 1500, 1600.
* Counting: 200, 300, 400, 600, 800, 1000, 1250, 1500, 1800.

At the main cutoff of 1,250, the binary microphone-only deployment error is 1.00% and counting error is 22.70%. Across the nine binary cutoffs, deployment error ranges from 0.86% to 1.31%; for counting, from 17.54% to 27.07% (rounding as printed by the notebooks). Cutoff comparisons use different training and deployment windows; they are not independent repetitions of the experiment.

## Reproducing corrupted-start experiments

`python code/reproduce\_tables\_9\_10.py` runs each of three supervising trajectories (annotated, expert-only, fused) from both a silenced and an inverted starting point, separately for binary and counting. It prints all twelve learning results and frozen controls and raises an error if a reported value departs from manuscript rounding tolerance. For interactive visual diagnostics, use the corresponding `analysis\_\*\_initialization.ipynb` files. The script and notebooks use the same filter and learning rules, including the full-recording mean subtraction in the binary case.

## Figure 1 regeneration

`python code/generate\_figure\_1\_traces.py` recreates the binary (samples 0–200) and counting (samples 1,440–1,640) representative windows from the released CSVs. Each panel presents two microphone levels, the camera's class output, and the annotated class, with occupied-state shading. The plots are regenerated directly from the data; font sizes and color styling may differ from the final typeset manuscript images.

## Acquisition notebooks and environment

`code/acquisition\_binary\_raspberry\_pi.ipynb` and `code/acquisition\_counting\_raspberry\_pi.ipynb` contain the original acquisition implementation. They are provided as documentation and **should not be executed on a reviewer workstation**; they require a Raspberry Pi, two ADS1115 ADCs (I2C 0x48 and 0x49) connected to MAX4466 modules, a camera, a compatible input event device, and their platform-specific drivers and libraries (Blinka/CircuitPython ADS1x15, Picamera2 for the binary session, OpenCV, Ultralytics YOLO, python-evdev). A fresh acquisition would also write wall-clock timestamps and the source filenames, by design. The supplied analysis CSVs make physical hardware unnecessary for reproducing the existing experimental results.

The analysis requirements are listed in `requirements-analysis.txt`. Tests were conducted with Python 3.13.5 on Linux. Acquisition-library versions depend on Raspberry Pi OS and were not validated as part of this release.

## Scope and limitations

The release contains one continuous recording per task, not independent repeated trials. The expert is a camera-derived person detector; the inexpert sensors are microphones. Binary analysis subtracts the mean of the entire recording before filtering, so its preprocessing is retrospective even when the update step is causal. The counting affine analysis does not use that full-session centering. The camera-derived time series and operator events are included, but not identifiable video frames. See the main manuscript for detailed limitations, methods, and interpretation.

