# MCCA-PRO

Anonymous research code for multi-robot coverage path planning on obstacle grids.
The tile-first planner constructs a global tiling, balances connected robot
regions, and generates obstacle-safe closed patrol routes with shortcut
postprocessing.

## Setup

Use Python 3.10 or newer (tested with Python 3.11). From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Generate a small planning example with five robots:

```bash
python lab4_path_planning_tile_first_v2_3_2.py --sizes 20 --robots 5 --seed 42
```

Figures and CSV/JSON metrics are saved under `LAB_DATA/labD_path_planning_<timestamp>/`.
Use `--profile paper` for the configured map-size sweep. For a step-by-step
illustration, run `python tile_first_5step_demo.py`; outputs go to
`tile_first_5step_output/`.

Run the comparison suite and its tests:

```bash
cd "labG-Comprehensive Comparison"
python -m unittest discover -s tests
python run_comparison.py --output-dir outputs --skip-visualizations
```

The suite writes CSV metrics and `outputs/comparison_report.md` for six seeds
on 30 × 30 grids with four robots. Omit `--skip-visualizations` to generate path
figures. External baseline source code is not bundled; official baselines run
only when their repositories are installed. See the
[comparison README](labG-Comprehensive%20Comparison/README.md) for details.

## Contents

- `mainline_tile_first_v2_3_2.py`: core planner (`TileFirstMCPP`, also exposed as `FACTMCCA`).
- `benchmark_MCCA_tile_first_v2_3_2.py`: experiment and plotting utilities.
- `lab4_path_planning_tile_first_v2_3_2.py`: path-planning experiments.
- `labC_*`, `labD-SOTA-Benchmarking/`, `labE_*`, `labF*`: additional experiments and plots.
- `labG-Comprehensive Comparison/`: shared-grid comparisons and coverage/path tests.
- `real-lab/`: optional ROS 2 / Nav2 integration and occupancy maps.

## Robot configuration

The optional robot integration requires a configured ROS 2 / Nav2 environment
and component-specific dependencies (for example, Flask, Requests, Pillow,
OpenCV, and Piper/pygame for speech). These are separate from the simulation
requirements. 

Copy `real-lab/autopatrol_robot/config.yaml` to `config.local.yaml` and point
`AUTOPATROL_CONFIG` to that file. Map and image paths are resolved relative to
the configuration file. Set `network.server_url` to your monitoring server;
the supplied default is loopback. Optional overrides are `AUTOPATROL_MAP_DIR`
for the web servers, `AUTOPATROL_IMAGE_DIR` for camera captures, and
`AUTOPATROL_TTS_DIR` for the Piper model directory (default: `~/MODELS/tts`).
