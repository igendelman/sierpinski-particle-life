# Particle Life in Sierpinski Space

A GPU-accelerated simulation of **particle life** on a **Sierpinski fractal**, where **scale is an explicit coordinate**. The system couples a fast, CUDA-backed graph engine with a multi-species interaction model and a clear visual layer (halos for scale hops, species-colored particles, and glowing trail accumulation on the lattice).


## Demo Videos

<video width="640" controls>
  <source src="https://youtube.com/shorts/4CcMPzsbshs?si=yMqHSp3-8Z6fNBY6" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

##  Features
- **True equilateral embedding** of the Sierpinski gasket with fast CSR adjacency per level.
- **Multi-species interactions** via a user-defined matrix (attraction/repulsion by species).
- **Binding dynamics** with tunable probability, max duration, and cooldown.
- **Heading bias** + jitter with drift-correction; scale transitions rendered with halos.
- **Glowing trails** (persistent color deposits with decay/gain controls).
- Optional **MP4 recording** (ffmpeg pipe), and user-set **visualization fps**.

---

##  Requirements

- **OS**: Linux / Windows (WSL supported with a stable Qt backend)
- **Python**: 3.10–3.12
- **GPU**: NVIDIA CUDA-capable (developed on 5090)
- **Libraries**:
  - [PyTorch](https://pytorch.org/) with **CUDA** (for GPU stepping)
  - `numpy`
  - `vispy` (rendering)
  - A **Qt** backend (prefer **PyQt6** or **PyQt5**) for stability  
    ```bash
    conda install pyqt6
    ```
  - **Optional (recording)**: `ffmpeg` or `imageio-ffmpeg`  
    ```bash
    conda install -c conda-forge ffmpeg
    # or
    pip install imageio-ffmpeg
    ```

### Quick install (conda)
```bash
conda create -n particle_cuda python=3.11 -y
conda activate particle_cuda

# PyTorch with CUDA (pick the line for your CUDA version from pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Rendering
conda install -c conda-forge vispy pyqt6 -y

# Utilities
pip install numpy imageio-ffmpeg
```

**Tip (Linux/WSL):** prefer Qt backends over GLFW to avoid X11 IME crashes.  
You can set: `export VISPY_USE_APP=pyqt6`

---

## Project Layout

```
engine_polar_bind.py   # Graph engine + particle stepping (GPU)
vis_polar_bind.py      # Visualization and CLI; runs the sim and optional recording
```

---

## Quickstart

**Basic run (default species, no recording):**
```bash
python vis_polar_bind.py --max-level 6 --spawn-level 6 --num-particles 200
```

**Multi-species with ring interactions, slower viz (24 fps):**
```bash
python vis_polar_bind.py --max-level 6 --spawn-level 6 --num-particles 300   --num-species 6 --interaction-preset ring --matrix-seed 123   --vis-fps 24
```

**Add binding churn and brighter trails:**
```bash
python vis_polar_bind.py --max-level 6 --spawn-level 6 --num-particles 300   --num-species 6 --interaction-preset ring --matrix-seed 123   --p-bind 0.004 --p-unbind-base 0.8 --bind-tau 90   --bind-cooldown 60 --max-bind-frames 120 --co-move-bias 1.2   --trail-gain 2.5 --trail-decay 0.96 --edge-alpha 0.55 --vis-fps 24
```

**Record a short MP4 (auto-starts on first frame):**
```bash
python vis_polar_bind.py ... --record demo.mp4 --vis-fps 24
```

---

## CLI Reference

### Core
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-level` | int | 6 | Deepest refinement level of the Sierpinski gasket. |
| `--spawn-level` | int | 6 | Level at which particles are initially placed. |
| `--num-particles` | int | 200 | Number of particles. |
| `--vis-fps` | int | 24 | Visualization (and recording) frames per second. |

### Motion / Heading
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--alpha` | float | 0.6 | Heading anisotropy (bias along orientation). |
| `--eta` | float | 0.03 | Orientation jitter per step. |
| `--co-move-bias` | float | 1.4 | When bound, relative co-motion bias. |

### Multi-Species Interactions
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-species` | int | 1 | Number of species (≥1). Enables matrix mode when >1. |
| `--interaction-preset` | {none, ring, random, clusters} | none | Build a K matrix for species interactions. |
| `--matrix-seed` | int | — | RNG seed for `random` preset. |
| `--matrix-bind-scale` | float | 1.0 | Scale factor on matrix binding probability. |

### Binding Dynamics
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--p-bind` | float | 0.05 | Bind probability per neighbor per step. |
| `--p-unbind-base` | float | 0.01 | Baseline unbinding probability per step. |
| `--bind-tau` | int | 200 | Time constant increasing unbind chance. |
| `--bind-cooldown` | int | 45 | Frames partners must wait before re-binding. |
| `--max-bind-frames` | int | 150 | Hard cap on any single bind duration. |
| `--bind-same-charge` | flag | off | Allow like-charge binding. |
| `--no-align-needed` | flag | off | Do not require alignment to bind. |

### Visuals (Trails / Particles)
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--particle-size` | float | 6.0 | Marker size. |
| `--trail-gain` | float | 1.5 | Multiplier to brighten trail colors. |
| `--trail-decay` | float | 0.95 | Per-frame trail decay (higher = longer persistence). |
| `--trail-deposit` | float | 0.3 | Color deposited per vertex visit. |
| `--edge-alpha` | float | 0.4 | Edge transparency. |

### Recording (optional)
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--record` | str | — | Output MP4 path; auto-starts on first frame. |


---
