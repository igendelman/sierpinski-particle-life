"""
visualize_vispy.py
~~~~~~~~~~~~~~~~~~~~

This script demonstrates how to visualize a Sierpinski‐fractal world and a set of
moving particles on the GPU using the VisPy library. It builds on the GPU
physics engine provided in ``engine_gpu.py`` (or the equivalent GPU‑enabled
``engine.py`` in your environment) to generate the graph structure and update
particle positions.  The resulting visualization renders all edges and
particles on your 5090 GPU via OpenGL shaders, allowing interactive pan/zoom
and smooth animation at large scales.

Prerequisites:

* ``engine_gpu.py`` or a compatible module exposing ``SierpinskiWorld``,
  ``Engine`` and ``Particle`` classes.  Adjust the import below if your
  engine lives under a different module name.  The engine must be
  initialized with ``device='cuda'`` support (see README).
* VisPy installed in your environment.  You can install it via conda or pip
  (e.g. ``conda install -c conda-forge vispy``).  VisPy requires an OpenGL
  context with at least version 2.1 and uses your GPU for rendering
  heavy‑weight datasets【261341323866250†L70-L72】【206563424294519†L150-L152】.

Running the script:

.. code-block:: bash

    conda activate particle_cuda
    python visualize_vispy.py  --max-level 6 --num-particles 200

You should see an interactive window with the Sierpinski gasket drawn in
faint grey lines and red dots representing particles that perform biased
random walks.  Use the mouse wheel or trackpad to zoom and drag to pan
around the fractal.  Press Esc to close the window.
"""

import argparse
from typing import List
import numpy as np
from vispy import scene, app
from vispy.scene import visuals
from vispy import app as vispy_app

import subprocess
try:
    import imageio_ffmpeg as _iio_ffmpeg
except Exception:
    _iio_ffmpeg = None

#
# Colour mapping (legacy charge and optional multi-species palette)
CHARGE_COLORS = {
    1: (1.0, 0.0, 0.0, 1.0),   # positive charge -> red
    -1: (0.0, 0.0, 1.0, 1.0),  # negative charge -> blue
}
# Distinct RGBA colours for species palette (cycled if num_species > len list)
SPECIES_PALETTE = [
    (0.90, 0.10, 0.10, 1.0),  # red
    (0.10, 0.55, 0.95, 1.0),  # blue
    (0.10, 0.80, 0.25, 1.0),  # green
    (0.95, 0.75, 0.10, 1.0),  # yellow
    (0.70, 0.20, 0.80, 1.0),  # purple
    (0.95, 0.45, 0.10, 1.0),  # orange
    (0.10, 0.85, 0.80, 1.0),  # teal
    (0.60, 0.60, 0.60, 1.0),  # grey
]

# Attempt to import the GPU‑enabled engine.  Adjust this import to match
# your environment.  If your GPU engine is named ``engine.py``, replace
# ``engine_gpu`` with ``engine``.  The engine must be on your Python path.
from engine_polar_bind import SierpinskiWorld, Engine, Particle

# Precompute sqrt(3)/2 for equilateral transform
SQRT3_OVER_2 = np.sqrt(3.0) * 0.5

def to_equilateral(coord: tuple[int, int]) -> tuple[float, float]:
    """Map (x,y) integer grid coords to equilateral grid coords."""
    x, y = coord
    return (x + 0.5 * y, SQRT3_OVER_2 * y)

def build_edge_segments(graph):
    """Construct line segment geometry for a GraphLevel.

    This helper iterates through the CSR adjacency of a ``GraphLevel`` and
    collects both the end points of each undirected edge and the index
    pairs for those vertices.  It returns the positions for the ``Line``
    visual as well as a second array of index pairs, which can be used
    later to compute per-vertex colours for each segment.

    Parameters
    ----------
    graph : GraphLevel
        The GraphLevel object containing ``indptr``, ``indices`` and
        ``idx_to_vertex`` attributes.

    Returns
    -------
    segments : numpy.ndarray
        Array of shape (num_edges*2, 2) of x,y coordinates for line
        segments in integer grid coordinates.  Each consecutive pair
        defines one segment.
    pairs : numpy.ndarray
        Array of shape (num_edges, 2) of integer vertex indices.
    """
    # Retrieve adjacency arrays from GPU and move to CPU for iteration
    indptr = graph.indptr.cpu().numpy()
    indices = graph.indices.cpu().numpy()
    segments = []
    pairs = []
    for i, v in enumerate(graph.idx_to_vertex):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        for nb_idx in indices[start:end]:
            v0 = v
            v1 = graph.idx_to_vertex[int(nb_idx)]
            # Append the coordinates for the segment endpoints
            segments.append([v0[0], v0[1]])
            segments.append([v1[0], v1[1]])
            # Record the index pair for this segment
            pairs.append((i, int(nb_idx)))
    return np.array(segments, dtype=np.float32), np.array(pairs, dtype=np.int32)


def main(max_level: int, num_particles: int, spawn_level: int, args=None):
    if args is None:
        class _A: pass
        args = _A()
        args.alpha = 0.6; args.eta = 0.03; args.p_bind = 0.05; args.p_unbind_base = 0.01
        args.bind_tau = 200; args.bind_same_charge = False; args.bind_need_alignment = True
        args.co_move_bias = 1.4; args.particle_size = 6.0
        args.vis_fps = 24
        args.trail_gain = 1.5; args.trail_decay = 0.95; args.trail_deposit = 0.3; args.edge_alpha = 0.4
    # Build the world and engine
    world = SierpinskiWorld(max_level=max_level)
    engine = Engine(world)
    # Prefer stable Qt backends; avoid GLFW on some X11/WSL setups
    try:
        for _bk in ('pyqt6', 'pyqt5', 'pyside6', 'pyside2', 'sdl2'):
            try:
                vispy_app.use_app(_bk)
                break
            except Exception:
                continue
    except Exception:
        pass

    # --- Multi-species and interaction matrix options ---
    num_species = int(getattr(args, 'num_species', 1))
    num_species = max(1, num_species)
    interaction_preset = getattr(args, 'interaction_preset', 'none')
    # K will stay None unless we actually enable matrix mode (num_species>1)
    K = None
    def build_interaction_matrix(preset: str, n: int):
        import numpy as _np
        if n <= 1 or preset in (None, 'none'):
            return None
        # Base zero matrix
        M = _np.zeros((n, n), dtype=_np.float32)
        if preset == 'ring':
            # attract immediate neighbor (+0.4), slight repel others (-0.1)
            for i in range(n):
                M[i, (i+1)%n] = 0.4
                for j in range(n):
                    if j != (i+1)%n and j != i:
                        M[i, j] = -0.1
        elif preset == 'random':
            rng = _np.random.default_rng(getattr(args, 'matrix_seed', None))
            M = rng.uniform(-0.3, 0.5, size=(n, n)).astype(_np.float32)
            _np.fill_diagonal(M, 0.0)
        elif preset == 'clusters':
            # Two clusters: [0..m-1] and [m..n-1]; strong intra, weak inter
            m = n//2
            M[:m, :m] = 0.45; M[m:, m:] = 0.45
            M[:m, m:] = -0.15; M[m:, :m] = -0.15
            _np.fill_diagonal(M, 0.0)
        else:
            return None
        return M.tolist()

    # Create particle instances at the chosen spawn level.  Assign each
    # particle a random charge of +1 or -1 to demonstrate the effect of
    # charge-based attraction and repulsion.  You can control the ratio
    # of positive to negative charges by changing the probability below.
    particles: List[Particle] = []
    verts = world.graphs[spawn_level].idx_to_vertex
    rng = np.random.default_rng()
    for pid in range(num_particles):
        idx = rng.integers(0, len(verts))
        vertex = verts[idx]
        # Flip a coin: half the particles positive charge, half negative
        charge = 1 if rng.random() < 0.5 else -1
        p = Particle(pid, vertex, spawn_level, charge=charge)
        if num_species > 1:
            # Random species id in 0..num_species-1
            p.species_id = int(rng.integers(0, num_species))
        particles.append(p)

    if num_species > 1:
        K = build_interaction_matrix(interaction_preset, num_species)

    # --- Blink/Halo state for scale transitions ---
    # Track previous level per particle (fallback to spawn_level if attribute absent)
    prev_levels = {i: getattr(p, 'level', spawn_level) for i, p in enumerate(particles)}
    num_p = len(particles)
    # Per-particle halo alpha (decays each frame) and color (RGBA)
    import numpy as _np
    halo_alpha = _np.zeros(num_p, dtype=_np.float32)
    # Default colors; will be set on transitions
    HALO_WHITE = (_np.array([1.0, 1.0, 1.0, 1.0], dtype=_np.float32))
    HALO_GREEN = (_np.array([0.0, 1.0, 0.0, 1.0], dtype=_np.float32))
    halo_colors = _np.tile(HALO_WHITE, (num_p, 1)).astype(_np.float32)
    HALO_DECAY = 0.85  # controls how fast the halo fades (smaller = faster)
    HALO_BASE_SIZE = 6.0
    HALO_BOOST = 18.0  # additional size when alpha=1


    # Prepare edge geometry for visualization at the deepest level.  We
    # request both the coordinates and the index pairs so that we can
    # compute per-vertex colours later on.
    graph = world.graphs[max_level]
    edge_segments, edge_pairs = build_edge_segments(graph)
    # Transform right‑triangle coordinates to equilateral ones; store
    # separately so that we can update colours without recomputing
    edge_segments_equil = np.array([to_equilateral((x, y)) for (x, y) in edge_segments], dtype=np.float32)
    # Create the VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', show=True, size=(2000, 2000), title='Sierpinski World')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.PanZoomCamera()  # 2D camera with pan/zoom controls
    # Set camera bounds to the fractal size (0..den x 0..den)
    den = 1 << max_level
    view.camera.set_range(x=(0, den), y=(0, den * SQRT3_OVER_2))

    # --- FFmpeg recording helpers ---
    class _Recorder:
        def __init__(self):
            self.proc = None
            self.fps = int(getattr(args, 'vis_fps', 24))
            self.crf = '20'
            self.out_path = getattr(args, 'record', None)
            self.frame_count = 0
            # VisPy expects plain Python ints for render size
            cs = canvas.size
            self.size = (int(cs[0]), int(cs[1]))  # (W, H) fixed at start
        def start(self):
            if not self.out_path or self.proc is not None:
                return
            W, H = self.size
            ffmpeg_exe = 'ffmpeg'
            try:
                if _iio_ffmpeg is not None:
                    ffmpeg_exe = _iio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass
            cmd = [
                ffmpeg_exe, '-y',
                '-f', 'rawvideo', '-pixel_format', 'rgba',
                '-video_size', f'{W}x{H}', '-framerate', str(self.fps),
                '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', self.crf, self.out_path
            ]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                self.frame_count = 0
                print(f'[record] started → {self.out_path} @ {self.fps} fps, size={W}x{H}')
            except FileNotFoundError:
                print('[record] ffmpeg not found. Try `conda install -c conda-forge ffmpeg` or `pip install imageio-ffmpeg`. Recording disabled.')
                self.proc = None
            except Exception as e:
                print('[record] failed to start ffmpeg:', e)
                self.proc = None
        def restart_with_size(self, size_wh: tuple[int, int]):
            # Close any existing proc and start with new WxH
            self.stop()
            self.size = (int(size_wh[0]), int(size_wh[1]))
            self.start()
        def stop(self):
            if self.proc is not None:
                try:
                    self.proc.stdin.flush(); self.proc.stdin.close()
                except Exception:
                    pass
                self.proc.wait()
                print(f'[record] finished {self.frame_count} frames → {self.out_path}')
                self.proc = None
        def write_frame(self, frame_rgba: np.ndarray):
            if self.proc is None:
                return
            try:
                self.proc.stdin.write(frame_rgba.tobytes())
                self.frame_count += 1
            except Exception as e:
                print('[record] write error:', e)
                self.stop()

    recorder = _Recorder()

    # Colour field for each vertex: accumulate RG components as particles
    # visit vertices.  Initialise with zeros.  We'll use the index->vertex
    # mapping from ``graph.idx_to_vertex``.
    color_field = {v: np.zeros(3, dtype=np.float32) for v in graph.idx_to_vertex}
    # Constants controlling decay and deposition of colour.  Each frame
    # multiplies the current colour by DECAY_FACTOR to fade trails, and
    # deposits DEPOSIT_AMOUNT * CHARGE_COLORS[charge][:3] when a particle
    # visits a vertex.
    DECAY_FACTOR = float(getattr(args, 'trail_decay', 0.95))
    DEPOSIT_AMOUNT = float(getattr(args, 'trail_deposit', 0.3))
    EDGE_ALPHA = float(getattr(args, 'edge_alpha', 0.4))
    TRAIL_GAIN = float(getattr(args, 'trail_gain', 1.5))

    # Add the gasket edges.  We'll set their colours dynamically based on
    # ``color_field`` in the update callback below.  Start with all
    # edges drawn semi-transparent grey.
    initial_colors = np.array([(0.5, 0.5, 0.5, EDGE_ALPHA)] * len(edge_segments_equil), dtype=np.float32)
    line = visuals.Line(pos=edge_segments_equil,
                        connect='segments',
                        color=initial_colors,
                        width=1.0)
    view.add(line)

    # Initialize the particle markers
    def particle_positions() -> np.ndarray:
        """Return the current positions of all particles in equilateral coords."""
        return np.array([to_equilateral(p.vertex) for p in particles], dtype=np.float32)

    def particle_colors() -> np.ndarray:
        """Return RGBA per particle.
        Uses species palette if multiple species are enabled; otherwise uses charge colours.
        """
        cols = []
        use_species = any(getattr(p, 'species_id', 0) != 0 for p in particles) or (num_species > 1)
        if use_species:
            for p in particles:
                sid = int(getattr(p, 'species_id', 0))
                col = SPECIES_PALETTE[sid % len(SPECIES_PALETTE)]
                cols.append(col)
        else:
            for p in particles:
                col = CHARGE_COLORS.get(getattr(p, 'charge', 1), (1.0, 1.0, 1.0, 1.0))
                cols.append(col)
        return np.array(cols, dtype=np.float32)

    marker_visual = visuals.Markers()
    # Use per-particle colours to reflect the charge of each particle
    marker_visual.set_data(particle_positions(), face_color=particle_colors(), size=args.particle_size)
    view.add(marker_visual)


    
    # Orientation ticks and binding links
    orient_lines = visuals.Line()
    link_lines = visuals.Line()
    view.add(orient_lines)
    view.add(link_lines)
# A second Markers layer for "halo blinks" around particles
    halo_visual = visuals.Markers()
    # Initialize halos as invisible
    halo_visual.set_data(particle_positions(), face_color=_np.zeros((len(particles), 4), dtype=_np.float32), size=HALO_BASE_SIZE)
    view.add(halo_visual)

    # Simulation update function called every frame
    # Simulation kwargs derived from CLI args
    sim_kwargs = dict(
        alpha=args.alpha,
        eta=args.eta,
        p_bind=args.p_bind,
        p_unbind_base=args.p_unbind_base,
        bind_tau=args.bind_tau,
        bind_same_charge=args.bind_same_charge,
        bind_need_alignment=getattr(args, 'bind_need_alignment', True),
        co_move_bias=args.co_move_bias,
    )
    # Pass-through for new binding controls (engine defaults used if omitted)
    sim_kwargs['bind_cooldown_frames'] = getattr(args, 'bind_cooldown', 45)
    sim_kwargs['max_bind_frames']      = getattr(args, 'max_bind_frames', 150)
    sim_kwargs['matrix_bind_scale']    = getattr(args, 'matrix_bind_scale', 1.0)
    if K is not None:
        sim_kwargs['interaction_matrix'] = K
        sim_kwargs['interaction_mode'] = 'matrix'

    def update(event):
        # Advance all particles using the GPU engine (bias is None for unbiased walk)
        engine.step_particles(particles, **sim_kwargs)

        # --- HALO: detect scale transitions and blink ---
        # Decay existing halos
        halo_alpha[:] *= HALO_DECAY
        # Capture previous levels before/after step to detect up/down
        # (Note: we do it here because engine.step_particles already advanced positions above.)
        for i, p in enumerate(particles):
            old_lvl = prev_levels.get(i, spawn_level)
            new_lvl = getattr(p, 'level', old_lvl)
            if new_lvl > old_lvl:
                # Went "down" to finer detail -> green halo
                halo_alpha[i] = 1.0
                halo_colors[i] = HALO_GREEN
            elif new_lvl < old_lvl:
                # Went "up" to coarser scale -> white halo
                halo_alpha[i] = 1.0
                halo_colors[i] = HALO_WHITE
            prev_levels[i] = new_lvl

        # Update halo visual (position follows particles; alpha controls visibility + size)
        pos = particle_positions()
        # Face colors with decayed alpha
        halo_rgba = halo_colors.copy()
        # Orientation ticks
        import numpy as _np
        tick_len = 0.6
        segs = _np.zeros((len(particles)*2, 2), dtype=_np.float32)
        kk = 0
        for i,p in enumerate(particles):
            x, y = p.vertex
            X = x + 0.5*y; Y = y*(3**0.5)*0.5
            th = getattr(p, 'theta', 0.0)
            dx = tick_len * _np.cos(th); dy = tick_len * _np.sin(th)
            segs[2*kk] = (X, Y)
            segs[2*kk+1] = (X+dx, Y+dy)
            kk += 1
        orient_lines.set_data(segs.reshape(-1,2), connect='segments', width=1.0, color=(1,1,0,0.7))

        # Binding links
        links = []
        for i,p in enumerate(particles):
            j = getattr(p, 'bound_to', -1)
            if j != -1 and j < len(particles):
                x1,y1 = p.vertex; x2,y2 = particles[j].vertex
                X1 = x1 + 0.5*y1; Y1 = y1*(3**0.5)*0.5
                X2 = x2 + 0.5*y2; Y2 = y2*(3**0.5)*0.5
                links.extend([(X1,Y1),(X2,Y2)])
        if links:
            link_lines.set_data(_np.array(links, dtype=_np.float32), connect='segments', width=1.0, color=(0.6,0.6,1,0.6))
        else:
            link_lines.set_data(_np.zeros((0,2), dtype=_np.float32), connect='segments')
        halo_rgba[:, 3] = halo_alpha
        # Size scales with alpha
        halo_sizes = HALO_BASE_SIZE + HALO_BOOST * halo_alpha
        halo_visual.set_data(pos, face_color=halo_rgba, size=halo_sizes)
        # Decay the colour field at each vertex to create fading trails
        for v in color_field:
            color_field[v] *= DECAY_FACTOR
        # Deposit colour based on the charge or species of each particle
        use_species = any(getattr(p, 'species_id', 0) != 0 for p in particles) or (num_species > 1)
        for i, p in enumerate(particles):
            if use_species:
                sid = int(getattr(p, 'species_id', 0))
                col_rgba = SPECIES_PALETTE[sid % len(SPECIES_PALETTE)]
            else:
                col_rgba = CHARGE_COLORS.get(getattr(p, 'charge', 1), (1.0, 1.0, 1.0, 1.0))
            deposit_rgb = np.array(col_rgba[:3], dtype=np.float32)
            color_field[p.vertex] += deposit_rgb * DEPOSIT_AMOUNT
            # Clamp to [0, 1] to avoid overflows
            color_field[p.vertex] = np.clip(color_field[p.vertex], 0.0, 1.0)
        # Compute per-vertex colours for the line segments.  We assign a
        # colour to each endpoint based on the colour_field of that vertex,
        # and append a constant alpha.
        line_colors_list = []
        for i0, i1 in edge_pairs:
            v0 = graph.idx_to_vertex[int(i0)]
            v1 = graph.idx_to_vertex[int(i1)]
            col0 = np.clip(color_field[v0] * TRAIL_GAIN, 0.0, 1.0)
            col1 = np.clip(color_field[v1] * TRAIL_GAIN, 0.0, 1.0)
            line_colors_list.append(np.append(col0, EDGE_ALPHA))
            line_colors_list.append(np.append(col1, EDGE_ALPHA))
        line_colors = np.array(line_colors_list, dtype=np.float32)
        # Update the line visual with new colours (positions remain the same)
        line.set_data(pos=edge_segments_equil,
                      color=line_colors,
                      connect='segments',
                      width=1.0)
        # Update marker positions and colours from new particle states
        marker_visual.set_data(particle_positions(), face_color=particle_colors(), size=args.particle_size)

        # --- Recording: grab frame & stream to ffmpeg (robust) ---
        if getattr(args, 'record', None):  # recording requested
            try:
                # Prefer backend-default render size (avoids RenderBuffer resize issues)
                frame = canvas.render()  # returns HxWx4 uint8 (RGBA)
                if frame is None:
                    return
                Hf, Wf = int(frame.shape[0]), int(frame.shape[1])
                # If ffmpeg is running with a different WxH, restart once to match
                if recorder.proc is not None:
                    W0, H0 = recorder.size
                    if (Wf, Hf) != (int(W0), int(H0)) and recorder.frame_count == 0:
                        recorder.restart_with_size((Wf, Hf))
                # If ffmpeg not started yet (lazy), start now with detected size
                if recorder.proc is None:
                    recorder.size = (Wf, Hf)
                    recorder.start()
                # Write the frame (may still be the first)
                recorder.write_frame(frame)
            except Exception as e:
                # Log and continue; don't kill the sim loop
                print('[record] capture error:', e)



    # Create a timer to drive the animation using the chosen FPS
    _vfps = max(1, int(getattr(args, 'vis_fps', 24)))
    timer = app.Timer(interval=1 / _vfps, connect=update, start=True)
    app.run()

    # On exit, finalize recording if still open
    try:
        recorder.stop()
    except Exception:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-level", type=int, default=6)
    parser.add_argument("--spawn-level", type=int, default=6)
    parser.add_argument("--num-particles", type=int, default=200)
    # New knobs
    parser.add_argument("--alpha", type=float, default=0.6, help="polarity → motion bias")
    parser.add_argument("--eta", type=float, default=0.03, help="orientation jitter per step")
    parser.add_argument("--p-bind", dest="p_bind", type=float, default=0.05, help="bind probability per neighbor per step")
    parser.add_argument("--p-unbind-base", dest="p_unbind_base", type=float, default=0.01, help="base unbinding probability per step")
    parser.add_argument("--bind-tau", type=int, default=200, help="frames to prefer staying bound")
    parser.add_argument("--bind-same-charge", action="store_true", help="allow binding of like charges")
    parser.add_argument("--no-align-needed", dest="bind_need_alignment", action="store_false", help="do not require alignment to bind")
    parser.add_argument("--co-move-bias", type=float, default=1.4, help="bias to choose same/similar move when bound")
    parser.add_argument("--particle-size", type=float, default=6.0, help="marker size for particles")
    parser.add_argument("--num-species", type=int, default=1, help="number of species (>=1). If >1, enables matrix interactions in engine")
    parser.add_argument("--interaction-preset", type=str, default="none", choices=["none","ring","random","clusters"], help="preset for species interaction matrix (only used if num-species>1)")
    parser.add_argument("--matrix-seed", type=int, default=None, help="seed for random interaction matrix (preset=random)")
    parser.add_argument("--bind-cooldown", type=int, default=45,
                        help="frames after unbinding during which a pair cannot rebind")
    parser.add_argument("--max-bind-frames", type=int, default=150,
                        help="hard cap (in frames) on a single bind's duration")
    parser.add_argument("--matrix-bind-scale", type=float, default=1.0,
                        help="scale factor for interaction matrix when applied to binding probability")
    parser.add_argument('--record', type=str, default=None, help='output mp4 path to enable recording (uses ffmpeg)')
    parser.add_argument("--vis-fps", type=int, default=24, help="visualization frames per second (also used for recording)")
    parser.add_argument("--trail-gain", type=float, default=1.5, help="multiplier for trail brightness on lattice edges")
    parser.add_argument("--trail-decay", type=float, default=0.95, help="per-frame decay factor for trails (lower fades faster)")
    parser.add_argument("--trail-deposit", type=float, default=0.3, help="amount of color deposited per visit")
    parser.add_argument("--edge-alpha", type=float, default=0.4, help="alpha transparency for lattice edges")
    args = parser.parse_args()
    main(args.max_level, args.num_particles, args.spawn_level, args)
