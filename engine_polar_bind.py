from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional
import random
import torch
import cudf
import cugraph

# Type aliases for clarity
Coord = Tuple[int, int]
Vertex = Coord


@dataclass
class GraphLevel:
    """Graph (vertices+edges) for a single refinement level of the Sierpinski gasket.
       Vertices are unique geometric points; edges are undirected.
    """
    level: int
    vertices: Set[Vertex] = field(default_factory=set)
    adj: Dict[Vertex, Set[Vertex]] = field(default_factory=lambda: defaultdict(set))
    # GPU structures
    vertex_to_idx: Dict[Vertex, int] = field(default_factory=dict)
    idx_to_vertex: List[Vertex] = field(default_factory=list)
    vertices_tensor: Optional[torch.Tensor] = None
    indptr: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None

    def add_edge(self, a: Vertex, b: Vertex):
        """Add an undirected edge between vertices a and b."""
        if a == b:
            return
        self.vertices.add(a)
        self.vertices.add(b)
        self.adj[a].add(b)
        self.adj[b].add(a)

    def has_vertex(self, v: Vertex) -> bool:
        """Check if vertex v exists in this level."""
        return v in self.vertices

    def neighbors(self, v: Vertex) -> Set[Vertex]:
        """Return the neighbors of vertex v at this level."""
        return self.adj.get(v, set())


@dataclass
class Triangle:
    level: int
    v0: Vertex
    v1: Vertex
    v2: Vertex


class SierpinskiWorld:
    """Fractal-first Sierpinski space builder.
       - Builds triangles intrinsically by recursive corner subdivision (no masks).
       - For each level L, builds GraphLevel with vertices & edges of all level-L small triangles.
       - Provides movement constraints across scale (gateway vertices only; |ΔL| ≤ 1).
    """

    def __init__(self, max_level: int):
        if max_level < 0:
            raise ValueError("max_level must be >= 0")
        self.max_level = max_level
        # denominator for integer coordinates (all vertices land on this grid)
        self.den = 1 << max_level  # 2^N
        # Registries
        self.triangles_by_level: Dict[int, List[Triangle]] = {l: [] for l in range(max_level + 1)}
        self.graphs: Dict[int, GraphLevel] = {l: GraphLevel(l) for l in range(max_level + 1)}
        # Build the fractal
        self._build()

    # --- Geometry helpers ---
    @staticmethod
    def _mid(a: Vertex, b: Vertex) -> Vertex:
        return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

    @staticmethod
    def to_eq(v: Vertex) -> Tuple[float, float]:
        """Map integer lattice (x,y) to equilateral-plane (X,Y).
        X = x + 0.5*y;  Y = (√3/2)*y
        """
        x, y = v
        X = x + 0.5 * y
        Y = (3 ** 0.5) * 0.5 * y
        return (float(X), float(Y))

    def _add_triangle(self, level: int, v0: Vertex, v1: Vertex, v2: Vertex):
        t = Triangle(level, v0, v1, v2)
        self.triangles_by_level[level].append(t)
        # Edges define adjacency at this level
        g = self.graphs[level]
        g.add_edge(v0, v1)
        g.add_edge(v1, v2)
        g.add_edge(v2, v0)

    def _subdivide_corners(self, level: int, v0: Vertex, v1: Vertex, v2: Vertex):
        """Produce the 3 corner children (omit the central triangle)."""
        m01 = self._mid(v0, v1)
        m12 = self._mid(v1, v2)
        m20 = self._mid(v2, v0)
        # 3 corner children
        return (
            (v0, m01, m20),
            (m01, v1, m12),
            (m20, m12, v2),
        )

    def _build(self):
        # root triangle in integer coordinates on denominator den = 2^N
        v0 = (0, 0)
        v1 = (self.den, 0)
        v2 = (0, self.den)
        # Recursive construction across all levels
        def rec(level: int, a: Vertex, b: Vertex, c: Vertex):
            # register triangle and its edges at THIS level
            self._add_triangle(level, a, b, c)
            if level < self.max_level:
                for child in self._subdivide_corners(level + 1, a, b, c):
                    rec(level + 1, *child)
        rec(0, v0, v1, v2)
        # Ensure adjacency dicts have entries for all vertices present
        for L in range(self.max_level + 1):
            g = self.graphs[L]
            for t in self.triangles_by_level[L]:
                g.vertices.add(t.v0)
                g.vertices.add(t.v1)
                g.vertices.add(t.v2)
                g.adj.setdefault(t.v0, set())
                g.adj.setdefault(t.v1, set())
                g.adj.setdefault(t.v2, set())
        # Build GPU adjacency arrays for each level
        for L in range(self.max_level + 1):
            g = self.graphs[L]
            g.vertex_to_idx = {}
            g.idx_to_vertex = []
            sorted_vertices = sorted(g.vertices)
            for i, v in enumerate(sorted_vertices):
                g.vertex_to_idx[v] = i
                g.idx_to_vertex.append(v)
            indptr_list = [0]
            indices_list: List[int] = []
            for i, v in enumerate(g.idx_to_vertex):
                neighbors = g.adj.get(v, set())
                idx_list = [g.vertex_to_idx[nv] for nv in neighbors]
                indices_list.extend(idx_list)
                indptr_list.append(indptr_list[-1] + len(idx_list))
            if g.idx_to_vertex:
                coords = [self.to_eq(v) for v in g.idx_to_vertex]
                g.vertices_tensor = torch.tensor(coords, device='cuda', dtype=torch.float32)
                g.indptr = torch.tensor(indptr_list, device='cuda', dtype=torch.int32)
                g.indices = torch.tensor(indices_list, device='cuda', dtype=torch.int32)
            else:
                g.vertices_tensor = torch.tensor([], device='cuda', dtype=torch.float32)
                g.indptr = torch.tensor([0], device='cuda', dtype=torch.int32)
                g.indices = torch.tensor([], device='cuda', dtype=torch.int32)

    # --- Movement layer ---
    def has_vertex_at_level(self, v: Vertex, level: int) -> bool:
        return self.graphs[level].has_vertex(v)

    def neighbors_at_level(self, v: Vertex, level: int) -> Set[Vertex]:
        return self.graphs[level].neighbors(v)

    def levels(self) -> int:
        return self.max_level + 1

    # Utilities
    def expected_vertex_count(self, L: int) -> int:
        # Vertex count for the standard Sierpinski graph approximation (iteration L).
        # Known closed form: V(L) = (3^{L+1} + 3) / 2
        return (3 ** (L + 1) + 3) // 2

def _edge_angle(v_from, v_to):
    X1, Y1 = SierpinskiWorld.to_eq(v_from)
    X2, Y2 = SierpinskiWorld.to_eq(v_to)
    dx = X2 - X1
    dy = Y2 - Y1
    import math
    return math.atan2(dy, dx)




@dataclass
class Particle:
    id: int
    vertex: Vertex
    level: int  # motion scale (0..max_level)
    species_id: int = 0
    energy: float = 1.0
    # Simple particle charge used for attraction/repulsion. Positive values attract
    # negative charges and repel other positive charges.  Default is +1.
    charge: int = 1
    theta: float = 0.0
    bound_to: int = -1
    bind_timer: int = 0
    bind_cooldown: int = 0  # frames remaining during which this particle cannot rebind

    def state(self) -> Tuple[Vertex, int]:
        return (self.vertex, self.level)


class Engine:
    """Stepwise movement with strict Sierpinski rules using GPU-based data structures."""
    def __init__(self, world: SierpinskiWorld):
        self.world = world

    def step_particles(self, particles: List[Particle], bias: Optional[Dict[str, float]] = None, **kwargs):
        """Update multiple particles using GPU adjacency arrays and optional biases.

        This variant supports two interaction modes:
        (A) legacy charge-based attraction/repulsion (default), and
        (B) multi-species interaction matrices (Particle-Life style).

        Charge mode (legacy): each particle has integer `charge` (±1). For
        lateral moves, the move weight is multiplied by `attract_factor` for
        each adjacent particle of opposite charge and by `repel_factor` for
        each adjacent particle of the same charge. Moves that would place the
        particle on an already-occupied vertex at the same level are skipped.

        Matrix mode (new): supply `interaction_matrix` (an N×N numeric list)
        via kwargs and set `interaction_mode='matrix'` (or omit to auto-detect).
        Each particle has `species_id ∈ {0..N-1}`. For each adjacent particle
        of species `j`, a multiplicative factor `(1.0 + K[i][j])` is applied to
        the candidate move weight of species `i`; factors are clamped at ≥0.01
        to avoid zeroing. This conservatively extends the existing weighting
        scheme without altering pathing or occupancy logic.
        """
        if not particles:
            return
        from collections import defaultdict
        # Track which vertices are occupied at each level
        occupied_by_level: Dict[int, Set[Vertex]] = defaultdict(set)
        # Track charge at each occupied vertex
        charge_by_level: Dict[int, Dict[Vertex, int]] = defaultdict(dict)
        for p in particles:
            occupied_by_level[p.level].add(p.vertex)
            charge_by_level[p.level][p.vertex] = getattr(p, 'charge', 1)
        # Track species at each occupied vertex (multi-species mode)
        species_by_level: Dict[int, Dict[Vertex, int]] = defaultdict(dict)
        for p in particles:
            species_by_level[p.level][p.vertex] = getattr(p, 'species_id', 0)
        # Group particles by their current level
        level_groups: Dict[int, List[Particle]] = defaultdict(list)
        for p in particles:
            level_groups[p.level].append(p)
        # Interaction mode: 'charge' (legacy) or 'matrix' (if interaction_matrix provided)
        interaction_mode = kwargs.get('interaction_mode', 'auto')
        K = kwargs.get('interaction_matrix', None)
        use_matrix = (interaction_mode == 'matrix') or (interaction_mode == 'auto' and K is not None)
        # Normalize K to a Python list of lists if provided; no hard failure if malformed
        if use_matrix:
            try:
                # Shallow-validate shape
                if isinstance(K, (list, tuple)) and len(K) > 0 and isinstance(K[0], (list, tuple)):
                    # Ensure numeric and rectangular (best-effort)
                    ncols = max(len(row) for row in K)
                    K_norm: List[List[float]] = []
                    for row in K:
                        row_f = [float(x) for x in row]
                        if len(row_f) < ncols:
                            row_f = row_f + [0.0] * (ncols - len(row_f))
                        K_norm.append(row_f)
                    K = K_norm
                else:
                    use_matrix = False
            except Exception:
                use_matrix = False
        # --- Binding controls (safe defaults; legacy behavior preserved) ---
        p_bind = float(kwargs.get('p_bind', 0.05))
        p_unbind_base = float(kwargs.get('p_unbind_base', 0.01))
        bind_tau = int(kwargs.get('bind_tau', 200))
        bind_same_charge = bool(kwargs.get('bind_same_charge', False))
        bind_need_alignment = bool(kwargs.get('bind_need_alignment', True))
        co_move_bias = float(kwargs.get('co_move_bias', 1.0))
        # New: cooldown/max duration and matrix scaling for binding
        bind_cooldown_frames = int(kwargs.get('bind_cooldown_frames', 45))
        max_bind_frames = int(kwargs.get('max_bind_frames', 150))
        matrix_bind_scale = float(kwargs.get('matrix_bind_scale', 1.0))
        # Optional: normalize lateral anisotropy per-step to remove net drift
        aniso_center = bool(kwargs.get('aniso_center', True))  # normalize lateral anisotropy per-step
        # Constants controlling attraction/repulsion; adjust as desired
        attract_factor = 1.5
        repel_factor = 0.75
        # Update particles level by level
        for L, plist in level_groups.items():
            # Shuffle to avoid systematic bias in processing order
            random.shuffle(plist)
            g = self.world.graphs[L]
            for p in plist:
                idx = g.vertex_to_idx.get(p.vertex, None)
                if idx is None:
                    continue
                # Housekeeping: advance bind timers and cooldowns
                if getattr(p, 'bind_cooldown', 0) > 0:
                    p.bind_cooldown -= 1
                if getattr(p, 'bound_to', -1) != -1:
                    p.bind_timer += 1
                moves: List[Tuple[int, int]] = []
                weights: List[float] = []
                lat_indices: List[int] = []  # track indices of lateral candidates for normalization
                # Lateral moves: iterate neighbors at the same level
                start = int(g.indptr[idx].item())
                end = int(g.indptr[idx + 1].item())
                for ni in g.indices[start:end].tolist():
                    target_vertex = g.idx_to_vertex[ni]
                    # Skip if target is occupied by another particle at this level (except self)
                    if target_vertex != p.vertex and target_vertex in occupied_by_level[L]:
                        continue
                    # Base weight from bias
                    base_w = (bias or {}).get('lat', 1.0)
                    factor = 1.0
                    if use_matrix:
                        # Multi-species matrix weighting
                        si = getattr(p, 'species_id', 0)
                        for nb_vertex in g.adj.get(target_vertex, set()):
                            if nb_vertex in species_by_level[L]:
                                sj = species_by_level[L][nb_vertex]
                                try:
                                    delta = K[si][sj]
                                except Exception:
                                    delta = 0.0
                                # Conservative multiplicative weight; clamp to avoid zeroing
                                factor *= max(0.01, 1.0 + float(delta))
                    else:
                        # Legacy charge-based weighting
                        for nb_vertex in g.adj.get(target_vertex, set()):
                            if nb_vertex in charge_by_level[L]:
                                if charge_by_level[L][nb_vertex] == p.charge:
                                    factor *= repel_factor
                                else:
                                    factor *= attract_factor
                    # --- Anisotropic weighting (polarity) ---
                    try:
                        ang = _edge_angle(p.vertex, target_vertex)
                        import math
                        align_move = 1.0 + kwargs.get('alpha', 0.0) * math.cos((getattr(p, 'theta', 0.0)) - ang)
                    except Exception:
                        align_move = 1.0
                    w_aniso = base_w * factor * align_move
                    moves.append((ni, L))
                    weights.append(w_aniso)
                    lat_indices.append(len(weights) - 1)

                # Up move: to coarser level if this vertex exists there and no particle occupies it
                if L > 0 and p.vertex in self.world.graphs[L - 1].vertex_to_idx:
                    parent_idx = self.world.graphs[L - 1].vertex_to_idx[p.vertex]
                    parent_vertex = self.world.graphs[L - 1].idx_to_vertex[parent_idx]
                    if parent_vertex not in occupied_by_level[L - 1]:
                        moves.append((parent_idx, L - 1))
                        weights.append((bias or {}).get('up', 1.0))
                # --- Soft co-move bias if bound ---
                if getattr(p, 'bound_to', -1) != -1:
                    try:
                        for k, (cand_idx, cand_L) in enumerate(moves):
                            if cand_L == L:
                                weights[k] *= co_move_bias
                    except Exception:
                        pass
                # Down move: to finer level if this vertex persists there and no particle occupies it
                if L < self.world.max_level and p.vertex in self.world.graphs[L + 1].vertex_to_idx:
                    child_idx = self.world.graphs[L + 1].vertex_to_idx[p.vertex]
                    child_vertex = self.world.graphs[L + 1].idx_to_vertex[child_idx]
                    if child_vertex not in occupied_by_level[L + 1]:
                        moves.append((child_idx, L + 1))
                        weights.append((bias or {}).get('down', 1.0))
                # Normalize lateral anisotropy to remove net directional drift
                if aniso_center and lat_indices:
                    mean_lat = sum(weights[i] for i in lat_indices) / float(len(lat_indices))
                    if mean_lat > 0.0:
                        for i in lat_indices:
                            weights[i] /= mean_lat
                if not moves:
                    continue
                # Sample a move using GPU
                w = torch.tensor(weights, device='cuda', dtype=torch.float32)
                if w.sum().item() == 0.0:
                    probs = torch.full_like(w, 1.0 / len(w))
                else:
                    probs = w / w.sum()
                choice = int(torch.multinomial(probs, 1).item())
                new_idx, new_L = moves[choice]
                # Remove old occupancy, charge, and species
                occupied_by_level[p.level].discard(p.vertex)
                charge_by_level[p.level].pop(p.vertex, None)
                species_by_level[p.level].pop(p.vertex, None)
                # Update particle position and level
                p.level = new_L
                p.vertex = self.world.graphs[new_L].idx_to_vertex[new_idx]
                # Record new occupancy, charge, and species
                occupied_by_level[p.level].add(p.vertex)
                charge_by_level[p.level][p.vertex] = p.charge
                species_by_level[p.level][p.vertex] = getattr(p, 'species_id', 0)

            # --- Post-move binding/unbinding pass at this level (conservative) ---
            # Build a map from vertex -> GLOBAL particle id for current level L
            id_at_vertex: Dict[Vertex, int] = {}
            for pp in plist:
                id_at_vertex[pp.vertex] = pp.id

            import math, random as _rnd

            # 1) Enforce max bind duration and probabilistic unbinding
            for i, pp in enumerate(plist):
                j = getattr(pp, 'bound_to', -1)
                if j != -1:
                    # Hard cap on duration
                    if getattr(pp, 'bind_timer', 0) >= max_bind_frames:
                        # Break the link and set cooldowns on both
                        jj = j
                        pp.bound_to = -1
                        pp.bind_timer = 0
                        pp.bind_cooldown = bind_cooldown_frames
                        if 0 <= jj < len(particles):
                            qq = particles[jj]
                            qq.bound_to = -1
                            qq.bind_timer = 0
                            qq.bind_cooldown = bind_cooldown_frames
                        continue
                    # Probabilistic unbinding (legacy style)
                    # Increasing probability with time: p_unbind_base + bind_timer/bind_tau
                    p_break = p_unbind_base + (getattr(pp, 'bind_timer', 0) / max(1, bind_tau))
                    if _rnd.random() < max(0.0, min(1.0, p_break)):
                        jj = j
                        pp.bound_to = -1
                        pp.bind_timer = 0
                        pp.bind_cooldown = bind_cooldown_frames
                        if 0 <= jj < len(particles):
                            qq = particles[jj]
                            qq.bound_to = -1
                            qq.bind_timer = 0
                            qq.bind_cooldown = bind_cooldown_frames

            # 2) Attempt new bindings (only consider currently unbound particles)
            for pp in plist:
                if getattr(pp, 'bound_to', -1) != -1:
                    continue
                if getattr(pp, 'bind_cooldown', 0) > 0:
                    continue
                # Explore neighbors at the same level
                neighs = self.world.graphs[L].adj.get(pp.vertex, set())
                # Shuffle to avoid bias
                neighs = list(neighs)
                _rnd.shuffle(neighs)
                for nv in neighs:
                    if nv not in id_at_vertex:
                        continue
                    partner_gid = id_at_vertex[nv]
                    if partner_gid == pp.id:
                        continue
                    qq = particles[partner_gid]
                    if getattr(qq, 'bound_to', -1) != -1:
                        continue
                    if getattr(qq, 'bind_cooldown', 0) > 0:
                        continue
                    # Same-charge policy
                    if not bind_same_charge and getattr(pp, 'charge', 1) == getattr(qq, 'charge', 1):
                        continue
                    # Alignment policy (use edge angle from pp to qq)
                    if bind_need_alignment:
                        try:
                            ang = _edge_angle(pp.vertex, qq.vertex)
                            al = math.cos((getattr(pp, 'theta', 0.0)) - ang)
                            if al < 0.3:
                                continue
                        except Exception:
                            pass
                    # Effective bind probability
                    eff_p_bind = p_bind
                    if use_matrix and K is not None:
                        si = int(getattr(pp, 'species_id', 0))
                        sj = int(getattr(qq, 'species_id', 0))
                        try:
                            kij = float(K[si][sj])
                        except Exception:
                            kij = 0.0
                        eff_p_bind *= max(0.1, 1.0 + kij * matrix_bind_scale)
                    # Draw once (only allow monotonically ordered ids to avoid double-binds in this pass)
                    if pp.id < partner_gid and _rnd.random() < max(0.0, min(1.0, eff_p_bind)):
                        pp.bound_to = partner_gid
                        qq.bound_to = pp.id
                        pp.bind_timer = 0
                        qq.bind_timer = 0
                        break  # stop searching neighbors for pp after successful bind

    def step(self, p: Particle, rng: random.Random, bias: Optional[Dict[str, float]] = None):
        """One step using GPU adjacency and bias (ignores rng)."""
        self.step_particles([p], bias=bias)

    def shortest_path(self, start: Tuple[Vertex, int], goal: Tuple[Vertex, int]) -> Optional[List[Tuple[Vertex, int]]]:
        (vs, Ls) = start
        (vg, Lg) = goal
        # Find lowest common ancestor level where both vertices exist
        minL = min(Ls, Lg)
        LCA = minL
        while LCA >= 0:
            unit = 1 << (self.world.max_level - LCA)
            if vs[0] % unit == 0 and vs[1] % unit == 0 and vg[0] % unit == 0 and vg[1] % unit == 0:
                break
            LCA -= 1
        if LCA < 0:
            return None
        # Build vertical path from start to LCA (coarsen)
        path_up_start: List[Tuple[Vertex, int]] = []
        current_vertex = vs
        current_level = Ls
        while current_level > LCA:
            path_up_start.append((current_vertex, current_level))
            current_level -= 1
        # Build vertical path from LCA to goal (refine)
        path_down_goal: List[Tuple[Vertex, int]] = []
        current_vertex_goal = vg
        current_level_goal = Lg
        down_list: List[Tuple[Vertex, int]] = []
        while current_level_goal > LCA:
            down_list.append((current_vertex_goal, current_level_goal))
            current_level_goal -= 1
        path_down_goal = list(reversed(down_list))
        # Lateral BFS at LCA-level using cuGraph
        g = self.world.graphs[LCA]
        # Build the edge list for this level
        src: List[int] = []
        dst: List[int] = []
        for i, v in enumerate(g.idx_to_vertex):
            start_idx = int(g.indptr[i].item())
            end_idx = int(g.indptr[i + 1].item())
            neighbors = g.indices[start_idx:end_idx].tolist()
            for nb_idx in neighbors:
                src.append(i)
                dst.append(nb_idx)
        if not src:
            return None
        # Build a GPU DataFrame for the edge list.  cudf.DataFrame accepts a single
        # dtype for all columns, so we construct the DataFrame first and then
        # explicitly cast each column to int32.  This avoids the TypeError
        # encountered when passing a list of dtypes directly to the constructor.
        edges_df = cudf.DataFrame({"src": src, "dst": dst})
        edges_df = edges_df.astype({"src": "int32", "dst": "int32"})
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges_df, source="src", destination="dst", renumber=False)
        # Ensure start and goal exist at LCA level
        if vs not in g.vertex_to_idx or vg not in g.vertex_to_idx:
            return None
        start_idx = g.vertex_to_idx[vs]
        goal_idx = g.vertex_to_idx[vg]
        # Use breadth-first search for unweighted graph instead of SSSP
        # BFS returns a DataFrame with columns: 'vertex', 'distance', 'predecessor'
        result_df = cugraph.bfs(G, start=start_idx)
        # Set index to 'vertex' for easy lookups
        if 'vertex' in result_df.columns:
            result_df = result_df.set_index('vertex')
        try:
            distance = float(result_df.loc[goal_idx, "distance"])
        except Exception:
            return None
        if distance == float("inf") or distance != distance:
            return None
        # Reconstruct path using predecessor information from BFS
        path_indices: List[int] = []
        current = goal_idx
        try:
            predecessor = int(result_df.loc[current, "predecessor"])
        except Exception:
            predecessor = -1
        while current != start_idx and predecessor != -1:
            path_indices.append(current)
            current = predecessor
            try:
                predecessor = int(result_df.loc[current, "predecessor"])
            except Exception:
                predecessor = -1
        path_indices.append(start_idx)
        path_indices.reverse()
        lateral_path: List[Tuple[Vertex, int]] = [(g.idx_to_vertex[i], LCA) for i in path_indices]
        return path_up_start + lateral_path + path_down_goal