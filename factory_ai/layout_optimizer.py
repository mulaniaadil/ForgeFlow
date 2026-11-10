"""Facility-layout optimiser: simulated annealing + shortest-path (aisle) evaluation.

Conventions
-----------
* Metres, origin at the bottom-left corner of the floor.
* A rectangle is a tuple ``(x, y, w, h)`` with (x, y) the lower-left corner.
* A layout is ``{machine_name: (x, y, w, h)}`` (movable machines only).
* Docks (Receiving / Dispatch) and columns are fixed and live in :class:`Problem`.

Objective  : sum_{(i,j)} flow_ij * distance_ij  (unit-load-metres per shift)
Search     : SA on Manhattan centre-to-centre distance (fast), then optional hill-climb
             refinement on true shortest-path distance through the free-floor grid.
Constraints: inside floor (wall margin), no overlap, minimum clearance between machines /
             columns / docks (this reserves aisles), optional 90-degree rotation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9
BIG = 1e6


@dataclass
class Problem:
    floor_w: float
    floor_h: float
    machines: dict                       # name -> (w, h) nominal footprint
    flows: dict                          # (a, b) -> unit loads per shift; a, b machine or dock names
    fixed: dict = field(default_factory=dict)       # dock name -> (x, y, w, h)
    obstacles: list = field(default_factory=list)   # [(x, y, w, h)] columns etc.
    clearance: float = 1.0               # min gap between any two objects (aisle width)
    wall_margin: float = 0.5             # min gap between machines and walls
    grid: float = 0.5                    # placement / path-grid resolution (m)
    allow_rotation: bool = True

    def nodes_with_flow(self):
        s = set()
        for a, b in self.flows:
            s.add(a)
            s.add(b)
        return s


# ----------------------------------------------------------------------------- geometry
def center(r):
    return (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)


def separated(a, b, gap):
    """True if rectangles a and b are at least `gap` apart (axis-aligned)."""
    return (a[0] + a[2] + gap <= b[0] + EPS or b[0] + b[2] + gap <= a[0] + EPS or
            a[1] + a[3] + gap <= b[1] + EPS or b[1] + b[3] + gap <= a[1] + EPS)


def inside_floor(p: Problem, r):
    m = p.wall_margin
    return (r[0] >= m - EPS and r[1] >= m - EPS and
            r[0] + r[2] <= p.floor_w - m + EPS and r[1] + r[3] <= p.floor_h - m + EPS)


def _static_ok(p: Problem, r):
    if not inside_floor(p, r):
        return False
    for o in p.obstacles:
        if not separated(r, o, p.clearance):
            return False
    for d in p.fixed.values():
        if not separated(r, d, p.clearance):
            return False
    return True


def violations(p: Problem, layout):
    """List human-readable constraint violations (empty list == feasible)."""
    if not layout:
        return ["no layout provided"]
    out = []
    missing = set(p.machines) - set(layout)
    if missing:
        out.append(f"missing machines: {', '.join(sorted(missing))}")
    names = list(layout)
    for n in names:
        r = layout[n]
        if not inside_floor(p, r):
            out.append(f"{n}: outside floor / wall margin")
        for k, o in enumerate(p.obstacles):
            if not separated(r, o, p.clearance):
                out.append(f"{n}: clearance to column {k}")
        for dn, d in p.fixed.items():
            if not separated(r, d, p.clearance):
                out.append(f"{n}: clearance to {dn}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not separated(layout[names[i]], layout[names[j]], p.clearance):
                out.append(f"{names[i]} / {names[j]}: clearance")
    return out


def is_feasible(p: Problem, layout):
    return not violations(p, layout)


def node_center(p: Problem, layout, name):
    if name in layout:
        return center(layout[name])
    return center(p.fixed[name])


def cost_manhattan(p: Problem, layout):
    if not layout:
        return 0.0
    tot = 0.0
    for (a, b), f in p.flows.items():
        if (a in layout or a in p.fixed) and (b in layout or b in p.fixed):
            ax, ay = node_center(p, layout, a)
            bx, by = node_center(p, layout, b)
            tot += f * (abs(ax - bx) + abs(ay - by))
    return tot


# ----------------------------------------------------------------------------- random start
def random_feasible_layout(p: Problem, rng: random.Random, max_restarts=200, tries=400):
    g = p.grid
    order = sorted(p.machines, key=lambda n: -p.machines[n][0] * p.machines[n][1])
    for _ in range(max_restarts):
        layout = {}
        ok = True
        for n in order:
            w0, h0 = p.machines[n]
            placed = False
            for _ in range(tries):
                w, h = (h0, w0) if (p.allow_rotation and rng.random() < 0.5) else (w0, h0)
                lo_x, hi_x = math.ceil((p.wall_margin - EPS) / g), math.floor((p.floor_w - p.wall_margin - w + EPS) / g)
                lo_y, hi_y = math.ceil((p.wall_margin - EPS) / g), math.floor((p.floor_h - p.wall_margin - h + EPS) / g)
                if hi_x < lo_x or hi_y < lo_y:
                    continue
                r = (rng.randint(lo_x, hi_x) * g, rng.randint(lo_y, hi_y) * g, w, h)
                if _static_ok(p, r) and all(separated(r, o, p.clearance) for o in layout.values()):
                    layout[n] = r
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return {n: layout[n] for n in p.machines}
    raise RuntimeError("Could not find a feasible random layout (floor too small / clearance too large?)")


# ----------------------------------------------------------------------------- SA
def _move_ok(p: Problem, layout, moved: dict):
    for n, r in moved.items():
        if not _static_ok(p, r):
            return False
        for m, o in layout.items():
            if m in moved:
                continue
            if not separated(r, o, p.clearance):
                return False
    ms = list(moved.values())
    if len(ms) == 2 and not separated(ms[0], ms[1], p.clearance):
        return False
    return True


def _propose(p: Problem, layout, names, rng: random.Random):
    if not names:
        return None
    g = p.grid
    r = rng.random()
    if r < 0.40:                                   # relocate (possibly rotate)
        n = rng.choice(names)
        w0, h0 = p.machines[n]
        w, h = (h0, w0) if (p.allow_rotation and rng.random() < 0.5) else (w0, h0)
        lo_x, hi_x = math.ceil((p.wall_margin - EPS) / g), math.floor((p.floor_w - p.wall_margin - w + EPS) / g)
        lo_y, hi_y = math.ceil((p.wall_margin - EPS) / g), math.floor((p.floor_h - p.wall_margin - h + EPS) / g)
        if hi_x < lo_x or hi_y < lo_y:
            return None
        return {n: (rng.randint(lo_x, hi_x) * g, rng.randint(lo_y, hi_y) * g, w, h)}
    if r < 0.85:                                   # local shift
        n = rng.choice(names)
        x, y, w, h = layout[n]
        dx, dy = rng.randint(-4, 4) * g, rng.randint(-4, 4) * g
        if dx == 0 and dy == 0:
            return None
        return {n: (x + dx, y + dy, w, h)}
    if r < 0.93 and p.allow_rotation:              # rotate in place about the centre
        n = rng.choice(names)
        x, y, w, h = layout[n]
        cx, cy = x + w / 2, y + h / 2
        nx, ny = round((cx - h / 2) / g) * g, round((cy - w / 2) / g) * g
        return {n: (nx, ny, h, w)}
    if len(names) >= 2:                            # swap centres of two machines
        a, b = rng.sample(names, 2)
        (xa, ya, wa, ha), (xb, yb, wb, hb) = layout[a], layout[b]
        cax, cay, cbx, cby = xa + wa / 2, ya + ha / 2, xb + wb / 2, yb + hb / 2
        ra = (round((cbx - wa / 2) / g) * g, round((cby - ha / 2) / g) * g, wa, ha)
        rb = (round((cax - wb / 2) / g) * g, round((cay - hb / 2) / g) * g, wb, hb)
        return {a: ra, b: rb}
    return None


def simulated_annealing(p: Problem, start, iters=30000, seed=0, t0_frac=0.05, t_end_frac=1e-4):
    """Return (best_layout, best_cost, history) using Manhattan flow-distance."""
    rng = random.Random(seed)
    cur = dict(start)
    cur_cost = cost_manhattan(p, cur)
    best, best_cost = dict(cur), cur_cost
    names = list(cur)
    t0 = max(1.0, t0_frac * cur_cost)
    cooling = (t_end_frac) ** (1.0 / max(iters, 1))
    T = t0
    history = [best_cost]
    for it in range(iters):
        moved = _propose(p, cur, names, rng)
        if moved is not None and _move_ok(p, cur, moved):
            cand = dict(cur)
            cand.update(moved)
            c = cost_manhattan(p, cand)
            d = c - cur_cost
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                cur, cur_cost = cand, c
                if c < best_cost - 1e-9:
                    best, best_cost = dict(cand), c
        T *= cooling
        if (it + 1) % 200 == 0:
            history.append(best_cost)
    return best, best_cost, history


# ----------------------------------------------------------------------------- shortest path
def _cells(p: Problem, r):
    g = p.grid
    ix0, ix1 = int(math.floor(r[0] / g + 1e-6)), int(math.ceil((r[0] + r[2]) / g - 1e-6))
    iy0, iy1 = int(math.floor(r[1] / g + 1e-6)), int(math.ceil((r[1] + r[3]) / g - 1e-6))
    return ix0, ix1, iy0, iy1


def _mask(p: Problem, shape, r):
    m = np.zeros(shape, bool)
    ix0, ix1, iy0, iy1 = _cells(p, r)
    m[max(iy0, 0):max(iy1, 0), max(ix0, 0):max(ix1, 0)] = True
    return m


def _dilate4(m):
    o = m.copy()
    o[1:, :] |= m[:-1, :]
    o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    return o


def _bfs(free, sources):
    dist = np.full(free.shape, np.inf)
    dist[sources] = 0
    visited = sources.copy()
    frontier = sources.copy()
    d = 0
    while frontier.any():
        d += 1
        nb = np.zeros_like(frontier)
        nb[1:, :] |= frontier[:-1, :]
        nb[:-1, :] |= frontier[1:, :]
        nb[:, 1:] |= frontier[:, :-1]
        nb[:, :-1] |= frontier[:, 1:]
        nb &= free & ~visited
        dist[nb] = d
        visited |= nb
        frontier = nb
    return dist


def path_distances(p: Problem, layout):
    """Shortest aisle distances (m) between every pair of nodes that have a flow.

    Machines and columns block the grid.  A machine's "access cells" are the free cells
    edge-adjacent to its footprint; a dock's access cells are the free cells in its box.
    Distance = grid path length between access cells (cell-centre to cell-centre) plus an
    in-machine allowance of (w+h)/4 at each machine end (centre -> access face), so the value
    is comparable with the centre-to-centre Manhattan distance.
    Returns (dist_dict[(a,b)] , access, bfs_maps).
    """
    g = p.grid
    ny, nx = int(round(p.floor_h / g)), int(round(p.floor_w / g))
    shape = (ny, nx)
    blocked = np.zeros(shape, bool)
    mach_masks = {}
    for n, r in layout.items():
        mach_masks[n] = _mask(p, shape, r)
        blocked |= mach_masks[n]
    for o in p.obstacles:
        blocked |= _mask(p, shape, o)
    free = ~blocked
    access = {}
    for n, m in mach_masks.items():
        access[n] = _dilate4(m) & ~m & free
    for n, r in p.fixed.items():
        a = _mask(p, shape, r) & free
        if not a.any():
            a = _dilate4(_mask(p, shape, r)) & free
        access[n] = a
    nodes = [n for n in p.nodes_with_flow() if n in access]
    maps = {n: _bfs(free, access[n]) for n in nodes}
    off = {n: ((layout[n][2] + layout[n][3]) / 4.0 if n in layout else 0.0) for n in nodes}
    dist = {}
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            if not access[b].any() or not access[a].any():
                dist[(a, b)] = BIG
                continue
            d = maps[a][access[b]].min()
            dist[(a, b)] = BIG if not np.isfinite(d) else float(d) * g + off[a] + off[b]
    return dist, access, maps, free


def cost_path(p: Problem, layout):
    if not layout:
        return 0.0
    dist, *_ = path_distances(p, layout)
    tot = 0.0
    for (a, b), f in p.flows.items():
        if (a, b) in dist:
            tot += f * dist[(a, b)]
    return tot


def refine_with_path(p: Problem, layout, iters=300, seed=0):
    """Greedy hill-climb on true shortest-path cost using small shifts / rotations / swaps."""
    if not layout:
        return {}, 0.0
    rng = random.Random(seed)
    cur = dict(layout)
    cur_cost = cost_path(p, cur)
    names = list(cur)
    for _ in range(iters):
        moved = _propose(p, cur, names, rng)
        if moved is None or not _move_ok(p, cur, moved):
            continue
        cand = dict(cur)
        cand.update(moved)
        c = cost_path(p, cand)
        if c < cur_cost - 1e-9:
            cur, cur_cost = cand, c
    return cur, cur_cost


def trace_path(p: Problem, maps, access, a, b):
    """Cell path (list of (ix, iy)) from node a's access to node b's access using BFS map of a."""
    dmap = maps[a]
    tgt = np.argwhere(access[b])
    if len(tgt) == 0:
        return []
    vals = [dmap[iy, ix] for iy, ix in tgt]
    k = int(np.argmin(vals))
    if not np.isfinite(vals[k]):
        return []
    iy, ix = tgt[k]
    path = [(ix, iy)]
    while dmap[iy, ix] > 0:
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny_, nx_ = iy + dy, ix + dx
            if 0 <= ny_ < dmap.shape[0] and 0 <= nx_ < dmap.shape[1] and dmap[ny_, nx_] == dmap[iy, ix] - 1:
                iy, ix = ny_, nx_
                path.append((ix, iy))
                break
        else:
            break
    return path[::-1]


# ----------------------------------------------------------------------------- top-level
def optimize(p: Problem, initial=None, n_seeds=8, iters=30000, refine_iters=300, base_seed=0):
    """Multi-start SA (+ path refinement of the best).  Returns a result dict."""
    start_ok = initial is not None and is_feasible(p, initial)
    init_manh = cost_manhattan(p, initial) if initial else float("nan")
    init_path = cost_path(p, initial) if initial else float("nan")
    runs = []
    for s in range(max(1, n_seeds)):
        rng = random.Random(base_seed * 1000 + s)
        if s == 0 and start_ok:
            start = initial
        else:
            try:
                start = random_feasible_layout(p, rng)
            except RuntimeError:
                continue          # this restart found no feasible start; others may succeed
        best, c, hist = simulated_annealing(p, start, iters=iters, seed=base_seed * 1000 + s)
        runs.append(dict(seed=s, layout=best, cost_manhattan=c, history=hist,
                         cost_path=cost_path(p, best)))
    if not runs:
        raise RuntimeError("No feasible arrangement found - the floor is too small or the clearance too large")
    # choose the seed with the best true path cost (finite), tie -> manhattan
    best_run = min(runs, key=lambda r: (r["cost_path"], r["cost_manhattan"]))
    refined, refined_cost = (best_run["layout"], best_run["cost_path"])
    if refine_iters > 0:
        refined, refined_cost = refine_with_path(p, best_run["layout"], iters=refine_iters, seed=base_seed)
    return dict(
        initial_feasible=start_ok,
        initial_violations=violations(p, initial) if initial else [],
        initial_cost_manhattan=init_manh, initial_cost_path=init_path,
        runs=runs, best_sa_layout=best_run["layout"],
        best_sa_cost_manhattan=best_run["cost_manhattan"], best_sa_cost_path=best_run["cost_path"],
        final_layout=refined, final_cost_path=refined_cost,
        final_cost_manhattan=cost_manhattan(p, refined),
        final_feasible=is_feasible(p, refined),
    )
