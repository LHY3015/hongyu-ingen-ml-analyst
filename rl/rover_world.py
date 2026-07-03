"""
Aido Rover world core

Shared stepable physics core. world.step(action) -> 10 sensor channels + label + info, driven by one shared latent world
(2D map + path-planned pose). Imported by the W2 preprocessing / sequence / transition-table
pipelines and the W5-6 RL environments, so offline data and the online env share identical
dynamics. 
"""

import numpy as np

SEED = 42

# ---- map-generation knobs ----
SITE = 200.0             # site side length (m)
MARGIN = 20.0            # inset of the patrol lanes from the boundary
N_SWEEPS = 4             # horizontal coverage sweeps
N_OBSTACLES = 8          # static obstacles (placed clear of every route/reroute edge)
OBST_CLEAR = 4.0         # min clearance from any edge to an obstacle
TERRAIN_WEIGHTS = [0.30, 0.34, 0.20, 0.16]   # P(mud, wet_grass, dry_grass, gravel) per patch

# ---- physical anchors ----
DT        = 0.1          # s, 10 Hz sensor cadence
V_CRUISE  = 1.0          # m/s, patrol band 0.8-1.2
MASS      = 395.0        # kg
G         = 9.81
WHEEL_R   = 0.20         # m, hub-motor wheel radius
N_WHEELS  = 4
GPS_SIGMA = 0.02         # m, RTK +-2cm
BATT_WH   = 9600.0       # Wh, 48V * 200Ah LiFePO4
P_IDLE    = 1200.0       # W, sensor+compute baseline (=> ~0.38 kWh/km @1 m/s)
TAU_IDLE  = 2.0          # Nm, baseline torque per wheel
C_TAU     = 6.0          # W per Nm, motor current proxy (sum-torque -> electrical power)
ALPHA_BATT= 8.0          # compressed-time discharge factor (NOT a real rate)
LAT0, LON0= 1.3500, 103.8000     # Singapore pilot origin
M_PER_DEG_LAT = 111320.0

# LiDAR feature params
LIDAR_MAX = 200.0        # m, max range of the LiDAR (no-return -> max-range spike)
LIDAR_FOV = np.deg2rad(120.0)
LIDAR_RAYS= 120
LIDAR_SIGMA0 = 0.03      # m, near-range noise floor
LIDAR_K      = 0.0015    # m/m, range-dependent noise growth
AR_PHI       = 0.8       # AR(1) coefficient on measurement noise
DROPOUT_P    = 0.02      # no-return / max-range spike probability

# torque deviation magnitudes (Nm), scaled by event severity
SPIKE_SLIP  = 20.0       # spinning wheel torque surge
SHED_SLIP   = 8.0        # load shed by the other wheels
RISE_STUCK  = 25.0       # all-wheel sustained rise
FAULT_COOLDOWN = 15      # refractory steps after a fault (traction recovers; no back-to-back chaining)
HALT_CLEARANCE = 1.0     # m, stop this far before a temporary blockage

# terrain: k_terrain (rolling-resistance proxy, AMDC 0.02-0.25) + slip weight
TERRAIN = {
    "asphalt":   dict(k=0.02, slip=0.00),
    "gravel":    dict(k=0.08, slip=0.10),
    "dry_grass": dict(k=0.12, slip=0.30),
    "wet_grass": dict(k=0.18, slip=0.70),
    "mud":       dict(k=0.25, slip=1.00),
}


def seg_pt_dist(A, B, P):
    ax, ay = A; bx, by = B; px, py = P
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


def _make_patch(rng, name, cx, cy):
    shape = ["circle", "rect", "poly"][int(rng.integers(0, 3))]   # mixed shapes
    r = float(rng.uniform(6, 11))                                  # bounded -> short crossings
    if shape == "circle":
        return dict(shape="circle", name=name, c=(cx, cy), r=r)
    if shape == "rect":
        return dict(shape="rect", name=name, x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r)
    a = float(rng.uniform(0, 2 * np.pi))
    pts = [(cx + r * np.cos(a + k * 2 * np.pi / 3), cy + r * np.sin(a + k * 2 * np.pi / 3))
           for k in range(3)]
    return dict(shape="poly", name=name, pts=pts)


def build_map(seed=0):
    """Procedurally generate one fixed map from `seed`: boustrophedon main cycle,
    a rejoining detour on each sweep, terrain on both main & branch edges, static
    obstacles placed clear of every route/reroute edge."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(MARGIN, SITE - MARGIN, N_SWEEPS)     # horizontal sweep y-levels
    xL, xR = MARGIN, SITE - MARGIN
    nodes, order, nid = {}, [], 0
    for i, y in enumerate(ys):                             # boustrophedon node order
        ends = [(xL, y), (xR, y)] if i % 2 == 0 else [(xR, y), (xL, y)]
        for p in ends:
            nodes[nid] = (float(p[0]), float(p[1])); order.append(nid); nid += 1
    route = {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}
    reroute = dict(route)

    branches = []                                          # one detour per sweep edge
    for i in range(N_SWEEPS):
        A, B = order[2 * i], order[2 * i + 1]
        ax, ay = nodes[A]; ex = nodes[B][0] - ax
        perp = 1.0 if ay < SITE / 2 else -1.0             # bow toward the site interior
        off = float(rng.uniform(15, 25))
        fa, fr = float(rng.uniform(0.30, 0.40)), float(rng.uniform(0.60, 0.72))
        aid, rid = nid, nid + 1; nid += 2
        nodes[aid] = (ax + fa * ex, ay + perp * off)      # apex
        nodes[rid] = (ax + fr * ex, ay)                   # rejoin (back on the line, past the apex)
        route[aid], route[rid] = rid, B
        reroute[aid], reroute[rid], reroute[A] = rid, B, aid
        branches.append((A, B, aid, rid))

    c = [(0, 0), (SITE, 0), (SITE, SITE), (0, SITE)]
    walls = [(c[j], c[(j + 1) % 4]) for j in range(4)]
    edges = {(a, b) for g in (route, reroute) for a, b in g.items()}

    obstacles, tries = [], 0                               # rejection-accept: clear of all edges
    while len(obstacles) < N_OBSTACLES and tries < 2000:
        tries += 1
        px, py = rng.uniform(MARGIN, SITE - MARGIN), rng.uniform(MARGIN, SITE - MARGIN)
        r = float(rng.uniform(6, 12))
        if all(seg_pt_dist(nodes[a], nodes[b], (px, py)) >= r + OBST_CLEAR for a, b in edges):
            obstacles.append((float(px), float(py), r))

    names = ["mud", "wet_grass", "dry_grass", "gravel"]   # terrain on main AND branch edges
    sev = {"mud": 0, "wet_grass": 1, "dry_grass": 2, "gravel": 3}
    zones = []
    for (a, b) in edges:
        ax, ay = nodes[a]; bx, by = nodes[b]
        for _ in range(int(rng.integers(1, 4))):          # 1-3 compact patches per edge
            f = float(rng.uniform(0.15, 0.85))
            name = str(rng.choice(names, p=TERRAIN_WEIGHTS))   # bias toward rougher terrain
            zones.append(_make_patch(rng, name, ax + f * (bx - ax), ay + f * (by - ay)))
    zones.sort(key=lambda z: sev[z["name"]])              # most-severe first (first match wins)
    return dict(nodes=nodes, route=route, reroute=reroute, branches=branches,
                obstacles=obstacles, walls=walls, zones=zones)


def _in_poly(x, y, pts):
    inside, n, j = False, len(pts), len(pts) - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def terrain_at(x, y, zones):
    for z in zones:
        s = z["shape"]
        if s == "rect" and z["x0"] <= x <= z["x1"] and z["y0"] <= y <= z["y1"]:
            return z["name"]
        if s == "circle" and (x - z["c"][0]) ** 2 + (y - z["c"][1]) ** 2 <= z["r"] ** 2:
            return z["name"]
        if s == "poly" and _in_poly(x, y, z["pts"]):
            return z["name"]
    return "asphalt"


def cast_min(px, py, heading, obstacles, walls):
    ang = heading + np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, LIDAR_RAYS)
    dx, dy = np.cos(ang), np.sin(ang)
    best = np.full(LIDAR_RAYS, LIDAR_MAX)
    # ray-circle
    for cx, cy, r in obstacles:
        fx, fy = px - cx, py - cy
        b = dx * fx + dy * fy
        cc = fx * fx + fy * fy - r * r
        disc = b * b - cc
        hit = disc >= 0
        t = -b - np.sqrt(np.where(hit, disc, 0.0))
        ok = hit & (t > 0)
        best = np.where(ok & (t < best), t, best)
    # ray-segment
    for (ax, ay), (bx, by) in walls:
        ex, ey = bx - ax, by - ay
        apx, apy = ax - px, ay - py
        den = dx * ey - dy * ex
        nz = np.abs(den) > 1e-12
        denom = np.where(nz, den, 1.0)
        t = (apx * ey - apy * ex) / denom
        u = (apx * dy - apy * dx) / denom
        ok = nz & (t > 0) & (u >= 0) & (u <= 1)
        best = np.where(ok & (t < best), t, best)
    return float(best.min())


class RoverWorld:

    def __init__(self, hazard_intensity=0.09, seed=SEED, total_steps=15000, map_seed=0,
                 blockages=False, p_block=0.6, p_full_block=0.3, block_radius=3.0):
        self.m = build_map(map_seed)
        self.hazard = hazard_intensity
        self.total_steps = total_steps
        # temporary-blockage config (default OFF -> W2 output unchanged)
        self.blockages_enabled = blockages
        self.p_block, self.p_full_block, self.block_radius = p_block, p_full_block, block_radius
        self.branch_nodes = [n for n in self.m["route"]
                             if self.m["reroute"][n] != self.m["route"][n]]
        n, L = 0, 0.0                          # main-cycle length: walk route from 0 back to 0
        while True:
            nx = self.m["route"][n]; L += self._elen(n, nx); n = nx
            if n == 0:
                break
        self.main_loop_len = L
        # per-wheel normal torque covariance: shared load -> positive correlation
        sig, rho = 0.5, 0.6
        self.Sigma = np.full((4, 4), rho * sig * sig)
        np.fill_diagonal(self.Sigma, sig * sig)
        self.reset(seed)

    def _elen(self, a, b):
        ax, ay = self.m["nodes"][a]
        bx, by = self.m["nodes"][b]
        return float(np.hypot(bx - ax, by - ay))

    def _blk_xy(self, edge):                    # blockage centre + radius -> LiDAR circle
        s, r = self.blockages[edge]
        a, b = edge
        ax, ay = self.m["nodes"][a]
        bx, by = self.m["nodes"][b]
        f = s / self._elen(a, b)
        return (ax + f * (bx - ax), ay + f * (by - ay), r)

    def _dist_to_blockage(self, route_map, lookahead=LIDAR_MAX):
        # walk forward from current pose along route_map; distance to first blockage else lookahead
        frm, to, start, dist = self.node, self.target, self.dist_into, 0.0
        for _ in range(20):
            blk = self.blockages.get((frm, to))
            if blk is not None and blk[0] >= start:
                return float(dist + blk[0] - start)
            dist += self._elen(frm, to) - start
            if dist >= lookahead:
                return float(lookahead)
            start, frm, to = 0.0, to, route_map[to]
        return float(lookahead)

    def _update_pose(self):
        ax, ay = self.m["nodes"][self.node]
        bx, by = self.m["nodes"][self.target]
        dx, dy = bx - ax, by - ay
        self.seg_len = np.hypot(dx, dy)
        f = self.dist_into / self.seg_len
        self.x, self.y = ax + f * dx, ay + f * dy
        self.heading = np.arctan2(dy, dx)

    def reset(self, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.node = 0                       # last node passed
        self.target = self.m["route"][0]    # node being driven toward
        self.dist_into = 0.0                # arc-length travelled along the current edge
        self._update_pose()
        self.soc = 100.0
        self.ar = 0.0
        self.fault_type = None
        self.fault_sev = 0.0
        self.fault_ttl = 0
        self.cooldown = 0
        self.cum_dist = 0.0
        self.halted = False
        self.blockages = {}                     # {(from,to): (s_along, radius)}
        if self.blockages_enabled:              # separate RNG stream -> does not perturb self.rng
            blk = np.random.default_rng(seed + 10007)
            for n in self.branch_nodes:
                if (n, self.m["route"][n]) == (self.node, self.target):
                    continue                    # can't reroute the edge we start already committed to
                if blk.random() < self.p_block:
                    e = (n, self.m["route"][n])
                    self.blockages[e] = (float(blk.uniform(0.25, 0.75) * self._elen(*e)),
                                         self.block_radius)
                    if blk.random() < self.p_full_block:      # also block the branch -> full block
                        eb = (n, self.m["reroute"][n])
                        self.blockages[eb] = (float(blk.uniform(0.25, 0.75) * self._elen(*eb)),
                                              self.block_radius)

    def step(self, action=0, dynamic_obstacles=()):
        # dynamic_obstacles: extra (cx,cy,r) circles seen by the LiDAR fan this step
        # (e.g. the other rover in W6 multi-agent); empty => single-agent, unchanged output
        m, rng = self.m, self.rng

        # (1) action -> speed factor & branch choice (no steering; motion is on the DAG)
        speed_factor = 0.5 if action == 1 else 1.0      # slow -> also de-risks the fault roll below
        reroute = (action == 2)                          # take the alternate branch at the next node
        v_cmd = V_CRUISE * speed_factor
        # raise-alert (3) / return-to-base (4): no motion effect here (W5 MDP wrapper)

        # (2) pose/heading are the current DAG edge (set by _update_pose; in-place turns at nodes)

        # (3) terrain at current pose
        terr = terrain_at(self.x, self.y, m["zones"])
        kt, slip_w = TERRAIN[terr]["k"], TERRAIN[terr]["slip"]

        # (4) fault state machine -> label (cooldown breaks the slip-trap feedback)
        if self.fault_ttl > 0:
            self.fault_ttl -= 1
            active = True
            if self.fault_ttl == 0:
                self.cooldown = FAULT_COOLDOWN   # event just ended -> refractory window
        else:
            active = False
            self.fault_type = None
            if self.cooldown > 0:
                self.cooldown -= 1
            elif rng.random() < self.hazard * slip_w * speed_factor:   # slow -> slip compensation lowers risk
                active = True
                if rng.random() < 0.2:
                    self.fault_type = "stuck"
                    self.fault_ttl = int(rng.integers(30, 80)) - 1
                else:
                    self.fault_type = "slip"
                    self.fault_ttl = int(rng.integers(10, 40)) - 1
                self.fault_sev = float(rng.beta(2, 5))
        label = 1 if active else 0
        sev = self.fault_sev if active else 0.0

        # (5) torque: terrain-demanded base + correlated noise + fault signature
        tau_demand = TAU_IDLE + kt * MASS * G * WHEEL_R / N_WHEELS
        tau = tau_demand + rng.multivariate_normal(np.zeros(4), self.Sigma)
        if active and self.fault_type == "slip":
            w = int(rng.integers(0, 4))
            tau[w] += SPIKE_SLIP * sev
            tau[np.arange(4) != w] -= SHED_SLIP * sev / 3.0
        elif active and self.fault_type == "stuck":
            tau += RISE_STUCK * sev
        tau = np.clip(tau, 0.0, None)

        # (6) advance along the edge by fault-reduced arc-length; (7) cross nodes, pick branch
        v_act = v_cmd
        if active and self.fault_type == "slip":
            v_act *= (1.0 - 0.7 * sev)
        elif active and self.fault_type == "stuck":
            v_act *= (1.0 - sev)
        ds = v_act * DT
        self.halted = False
        blk = self.blockages.get((self.node, self.target))
        if blk is not None and blk[0] > self.dist_into:  # blockage ahead on this edge -> can't cross
            max_reach = blk[0] - HALT_CLEARANCE
            ds = 0.0 if self.dist_into >= max_reach else min(ds, max_reach - self.dist_into)
            self.dist_into += ds
            self.halted = self.dist_into >= max_reach - 1e-9
        else:
            self.dist_into += ds
            while self.dist_into >= self.seg_len:        # reached/overshot the target node
                self.dist_into -= self.seg_len
                self.node = self.target
                self.target = (m["reroute"] if reroute else m["route"])[self.node]
                self.seg_len = self._elen(self.node, self.target)
        self.cum_dist += ds
        self._update_pose()

        # (8) battery: discharge coupled to torque (current proxy), compressed time
        P = P_IDLE + C_TAU * tau.sum()
        dWh = ALPHA_BATT * P * DT / 3600.0
        self.soc = max(0.0, self.soc - dWh / BATT_WH * 100.0)

        # (9) LiDAR: ray-cast min + range-dependent AR(1) noise + max-range dropout
        blk_obs = [self._blk_xy(e) for e in self.blockages]
        geo = cast_min(self.x, self.y, self.heading,
                       m["obstacles"] + list(dynamic_obstacles) + blk_obs, m["walls"])
        sigma = LIDAR_SIGMA0 + LIDAR_K * geo
        self.ar = AR_PHI * self.ar + np.sqrt(1 - AR_PHI ** 2) * rng.normal()
        meas = geo + sigma * self.ar
        if rng.random() < DROPOUT_P:
            meas = LIDAR_MAX
        lidar = float(np.clip(meas, 0.0, LIDAR_MAX))

        # (10) GPS (xy->latlon + RTK noise, multipath near structures) + ambient temp
        mp = 1.0 + 3.0 * np.exp(-geo / 10.0)
        gx = self.x + rng.normal(0, GPS_SIGMA * mp)
        gy = self.y + rng.normal(0, GPS_SIGMA * mp)
        lat = LAT0 + gy / M_PER_DEG_LAT
        lon = LON0 + gx / (M_PER_DEG_LAT * np.cos(np.deg2rad(LAT0)))
        temp = 28.0 + 3.0 * np.sin(2 * np.pi * self.t / self.total_steps) + rng.normal(0, 0.1)

        self.t += 1
        return dict(gps_lat=lat, gps_lon=lon, lidar_distance=lidar, battery_soc=self.soc,
                    torque_0=tau[0], torque_1=tau[1], torque_2=tau[2], torque_3=tau[3],
                    ambient_temp=temp, anomaly_label=label,
                    info=dict(terrain=terr, fault=self.fault_type, sev=sev,
                              x=self.x, y=self.y, geo=geo,
                              node=self.node, target=self.target, halted=self.halted,
                              route_progress=(self.cum_dist % self.main_loop_len) / self.main_loop_len,
                              next_main_block_dist=self._dist_to_blockage(m["route"]),
                              branch_block_dist=self._dist_to_blockage(m["reroute"])))


def realized_rate(hz, seed=SEED, n=15000, map_seed=0):
    w = RoverWorld(hazard_intensity=hz, seed=seed, total_steps=n, map_seed=map_seed)
    return np.mean([w.step(0)["anomaly_label"] for _ in range(n)])
