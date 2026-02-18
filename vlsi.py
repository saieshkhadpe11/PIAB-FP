"""
================================================================================
Physics-Inspired Agent-Based Optimization for VLSI Floorplanning
================================================================================

Authors : [Your Name]
Affiliation : [Your Institution]
Date : February 2026

Description:
    This module implements a physics-inspired, agent-based optimization
    framework for VLSI macro-block floorplanning. Each functional block
    is modeled as an autonomous agent subject to four classes of forces:

        1. Attraction   – between electrically connected blocks
        2. Repulsion    – to resolve geometric overlaps
        3. Gravity      – thermal-aware placement (high-power → chip edge)
        4. Boundary     – to enforce die-area constraints

    The system evolves via damped Newtonian integration with adaptive
    force scheduling until a convergence criterion on total kinetic
    energy is satisfied.

References:
    [1] Shahookar & Mazumder, "VLSI Cell Placement Techniques," ACM
        Computing Surveys, 1991.
    [2] Caldwell, Kahng & Markov, "Can Recursive Bisection Alone Produce
        Routable Placements?", DAC 2000.
    [3] Eisenmann & Johannes, "Generic Global Placement and Floorplanning,"
        DAC 1998.

License : MIT
================================================================================
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Block:
    """Autonomous agent representing a VLSI macro-block.

    Attributes
    ----------
    name   : human-readable identifier (e.g. "ALU", "Cache")
    w, h   : width and height in layout units
    power  : power dissipation (W) — drives thermal-gravity force
    x, y   : centre-of-mass position (layout units)
    vx, vy : velocity components (units / iteration)
    """
    name  : str
    w     : float
    h     : float
    power : float
    x     : float = 0.0
    y     : float = 0.0
    vx    : float = 0.0
    vy    : float = 0.0


@dataclasses.dataclass
class Net:
    """Hyperedge connecting two or more blocks.

    Attributes
    ----------
    block_ids    : indices into the block list
    connectivity : edge weight (higher ⇒ stronger attraction)
    """
    block_ids    : List[int]
    connectivity : float = 1.0


# ──────────────────────────────────────────────────────────────────────
# 2. BENCHMARK GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_benchmark(
    n_blocks: int = 12,
    n_nets  : int = 18,
    seed    : int = 42,
) -> Tuple[List[Block], List[Net]]:
    """Create a reproducible synthetic benchmark.

    Block dimensions are drawn uniformly from [5, 18].
    Power values follow a log-normal distribution to mimic real chips
    where a few blocks (e.g. processor cores) consume significantly
    more power than peripheral logic.

    Net connectivity follows U(0.5, 3.0).
    """
    rng = np.random.default_rng(seed)

    # Realistic VLSI block names
    block_names = [
        "ALU", "FPU", "RegFile", "L1-I$", "L1-D$", "L2$",
        "BranchPred", "Decoder", "ROB", "LSQ", "TLB", "IOCtrl",
        "MemCtrl", "DMA", "PCIe", "NoC_Router", "PMU", "ClkGen",
        "PHY", "SerDes",
    ]

    blocks: List[Block] = []
    for i in range(n_blocks):
        w = rng.uniform(5, 18)
        h = rng.uniform(5, 18)
        power = float(np.clip(rng.lognormal(1.0, 0.8), 0.5, 25.0))
        name = block_names[i % len(block_names)]
        # Random initial position inside die  (well within bounds)
        x = rng.uniform(w / 2 + 2, 100 - w / 2 - 2)
        y = rng.uniform(h / 2 + 2, 100 - h / 2 - 2)
        blocks.append(Block(name=name, w=w, h=h, power=power, x=x, y=y))

    nets: List[Net] = []
    all_pairs = [(i, j) for i in range(n_blocks) for j in range(i + 1, n_blocks)]
    chosen = rng.choice(len(all_pairs), size=min(n_nets, len(all_pairs)), replace=False)
    for idx in chosen:
        i, j = all_pairs[idx]
        conn = float(rng.uniform(0.5, 3.0))
        nets.append(Net(block_ids=[i, j], connectivity=conn))

    return blocks, nets


# ──────────────────────────────────────────────────────────────────────
# 3. PHYSICS FORCE MODEL
# ──────────────────────────────────────────────────────────────────────

class PhysicsEngine:
    r"""Computes the four force contributions on every block.

    Mathematical formulation
    ------------------------
    Let block *i* have centre $(x_i, y_i)$, dimensions $(w_i, h_i)$,
    and power $P_i$.

    **Attraction (Hooke-spring model):**

    $$
    \mathbf{F}_{attr}^{(i)} = \sum_{j \in \mathcal{N}(i)} c_{ij}\,
        (\mathbf{r}_j - \mathbf{r}_i)
    $$

    **Repulsion (overlap penalty — soft-body):**

    For each pair $(i, j)$ whose bounding boxes overlap:

    $$
    \mathbf{F}_{rep}^{(i)} = -k_{rep}\,\mathrm{overlap\_area}(i,j)\,
        \hat{\mathbf{d}}_{ij}
    $$

    **Gravity (thermal-aware):**

    $$
    F_{grav,y}^{(i)} = -k_g \, P_i
    $$

    **Boundary (elastic wall):**

    $$
    \mathbf{F}_{bnd}^{(i)} = k_b \, \max(0,\, \delta)
    $$

    The system uses an **adaptive force schedule**: attraction strength
    ramps up slowly while repulsion stays strong throughout, ensuring
    overlap resolution has priority over wirelength minimisation.

    Parameters
    ----------
    chip_w, chip_h : die dimensions
    k_attr         : base attraction spring constant
    k_rep          : repulsion stiffness
    k_grav         : gravity coefficient
    k_bnd          : boundary spring constant
    damping        : velocity damping factor (∈ (0, 1])
    dt             : integration time-step
    v_max          : maximum velocity clamp (prevents oscillation)
    """

    def __init__(
        self,
        chip_w : float = 100.0,
        chip_h : float = 100.0,
        k_attr : float = 0.04,
        k_rep  : float = 1.2,
        k_grav : float = 0.08,
        k_bnd  : float = 2.0,
        damping: float = 0.65,
        dt     : float = 1.0,
        v_max  : float = 5.0,
    ):
        self.chip_w  = chip_w
        self.chip_h  = chip_h
        self.k_attr  = k_attr
        self.k_rep   = k_rep
        self.k_grav  = k_grav
        self.k_bnd   = k_bnd
        self.damping = damping
        self.dt      = dt
        self.v_max   = v_max

    # ── helper: overlap area between two blocks ───────────────────────
    @staticmethod
    def _overlap_area(a: Block, b: Block) -> float:
        """Axis-aligned bounding-box overlap area."""
        ox = max(0.0, min(a.x + a.w/2, b.x + b.w/2) - max(a.x - a.w/2, b.x - b.w/2))
        oy = max(0.0, min(a.y + a.h/2, b.y + b.h/2) - max(a.y - a.h/2, b.y - b.h/2))
        return ox * oy

    # ── force computation ─────────────────────────────────────────────
    def compute_forces(
        self,
        blocks: List[Block],
        nets  : List[Net],
        iteration: int = 0,
        max_iter : int = 500,
    ) -> np.ndarray:
        """Return (N, 2) force array for all blocks.

        Uses adaptive scheduling: attraction ramps up with iteration
        progress while repulsion remains constant.
        """
        n = len(blocks)
        F = np.zeros((n, 2), dtype=np.float64)

        # Adaptive scheduling factor: attraction grows from 0.3× to 1.0×
        progress = min(iteration / max(max_iter * 0.6, 1), 1.0)
        attr_scale = 0.3 + 0.7 * progress

        # 3a. Attraction ──────────────────────────────────────────────
        for net in nets:
            ids = net.block_ids
            for ii in range(len(ids)):
                for jj in range(ii + 1, len(ids)):
                    i, j = ids[ii], ids[jj]
                    dx = blocks[j].x - blocks[i].x
                    dy = blocks[j].y - blocks[i].y
                    fx = self.k_attr * attr_scale * net.connectivity * dx
                    fy = self.k_attr * attr_scale * net.connectivity * dy
                    F[i, 0] += fx;  F[i, 1] += fy
                    F[j, 0] -= fx;  F[j, 1] -= fy

        # 3b. Repulsion ───────────────────────────────────────────────
        for i in range(n):
            for j in range(i + 1, n):
                oa = self._overlap_area(blocks[i], blocks[j])
                if oa > 0:
                    dx = blocks[i].x - blocks[j].x
                    dy = blocks[i].y - blocks[j].y
                    dist = max(np.hypot(dx, dy), 0.5)

                    # Proportional + minimum-separation component
                    mag = self.k_rep * oa + self.k_rep * 2.0

                    F[i, 0] += mag * dx / dist
                    F[i, 1] += mag * dy / dist
                    F[j, 0] -= mag * dx / dist
                    F[j, 1] -= mag * dy / dist

        # 3c. Gravity (thermal-aware) ─────────────────────────────────
        for i, b in enumerate(blocks):
            F[i, 1] -= self.k_grav * b.power      # push toward y = 0

        # 3d. Boundary forces ─────────────────────────────────────────
        for i, b in enumerate(blocks):
            # left wall
            pen = (b.w / 2) - b.x
            if pen > 0:
                F[i, 0] += self.k_bnd * pen
            # right wall
            pen = (b.x + b.w / 2) - self.chip_w
            if pen > 0:
                F[i, 0] -= self.k_bnd * pen
            # bottom wall
            pen = (b.h / 2) - b.y
            if pen > 0:
                F[i, 1] += self.k_bnd * pen
            # top wall
            pen = (b.y + b.h / 2) - self.chip_h
            if pen > 0:
                F[i, 1] -= self.k_bnd * pen

        return F

    # ── clamp position to die area ────────────────────────────────────
    def clamp_to_die(self, b: Block):
        """Hard-clamp block position so it stays within the die."""
        b.x = max(b.w / 2, min(b.x, self.chip_w - b.w / 2))
        b.y = max(b.h / 2, min(b.y, self.chip_h - b.h / 2))

    # ── single integration step ───────────────────────────────────────
    def step(
        self,
        blocks: List[Block],
        nets  : List[Net],
        iteration: int = 0,
        max_iter : int = 500,
    ) -> float:
        """Advance one iteration.  Returns total kinetic energy."""
        F = self.compute_forces(blocks, nets, iteration, max_iter)
        ke = 0.0

        # Adaptive damping: increase damping as system settles
        progress = min(iteration / max(max_iter * 0.8, 1), 1.0)
        damp = self.damping * (1.0 - 0.3 * progress)  # 0.65 → ~0.46

        for i, b in enumerate(blocks):
            b.vx = (b.vx + F[i, 0] * self.dt) * damp
            b.vy = (b.vy + F[i, 1] * self.dt) * damp

            # Velocity clamping to prevent oscillation
            speed = np.hypot(b.vx, b.vy)
            if speed > self.v_max:
                scale = self.v_max / speed
                b.vx *= scale
                b.vy *= scale

            b.x += b.vx * self.dt
            b.y += b.vy * self.dt

            # Hard-clamp to die boundary
            self.clamp_to_die(b)

            ke += 0.5 * (b.vx**2 + b.vy**2)
        return ke


# ──────────────────────────────────────────────────────────────────────
# 4. METRICS
# ──────────────────────────────────────────────────────────────────────

def half_perimeter_wirelength(blocks: List[Block], nets: List[Net]) -> float:
    r"""Half-perimeter wirelength (HPWL).

    $$
      \mathrm{HPWL} = \sum_{e \in \mathcal{E}} c_e \bigl[
          (\max_i x_i - \min_i x_i) + (\max_i y_i - \min_i y_i)
      \bigr]
    $$
    """
    total = 0.0
    for net in nets:
        xs = [blocks[i].x for i in net.block_ids]
        ys = [blocks[i].y for i in net.block_ids]
        total += net.connectivity * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return total


def total_overlap(blocks: List[Block]) -> float:
    """Sum of pairwise bounding-box overlap areas."""
    n = len(blocks)
    ov = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ov += PhysicsEngine._overlap_area(blocks[i], blocks[j])
    return ov


def power_gradient_score(blocks: List[Block], chip_h: float = 100.0) -> float:
    """Pearson correlation between block power and distance from bottom.

    A strong negative correlation means high-power blocks successfully
    clustered near y = 0 (heat-sink side).
    """
    powers = np.array([b.power for b in blocks])
    ys     = np.array([b.y     for b in blocks])
    if np.std(powers) < 1e-8 or np.std(ys) < 1e-8:
        return 0.0
    return float(np.corrcoef(powers, ys)[0, 1])


# ──────────────────────────────────────────────────────────────────────
# 5. SIMULATION LOOP
# ──────────────────────────────────────────────────────────────────────

def run_simulation(
    blocks     : List[Block],
    nets       : List[Net],
    engine     : PhysicsEngine,
    max_iter   : int   = 500,
    ke_tol     : float = 0.01,
    record_every: int  = 2,
) -> dict:
    """Execute the physics simulation until convergence or max_iter.

    Returns a dictionary of per-iteration metrics and snapshot history.
    """
    history: dict = {
        "wirelength" : [],
        "overlap"    : [],
        "kinetic"    : [],
        "power_corr" : [],
        "snapshots"  : [],          # list of block-state lists for animation
    }

    converged_at = max_iter

    for it in range(1, max_iter + 1):
        ke = engine.step(blocks, nets, iteration=it, max_iter=max_iter)

        wl = half_perimeter_wirelength(blocks, nets)
        ov = total_overlap(blocks)
        pc = power_gradient_score(blocks, engine.chip_h)

        history["wirelength"].append(wl)
        history["overlap"].append(ov)
        history["kinetic"].append(ke)
        history["power_corr"].append(pc)

        if it % record_every == 0:
            snap = [(b.x, b.y, b.w, b.h, b.power, b.name) for b in blocks]
            history["snapshots"].append(snap)

        # Convergence check (must be past the initial transient)
        if ke < ke_tol and it > 50:
            print(f"  ✓ Converged at iteration {it}  (KE = {ke:.6f})")
            converged_at = it
            break
    else:
        print(f"  ✗ Reached max iterations ({max_iter})")

    history["converged_iter"] = converged_at
    return history


# ──────────────────────────────────────────────────────────────────────
# 6. RANDOM BASELINE
# ──────────────────────────────────────────────────────────────────────

def random_placement(
    blocks: List[Block],
    seed  : int = 99,
) -> List[Block]:
    """Generate a uniformly random placement (no overlap guarantee)."""
    rng = np.random.default_rng(seed)
    placed = copy.deepcopy(blocks)
    for b in placed:
        b.x = rng.uniform(b.w / 2, 100 - b.w / 2)
        b.y = rng.uniform(b.h / 2, 100 - b.h / 2)
        b.vx = b.vy = 0.0
    return placed


# ──────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────

def _draw_floorplan(ax, snap, chip_w, chip_h, nets=None, blocks_for_nets=None,
                    title=""):
    """Render a single floorplan snapshot on the given axes."""
    ax.clear()
    ax.set_xlim(-5, chip_w + 5)
    ax.set_ylim(-5, chip_h + 5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")

    # Die outline
    die = mpatches.Rectangle((0, 0), chip_w, chip_h,
                              linewidth=2, edgecolor="#1a1a2e",
                              facecolor="#f0f0f5", linestyle="--")
    ax.add_patch(die)

    # Colour-map by power
    powers = [s[4] for s in snap]
    norm = Normalize(vmin=min(powers), vmax=max(powers))
    cmap = matplotlib.colormaps["YlOrRd"]

    for (x, y, w, h, pwr, name) in snap:
        colour = cmap(norm(pwr))
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.3",
            facecolor=colour, edgecolor="#333", linewidth=1.2, alpha=0.90,
        )
        ax.add_patch(rect)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="#1a1a2e")

    # Draw net connections as thin lines
    if nets and blocks_for_nets is None:
        # Use snap positions
        for net_ids in nets:
            if len(net_ids) >= 2:
                for k in range(len(net_ids) - 1):
                    x1, y1 = snap[net_ids[k]][0], snap[net_ids[k]][1]
                    x2, y2 = snap[net_ids[k+1]][0], snap[net_ids[k+1]][1]
                    ax.plot([x1, x2], [y1, y2], color="#3498db",
                            linewidth=0.4, alpha=0.3)

    # Heat-sink annotation
    ax.annotate("HEAT SINK", xy=(chip_w / 2, -3), fontsize=8,
                ha="center", color="#c0392b", fontweight="bold")


def plot_metrics(history: dict, baseline_wl: float, baseline_ov: float,
                 save_prefix: str = "result"):
    """Generate publication-quality metric plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    iters = range(1, len(history["wirelength"]) + 1)

    # ── Wirelength ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(iters, history["wirelength"], color="#2980b9", linewidth=1.4,
            label="Physics-AI", alpha=0.85)
    ax.axhline(baseline_wl, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({baseline_wl:.1f})")
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("HPWL (units)", fontsize=10)
    ax.set_title("Half-Perimeter Wirelength", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Overlap ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(iters, history["overlap"], color="#27ae60", linewidth=1.4,
            label="Physics-AI", alpha=0.85)
    ax.axhline(baseline_ov, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({baseline_ov:.1f})")
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Overlap area (units²)", fontsize=10)
    ax.set_title("Total Overlap Area", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Kinetic energy ────────────────────────────────────────────────
    ax = axes[2]
    ax.semilogy(iters, np.array(history["kinetic"]) + 1e-12,
                color="#8e44ad", linewidth=1.4, alpha=0.85)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Kinetic Energy (log)", fontsize=10)
    ax.set_title("System Kinetic Energy", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}_metrics.png", dpi=200, bbox_inches="tight")
    print(f"  → Saved {save_prefix}_metrics.png")
    plt.close(fig)


def plot_floorplan_comparison(
    snap_initial, snap_final, snap_random,
    chip_w=100, chip_h=100, save_prefix="result",
):
    """Side-by-side floorplan comparison (initial / optimised / random)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_floorplan(axes[0], snap_initial, chip_w, chip_h,
                    title="(a) Initial Placement")
    _draw_floorplan(axes[1], snap_final, chip_w, chip_h,
                    title="(b) Physics-AI Optimized")
    _draw_floorplan(axes[2], snap_random, chip_w, chip_h,
                    title="(c) Random Baseline")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_floorplans.png", dpi=200, bbox_inches="tight")
    print(f"  → Saved {save_prefix}_floorplans.png")
    plt.close(fig)


def create_animation(
    history: dict,
    chip_w: float = 100.0,
    chip_h: float = 100.0,
    save_path: str = "result_animation.gif",
    fps: int = 12,
):
    """Export an animated GIF of the optimisation process."""
    snaps = history["snapshots"]
    if not snaps:
        print("  ⚠ No snapshots to animate.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    def _update(frame_idx):
        title = f"Iteration {(frame_idx + 1) * 2} / {len(snaps) * 2}"
        _draw_floorplan(ax, snaps[frame_idx], chip_w, chip_h, title=title)

    anim = animation.FuncAnimation(fig, _update, frames=len(snaps),
                                   interval=1000 // fps, repeat=False)
    anim.save(save_path, writer="pillow", fps=fps)
    print(f"  → Saved {save_path}")
    plt.close(fig)


def plot_power_thermal(history: dict, save_prefix: str = "result"):
    """Plot power-position correlation over iterations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = range(1, len(history["power_corr"]) + 1)

    # Apply smoothing for clearer trends
    corr = np.array(history["power_corr"])
    window = min(15, len(corr) // 4)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(corr, kernel, mode="same")
    else:
        smoothed = corr

    ax.plot(iters, corr, color="#f39c12", linewidth=0.6, alpha=0.35,
            label="Raw")
    ax.plot(iters, smoothed, color="#d35400", linewidth=2.0,
            label=f"Smoothed (window={window})")
    ax.axhline(0, color="#7f8c8d", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Pearson r (Power vs. Y-position)", fontsize=10)
    ax.set_title("Thermal-Aware Clustering Convergence", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_thermal.png", dpi=200, bbox_inches="tight")
    print(f"  → Saved {save_prefix}_thermal.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 8. COMPARATIVE SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────

def print_summary(history: dict, baseline_wl: float, baseline_ov: float,
                  baseline_pc: float):
    """Print a LaTeX-ready comparison table to stdout."""
    final_wl = history["wirelength"][-1]
    final_ov = history["overlap"][-1]
    final_pc = history["power_corr"][-1]
    conv     = history["converged_iter"]

    wl_impr = (baseline_wl - final_wl) / max(baseline_wl, 1e-8) * 100
    ov_impr = (baseline_ov - final_ov) / max(baseline_ov, 1e-8) * 100

    sep = "─" * 60
    print(f"\n{sep}")
    print("  COMPARATIVE RESULTS SUMMARY")
    print(sep)
    print(f"  {'Metric':<30} {'Physics-AI':>12} {'Random':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12}")
    print(f"  {'HPWL (units)':<30} {final_wl:>12.2f} {baseline_wl:>12.2f}")
    print(f"  {'Overlap area (units²)':<30} {final_ov:>12.2f} {baseline_ov:>12.2f}")
    print(f"  {'Power-Y correlation':<30} {final_pc:>12.4f} {baseline_pc:>12.4f}")
    print(f"  {'Convergence iterations':<30} {conv:>12d} {'N/A':>12}")
    print(sep)
    print(f"  HPWL improvement vs. random   : {wl_impr:+.1f}%")
    print(f"  Overlap reduction vs. random   : {ov_impr:+.1f}%")
    print(sep)

    # LaTeX table
    print("\n  % ── LaTeX table (paste into paper) ──")
    print(r"  \begin{table}[h]")
    print(r"  \centering")
    print(r"  \caption{Comparison of Physics-AI vs.\ Random Placement}")
    print(r"  \label{tab:results}")
    print(r"  \begin{tabular}{lcc}")
    print(r"  \toprule")
    print(r"  Metric & Physics-AI & Random \\")
    print(r"  \midrule")
    print(f"  HPWL (units) & {final_wl:.2f} & {baseline_wl:.2f} \\\\")
    print(f"  Overlap (units$^2$) & {final_ov:.2f} & {baseline_ov:.2f} \\\\")
    print(f"  Power corr. & {final_pc:.4f} & {baseline_pc:.4f} \\\\")
    print(f"  Convergence iter. & {conv} & N/A \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \end{table}")


# ──────────────────────────────────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Physics-Inspired Agent-Based VLSI Floorplanning")
    print("=" * 64)

    # ── Generate benchmark ────────────────────────────────────────────
    blocks, nets = generate_benchmark(n_blocks=12, n_nets=18, seed=42)
    print(f"\n  Benchmark: {len(blocks)} blocks, {len(nets)} nets")

    # Record initial snapshot
    snap_initial = [(b.x, b.y, b.w, b.h, b.power, b.name) for b in blocks]

    # ── Physics engine  ───────────────────────────────────────────────
    engine = PhysicsEngine(
        chip_w  = 100.0,
        chip_h  = 100.0,
        k_attr  = 0.04,
        k_rep   = 1.2,
        k_grav  = 0.08,
        k_bnd   = 2.0,
        damping = 0.65,
        dt      = 1.0,
        v_max   = 5.0,
    )

    # ── Run optimisation ──────────────────────────────────────────────
    print("\n  Running physics simulation ...")
    history = run_simulation(blocks, nets, engine,
                             max_iter=500, ke_tol=0.01, record_every=2)
    snap_final = [(b.x, b.y, b.w, b.h, b.power, b.name) for b in blocks]

    # ── Random baseline ───────────────────────────────────────────────
    rand_blocks = random_placement(blocks, seed=99)
    baseline_wl = half_perimeter_wirelength(rand_blocks, nets)
    baseline_ov = total_overlap(rand_blocks)
    baseline_pc = power_gradient_score(rand_blocks)
    snap_random = [(b.x, b.y, b.w, b.h, b.power, b.name) for b in rand_blocks]

    # ── Results ───────────────────────────────────────────────────────
    print_summary(history, baseline_wl, baseline_ov, baseline_pc)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_metrics(history, baseline_wl, baseline_ov, save_prefix="result")
    plot_floorplan_comparison(snap_initial, snap_final, snap_random,
                              save_prefix="result")
    plot_power_thermal(history, save_prefix="result")
    create_animation(history, save_path="result_animation.gif", fps=12)
    print("\n  Done. ✓")


if __name__ == "__main__":
    main()
