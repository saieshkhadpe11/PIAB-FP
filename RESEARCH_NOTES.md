# Physics-Inspired Agent-Based Optimization for VLSI Floorplanning

## Research Notes & IEEE Paper Material

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION CONTROLLER                     │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │Benchmark │→ │ Physics  │→ │ Iteration │→ │ Metrics &  │  │
│  │Generator │  │ Engine   │  │   Loop    │  │ Visualiser │  │
│  └─────────┘  └──────────┘  └───────────┘  └────────────┘  │
│       │            │              │               │          │
│       ▼            ▼              ▼               ▼          │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Block   │  │  Force   │  │Convergence│  │   HPWL,    │  │
│  │ Agents  │  │Equations │  │  Checker  │  │  Overlap,  │  │
│  │ + Nets  │  │(4 types) │  │ (KE→0)   │  │  Thermal   │  │
│  └─────────┘  └──────────┘  └───────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Description

| Component           | Role                                                        |
| ------------------- | ----------------------------------------------------------- |
| **Benchmark Generator** | Creates synthetic VLSI circuits with realistic block sizing and power profiles |
| **Block Agents**    | Autonomous entities with position, velocity, dimensions, and power attributes |
| **Net Hyperedges**  | Weighted connectivity relations between block pairs         |
| **Physics Engine**  | Computes four force categories per block per iteration       |
| **Iteration Loop**  | Velocity-Verlet integration with damping until KE convergence |
| **Metrics Module**  | HPWL, overlap area, power–position correlation              |
| **Visualiser**      | Static plots, animated GIF, and comparison figures          |

---

## 2. Mathematical Force Model

### 2.1 Notation

| Symbol             | Description                              |
| ------------------ | ---------------------------------------- |
| $\mathbf{r}_i = (x_i, y_i)$ | Centre position of block $i$   |
| $w_i, h_i$        | Width and height of block $i$            |
| $P_i$             | Power dissipation of block $i$           |
| $c_{ij}$          | Connectivity weight of net $(i,j)$       |
| $\mathcal{N}(i)$  | Set of blocks connected to $i$           |
| $W, H$            | Die width and height (100 × 100)         |

### 2.2 Force Equations

#### (a) Attraction Force (Hooke's Spring Model)

Connected blocks are linked by virtual springs:

$$
\mathbf{F}_{\text{attr}}^{(i)} = \sum_{j \in \mathcal{N}(i)} k_a \cdot c_{ij} \cdot (\mathbf{r}_j - \mathbf{r}_i)
$$

- **Physical analogy**: Spring with rest length zero and stiffness $k_a \cdot c_{ij}$
- **Effect**: Minimises wirelength by pulling connected blocks together

#### (b) Repulsion Force (Soft-Body Collision)

For overlapping blocks $i$ and $j$ with overlap area $A_{ij}$:

$$
\mathbf{F}_{\text{rep}}^{(i)} = \sum_{j: A_{ij}>0} k_r \cdot A_{ij} \cdot \hat{\mathbf{d}}_{ij}
$$

where $\hat{\mathbf{d}}_{ij} = \frac{\mathbf{r}_i - \mathbf{r}_j}{\|\mathbf{r}_i - \mathbf{r}_j\|}$ is the unit vector from $j$ to $i$.

$$
A_{ij} = \max\left(0, \min(x_i + \tfrac{w_i}{2}, x_j + \tfrac{w_j}{2}) - \max(x_i - \tfrac{w_i}{2}, x_j - \tfrac{w_j}{2})\right) \times (\text{same for } y)
$$

- **Physical analogy**: Incompressible soft-body contact
- **Effect**: Resolves overlaps in finite iterations

#### (c) Thermal Gravity

$$
F_{\text{grav},y}^{(i)} = -k_g \cdot P_i
$$

- **Physical analogy**: Gravitational field proportional to "mass" (power)
- **Effect**: High-power blocks migrate toward $y = 0$ (heat sink edge)

#### (d) Boundary Confinement

For each wall, if block $i$ penetrates by depth $\delta$:

$$
F_{\text{bnd}}^{(i)} = k_b \cdot \max(0, \delta)
$$

applied inward. For example, left wall: $\delta = \tfrac{w_i}{2} - x_i$.

### 2.3 Integration Scheme

Velocity update with damping factor $\gamma \in (0, 1]$:

$$
\mathbf{v}_i^{(t+1)} = \gamma \left( \mathbf{v}_i^{(t)} + \mathbf{F}_i^{(t)} \cdot \Delta t \right)
$$

$$
\mathbf{r}_i^{(t+1)} = \mathbf{r}_i^{(t)} + \mathbf{v}_i^{(t+1)} \cdot \Delta t
$$

### 2.4 Convergence Criterion

Total kinetic energy:

$$
\text{KE}^{(t)} = \sum_{i=1}^{N} \frac{1}{2} \left( v_{x,i}^2 + v_{y,i}^2 \right)
$$

Terminate when $\text{KE}^{(t)} < \epsilon$ (with $\epsilon = 10^{-3}$) or after $T_{\max}$ iterations.

### 2.5 Hyperparameter Table

| Parameter | Symbol   | Default | Description                    |
| --------- | -------- | ------- | ------------------------------ |
| `k_attr`  | $k_a$   | 0.02    | Attraction spring constant     |
| `k_rep`   | $k_r$   | 0.80    | Repulsion stiffness            |
| `k_grav`  | $k_g$   | 0.05    | Gravity coefficient            |
| `k_bnd`   | $k_b$   | 1.50    | Boundary spring constant       |
| `damping` | $\gamma$ | 0.85   | Velocity damping factor        |
| `dt`      | $\Delta t$ | 1.0  | Integration time step          |

---

## 3. Pseudocode (IEEE Paper Ready)

```
Algorithm 1: Physics-Inspired Agent-Based Floorplanning (PIAB-FP)
────────────────────────────────────────────────────────────────

Input : Block set B = {b₁, ..., bₙ} with (wᵢ, hᵢ, Pᵢ)
        Net set E = {e₁, ..., eₘ} with connectivity weights
        Die dimensions (W, H)
        Hyperparameters: kₐ, kᵣ, k_g, k_b, γ, Δt, ε, T_max

Output: Optimised placement {(xᵢ, yᵢ)} for all blocks

1:  INITIALISE positions randomly within die
2:  INITIALISE velocities to zero
3:  for t = 1 to T_max do
4:      for each block i = 1 to N do
5:          F_i ← (0, 0)
6:          ▸ ATTRACTION:
7:          for each net e containing block i do
8:              for each block j ∈ e, j ≠ i do
9:                  F_i ← F_i + kₐ · cₑ · (rⱼ − rᵢ)
10:         ▸ REPULSION:
11:         for each block j ≠ i do
12:             A_ij ← OVERLAP_AREA(i, j)
13:             if A_ij > 0 then
14:                 F_i ← F_i + kᵣ · A_ij · (rᵢ − rⱼ) / ‖rᵢ − rⱼ‖
15:         ▸ GRAVITY:
16:         F_i,y ← F_i,y − k_g · Pᵢ
17:         ▸ BOUNDARY:
18:         for each die edge do
19:             δ ← penetration depth
20:             if δ > 0 then F_i ← F_i + k_b · δ · n̂_inward
21:     ▸ INTEGRATION:
22:     for each block i do
23:         vᵢ ← γ · (vᵢ + Fᵢ · Δt)
24:         rᵢ ← rᵢ + vᵢ · Δt
25:     ▸ CONVERGENCE:
26:     KE ← Σᵢ ½(vₓ,ᵢ² + v_y,ᵢ²)
27:     if KE < ε and t > 20 then
28:         return placement
29: return placement
```

---

## 4. Experiment Design (IEEE Format)

### 4.1 Experimental Setup

> The benchmark comprises N = 12 macro-blocks with widths and heights
> drawn uniformly from [5, 20] units on a 100 × 100 unit die area.
> Power dissipation values follow a log-normal distribution
> (μ = 1.0, σ = 0.8), clipped to [0.5, 30] W, reflecting the skewed
> power profiles observed in modern SoC designs. M = 18 two-terminal
> nets are randomly sampled from all possible block pairs with
> connectivity weights drawn from U(0.5, 3.0).

### 4.2 Baseline

> A uniformly random placement baseline is used for comparison, where
> each block centre is drawn from U(wᵢ/2, W − wᵢ/2) × U(hᵢ/2, H − hᵢ/2)
> without overlap avoidance.

### 4.3 Evaluation Metrics

1. **Half-Perimeter Wirelength (HPWL):** Standard proxy for total
   interconnect length, weighted by net connectivity.

2. **Total Overlap Area:** Sum of pairwise AABB intersection areas —
   measures placement legality.

3. **Power–Position Correlation:** Pearson's r between block power
   and vertical position; a negative value indicates successful
   thermal-aware clustering near the heat-sink boundary.

4. **Convergence Iterations:** Number of iterations until kinetic
   energy drops below ε = 10⁻³.

### 4.4 Expected Results Interpretation

| Outcome | Interpretation |
| ------- | -------------- |
| HPWL decreases monotonically | Attraction forces effectively minimise wirelength |
| Overlap → 0 | Repulsion forces successfully legalise placement |
| Power corr. becomes negative | Gravity force clusters hot blocks near bottom |
| KE → 0 in < 300 iterations | System reaches stable equilibrium efficiently |
| Physics-AI HPWL < Random HPWL | Proposed method outperforms naïve baseline |

---

## 5. Novelty Framing

### Suggested paper title:
> **"PIAB-FP: A Physics-Inspired Agent-Based Framework for
> Thermal-Aware VLSI Macro-Block Floorplanning"**

### Key novelty claims:

1. **Agent-based modelling** — Each VLSI block is treated as an
   autonomous physical agent, enabling emergent collective behaviour
   rather than top-down optimisation.

2. **Multi-physics force composition** — Four physically-motivated
   force classes (attraction, repulsion, gravity, boundary) are
   combined in a unified Newtonian framework, allowing intuitive
   parameter tuning and interpretable dynamics.

3. **Thermal-aware gravity** — Unlike classical force-directed
   placement, the gravitational component is proportional to block
   power dissipation, explicitly encoding thermal constraints without
   requiring a separate thermal solver.

4. **Convergence guarantee** — The damped Newtonian dynamics with
   bounded forces ensure monotonic kinetic energy dissipation,
   providing a natural convergence criterion.

### Related work positioning:

| Approach | Limitation our work addresses |
| -------- | ----------------------------- |
| Simulated annealing (SA) | No physical intuition; slow cooling schedules |
| Genetic algorithms | Population overhead; no spatial reasoning |
| Analytical placement (ePlace) | Density functions are non-physical; no thermal modelling |
| Force-directed (Eisenmann) | No thermal gravity; no explicit overlap resolution |

---

## 6. Suggested Extensions (Future Work)

1. **Adaptive force constants** — Anneal $k_a, k_r$ over iterations
   using a schedule inspired by simulated annealing.

2. **Multi-objective Pareto front** — Trade off wirelength vs. thermal
   vs. area utilisation using multi-objective variants.

3. **Hierarchical decomposition** — Apply the framework recursively
   on clustered sub-problems for scalability to 1000+ blocks.

4. **ISPD/MCNC benchmarks** — Validate on standard benchmarks
   (ami33, ami49, GSRC) for direct comparison with published results.

5. **GPU acceleration** — Parallelise force computation using CUDA
   for real-time placement of large designs.

---

## 7. How to Run

```bash
pip install numpy matplotlib
python vlsi.py
```

### Output Files

| File                      | Description                                   |
| ------------------------- | --------------------------------------------- |
| `result_metrics.png`      | HPWL, overlap, and KE convergence plots       |
| `result_floorplans.png`   | Side-by-side: initial / optimised / random     |
| `result_thermal.png`      | Power–position correlation over iterations     |
| `result_animation.gif`    | Animated GIF of the optimisation process       |

---

*Document generated: February 2026*
