#!/usr/bin/env python3
"""Regenerate XLEAP_detector.ipynb (nbformat 4.4) with stdlib json only.

Produces a research-paper-style notebook: prose, equations, and figures, with
code cells collapsed by default (jupyter.source_hidden) so it reads top to bottom
like a report. Run All once on a pandas-equipped kernel to populate the figures.

Portable: writes the notebook next to this script.
Run:  python build_notebook.py
"""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s, hide=True):
    """A code cell. ``hide`` collapses the input so only the output shows."""
    meta = {"jupyter": {"source_hidden": True}} if hide else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None,
            "outputs": [], "source": s}


cells = []

# ============================================================================
#  Title + how-to-read
# ============================================================================
cells.append(md(r"""# Detecting XLEAP from Undulator K-Taper

**A data-driven detector that identifies XLEAP, quantifies the undulator taper in MeV/fs, and reports the lasing undulators, from archived $K$ and beam-energy snapshots.**

*N. Mamais, LCLS. 2026-07-01*

---

**Abstract.** X-ray Laser-Enhanced Attosecond Pulse generation (XLEAP) imprints a characteristic *taper* on the undulator line: a monotonic ramp in the deflection parameter $K$ across a run of undulators. This notebook reconstructs, for every archived snapshot, (i) whether such a tapered lasing group exists, (ii) which undulators participate, and (iii) the taper expressed as resonant-energy gain per slippage length in MeV/fs. All physics lives in the [`taper`](taper/) package; the notebook is a thin, reproducible driver over the long-form `snapshots.csv`. We validate the pipeline on the SXR soft line (June 2026), characterise a class of high-taper false positives, and show that they arise from off-nominal beam energy rather than undulator motion.

**Reference:** K.-J. Kim, Z. Huang, and R. Lindberg, *Undulator Radiation*, in *Synchrotron Radiation and Free-Electron Lasers*, Cambridge University Press, 2017, pp. 43 to 45."""))

cells.append(md(r"""> **How to read this notebook.** Code cells are collapsed by default, so you can read the prose and figures top to bottom like a paper. Click a collapsed cell to expand its source. To populate every figure and table, run **Kernel > Restart & Run All** once on a kernel that has `numpy`, `pandas`, `scipy`, and `matplotlib` (the `xleap` conda env)."""))

# ============================================================================
#  1. Introduction
# ============================================================================
cells.append(md(r"""## 1. Introduction

In an undulator-based free-electron laser the beam radiates at a resonant wavelength set by the beam energy $\gamma$ and the undulator deflection parameter $K$. **XLEAP** drives the machine into a regime where a short current spike lases while a deliberate *taper*, a controlled increase of $K$ from one undulator to the next, keeps the radiation resonant with the energy-chirped slices of the spike. Operationally, XLEAP therefore shows up as a **"hockey stick"** in the $K$ profile: a flat head followed by a monotonic ramp over several consecutive undulators.

Our task is to turn the machine's archived history into a physically meaningful timeline. For each nominal snapshot time we answer three questions:

1. **Is XLEAP happening?** Does a lasing group of $\ge 4$ tapered undulators exist?
2. **Which undulators lase?** Reported as a `{undulator: K}` map.
3. **What is the taper?** The first (most-upstream) group's taper in **MeV/fs**, from the median $\Delta K$, the group's first $K$, and the beam $\gamma$.

The remainder of the notebook derives the taper from first principles (Section 2), describes the data pipeline and detection algorithm (Section 3), presents the reconstructed timeline (Section 4), validates the result and the motion filter (Section 5), and collects the open physics questions (Section 6)."""))

# ============================================================================
#  2. Theory
# ============================================================================
cells.append(md(r"""## 2. Theory

In an undulator with magnetic field strength $B_0$ and wavenumber $k_u$, the electrons encounter a magnetic field in $\hat y$

$$
\vec{B}_u = B_0 \sin(k_u z) \hat y
$$

that produces a force in $\hat x$:

$$
\begin{align*}
\dot {\vec p} = \partial_t (\gamma mc \vec \beta)&= -e (\xcancel{\vec E} + \vec v \times \vec B) \\
&= -e(\beta c \hat z) \times B_0 \sin(k_u z) \hat y \\
&= +e\beta c B_0 \sin(k_u z) \hat x \\
\end{align*}
$$
Neglecting energy losses from steering radiation, we take $\dot \gamma = 0$ and integrate (taking $\beta_x = 0$ at $t=0$ from before the undulator) to find $\beta_x$:
$$
\begin{align*}
\dot \beta_x &= \frac{e \beta c B_0}{\gamma m c} \sin(k_u z) = \frac{e \beta B_0}{\gamma m} \sin(k_u \beta c t) \tag{$\beta_z \approx \beta \lesssim 1$} \\
\beta_x &= \left. \frac{e \beta B_0}{\gamma m k_u \beta c} \cos(k_u \beta c t)\right \vert_0^t = \frac{e B_0}{\gamma k_u m c} \cos(k_u \beta c t) - 0 = \frac{1}{\gamma}{\frac{e B_0}{k_u m c}}\cos(k_u z)  \\
\beta_x &= \frac{K}{\gamma} \cos(k_u z) \; \llap{\boxed{\phantom{\beta_x = \frac{K}{\gamma} \cos(k_u z)}}} \tag{$K \coloneqq {e B_0}/{k_u m c}$}
\end{align*}
$$

Again neglecting energy losses from steering radiation, we say that energy $\gamma mc^2$ and therefore total velocity $\beta$ is conserved from before the undulator, but once inside the undulator, the velocity is split between $x$ and $z$ components:

$$
\begin{align*}
{\vec \beta}^2 &= \beta_z^2 + \beta_x^2 \\
\beta_z^2 &= \vec \beta^2 - \beta_x^2 = \left(1 - \frac {1}{\gamma^2} \right) - \beta_x^2 \\
\beta_z &= \sqrt{1-\frac{1}{\gamma^2}-\beta_x^2} \\
&\approx 1- \frac{1}{2\gamma^2} - \frac{K^2}{2\gamma^2}\cos^2(k_u z) \tag{$\sqrt{1-\varepsilon}\approx 1-\varepsilon/2$} \\
\left\langle \beta_z \right\rangle &= 1- \frac{1}{2\gamma^2} - \frac{K^2}{2\gamma^2}\left\langle\cos^2({k_u z})\right\rangle = 1- \frac{1}{2\gamma^2} - \frac{K^2}{4\gamma^2} \\
\left\langle \beta_z \right\rangle &= 1- \left(\frac{1+K^2/2}{2\gamma^2}\right) \; \llap{\boxed{\phantom{\left\langle \beta_z \right\rangle = 1- \left(\frac{1+K^2/2}{2\gamma^2}\right)}}}
\end{align*}
$$
Therefore, the radiation emitted by the electrons is faster than the electrons themselves by $\Delta \beta \coloneqq 1-\left\langle \beta_z \right\rangle = (1+K^2/2)/2\gamma^2$.

The electrons now experience fields due to both the undulator magnets and their own radiation. To achieve resonance, these contributions must be in phase, so the radiation must slip ahead of the electrons by one radiation wavelength $\lambda_r$ per undulator period $\lambda_u$. Light traverses an undulator period in time $T = \lambda_u/c$, during which time the electrons fall behind by a distance $\Delta \beta c T = \lambda_u(1+K^2/2)/2\gamma^2$, called the **slippage length**. Therefore we set the radiation wavelength equal to one slippage length:
$$
\boxed{\lambda_r = \frac{\lambda_u}{2\gamma^2}\left(1+\frac{K^2}{2}\right)}
$$

Meanwhile, space charge forces create an energy spread within the current spike by pushing head electrons forward and tail electrons backward, so the radiation slips ahead from lower-energy to higher-energy slices of electrons: $\gamma \to \gamma + \Delta \gamma$. Due to the taper, $K$ also increases downstream: $K \to K + \Delta K$. To maintain resonance, we now need
$$
\lambda_r = \frac{\lambda_u}{2(\gamma + \Delta \gamma)^2}\left(1+\frac{(K + \Delta K)^2}{2}\right)
$$

Therefore, given a $\Delta K$ between two undulators, we can find the corresponding $\Delta \gamma$ needed to maintain resonance:
$$
\begin{align*}
\lambda_r = \frac{\lambda_u}{2\gamma^2}\left(1+\frac{K^2}{2}\right) &= \frac{\lambda_u}{2(\gamma+\Delta \gamma)^2}\left(1+\frac{(K+\Delta K)^2}{2}\right) \\
\implies \Delta \gamma &= \gamma\left(\sqrt{1+\frac{(K+\Delta K/2)\Delta K}{1+K^2/2}}-1\right) \; \llap{\boxed{\phantom{\Delta \gamma = \gamma\left(\sqrt{1+\frac{(K+\Delta K/2)\Delta K}{1+K^2/2}}-1\right)}}} \approx \gamma\left(\frac{K\Delta K + \Delta K^2/2}{K^2+2}\right)
\end{align*}
$$

Finally, we define *undulator taper* as change in resonant energy $\gamma$ per slippage length $\lambda_r$. Since we find $\Delta K$ between undulators and we get $N_u = L_u/\lambda_u$ periods per undulator ($L_u = 3.4 \text{ m}$), we divide $\Delta \gamma$ by the total slippage in one undulator $\Delta s \coloneqq N_u \lambda_r$. However, we write the slippage length in units of time, $N_u \lambda_r/c$, so that we can write the result in MeV/fs.
$$
\mathrm{taper} \coloneqq \frac{\Delta \gamma}{\Delta s} = \frac{\Delta \gamma}{N_u \lambda_r/c} = \frac{\Delta \gamma \lambda_u c}{L_u} \left[\frac{\lambda_u}{2\gamma^2}\left(1+\frac{K^2}{2}\right)\right]^{-1} = \frac{2 \gamma^2 \Delta \gamma c}{L_u(1+K^2/2)}
$$

Surprisingly, $\lambda_u$ cancels, so we can get the taper from the list of $K$ values (which also gives $\Delta K \to \Delta \gamma$), the beam energy $\gamma$, and the length of an undulator $L_u = 3.4\text{ m}$. These relations live in [`taper/physics.py`](taper/physics.py) as `dk_to_dgamma`, `slippage_time_s`, and `taper_mev_per_fs`."""))

cells.append(md(r"""### 2.1 The $\gamma^3$ scaling (important for Section 5)

Because $\Delta\gamma \propto \gamma$, the taper carries a strong **cubic** dependence on beam energy:
$$
\mathrm{taper} \;\propto\; \gamma^{2}\,\Delta\gamma \;\propto\; \gamma^{3}.
$$
A given $K$-ramp therefore produces a taper that grows like $\gamma^3$. The *same* ramp at $15.7$ GeV/c rather than $10$ GeV/c is about $(15.7/10)^3\approx 3.9$ times larger from energy alone. As Section 5 shows, this is exactly why an ordinary $\Delta K$ can read as a $\sim700$ MeV/fs taper when the machine is at off-nominal energy, and why a *flat* taper ceiling would be the wrong way to reject such points."""))

# ============================================================================
#  3. Methods
# ============================================================================
cells.append(md(r"""## 3. Methods

### 3.1 Data pipeline and the beamline switch

The tool cleanly separates **fetch** from **analysis**:

| module | role |
|---|---|
| `beamlines` | the single line switch (`HXR` / `SXR`); flip `DEFAULT_LINE` there to change lines |
| `snapshots` | fetch binned PV history from the MEME archive into long `snapshots.csv` |
| `taper.constants` | physical constants and the I/O contract (re-exports `DEFAULT_LINE`) |
| `taper.physics` | pure, vectorized taper math |
| `taper.detect` | lasing-group detection (binary erosion / dilation) |
| `taper.store` | long `snapshots.csv` into wide pandas frames |
| `taper.service` | per-time XLEAP timeline assembly |

Both the fetch default PV list and the analysis PV parsing read the same `beamlines.DEFAULT_LINE`, so they cannot disagree. The long CSV `nominal_time, pv, timestamp, value, moved` is the data source of record, with no SQLite or sparklines dependency.

- **$L_u = 3.4$ m** per undulator; inter-undulator drifts are neglected (negligible slippage without the field coupling $\beta_z$ into $\beta_x$).
- **$\gamma$** comes from the active line's dump PV (`BEND:DMPH:400:BACT` on HXR, `BEND:DMPS:400:BACT` on SXR), the beam momentum $p$ in **GeV/c**: $\gamma = \sqrt{(pc)^2+(mc^2)^2}/mc^2$ with $mc^2=0.511$ MeV.
- **$K$** is each undulator's `KAct`; `store.wide_values` pivots the long CSV into a `time x undulator` matrix ordered upstream to downstream.

The setup cell below loads the data and reports the active line."""))

cells.append(code(r"""from datetime import datetime as dt

import numpy as np
import pandas as pd

from taper import (
    SnapshotStore,
    DetectionParams,
    DEFAULT_LINE as LINE,   # active undulator line; flip the ONE switch in beamlines.py
    lasing_kvals,
    first_group,
    gamma_from_momentum_gev,
    dk_to_dgamma,
    taper_mev_per_fs,
    xleap_timeline,
    timeline_frame,
)

# snapshots.csv is produced by:  python -m snapshots --out snapshots.csv --from ... --to ...
print(f"Active line: {LINE.name}   K pattern: {LINE.kact_pattern.pattern}   momentum PV: {LINE.momentum_pv}")
store = SnapshotStore.from_csv("snapshots.csv")
store.frame.head()"""))

cells.append(md(r"""### 3.2 Detection: the hockey-stick as a morphological opening

XLEAP is declared when a run of at least **`min_group` = 4** consecutive undulators each step up in $K$ by at least $\rho\cdot\texttt{rho\_scale} = 4\rho$ with $\rho\sim10^{-3}$ (the FEL Pierce parameter). We build the boolean "is this step big enough" mask along the undulator axis, then keep only runs of length $\ge 4$ using **binary erosion followed by dilation**, a morphological *opening*, the vectorized way to say "at least $N$ in a row." The toy example below shows opening discarding the short runs and keeping the length-4 run:"""))

cells.append(code(r"""from scipy.ndimage import binary_erosion, binary_dilation


def keep_long_runs(arr, n=4):
    structure = [1] * n
    return binary_dilation(binary_erosion(arr, structure=structure), structure=structure)


keep_long_runs(np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=bool)).astype(int)"""))

cells.append(md(r"""`DetectionParams` bundles the thresholds (`rho=1e-3`, `rho_scale=4`, `min_group=4`). `lasing_mask` returns a boolean `time x undulator` frame marking every undulator in a lasing group, including the base undulator just upstream of the first big step (the fencepost correction, since $\Delta K_i$ describes the rise from undulator $i-1$ to $i$, so the group owns one more undulator on its upstream edge). `lasing_kvals` blanks the non-lasing entries to `NaN`.

### 3.3 Taper of a lasing group

For the first (most-upstream) lasing group we take the **median** $\Delta K$ across the group, convert it to $\Delta\gamma$ using the group's first $K$ and the beam $\gamma$, and divide by one undulator's slippage time:
$$
\mathrm{taper} = \frac{\Delta\gamma}{N_u\lambda_r/c} = \frac{2\gamma^2\,\Delta\gamma\,c}{L_u\,(1+K^2/2)}
\qquad\text{(}\texttt{taper\_mev\_per\_fs(dK, K, gamma)}\text{)}.
$$
The **median** rather than the mean makes the estimate robust to the single large step created by the missing undulator slot in the physical lattice."""))

# ============================================================================
#  4. Results  (single figure only)
# ============================================================================
cells.append(md(r"""## 4. Results

`xleap_timeline(store, line=LINE)` returns one `XleapPoint` per nominal time, the documented data object `{ 'datetime', 'xleap_on', 'K_lasing', 'n_und', 'taper', 'moving' }`. The figure below is the full reconstructed timeline over the fetched window: the taper (left axis, blue) and the number of lasing undulators (right axis, orange). Motion-flagged points are force-cleared to `taper = NaN` (Section 5.1). The clean $\sim20$ MeV/fs band is real XLEAP; the isolated spikes up to $\sim730$ MeV/fs are the subject of Section 5."""))

cells.append(code(r"""%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

points = xleap_timeline(store, line=LINE)
xleap_df = timeline_frame(points)

fig, taper_ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
und_ax = taper_ax.twinx()
mask = xleap_df["xleap_on"]
taper_ax.plot(xleap_df.index, xleap_df["taper"], marker="o", ms=3, lw=1, color="tab:blue", label="Taper")
und_ax.step(xleap_df.index, xleap_df["n_und"], where="mid", lw=1.5, color="tab:orange", label="Lasing undulators")
if mask.any():
    taper_ax.scatter(xleap_df.index[mask], xleap_df.loc[mask, "taper"], s=18, color="tab:blue", zorder=3)
taper_ax.set_xlim(xleap_df.index.min(), xleap_df.index.max())
taper_ax.set_title(f"{LINE.name}: XLEAP taper and lasing undulator count")
taper_ax.set_xlabel("Date"); taper_ax.set_ylabel("Taper [MeV/fs]", color="tab:blue")
und_ax.set_ylabel("Lasing undulators", color="tab:orange")
und_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
taper_ax.tick_params(axis="y", labelcolor="tab:blue"); und_ax.tick_params(axis="y", labelcolor="tab:orange")
taper_ax.grid(True, alpha=0.3)
loc = mdates.AutoDateLocator()
taper_ax.xaxis.set_major_locator(loc); taper_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
lines = taper_ax.get_lines() + und_ax.get_lines()
taper_ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
plt.show()"""))

# ============================================================================
#  5. Validation & investigation
# ============================================================================
cells.append(md(r"""## 5. Validation and investigation

The June 2026 SXR timeline shows a clean $\sim20$ MeV/fs XLEAP band plus a handful of extreme spikes up to $\sim730$ MeV/fs. This section establishes what those spikes are, confirms the motion filter neither misses real motion nor discards real XLEAP, and settles the motion-threshold choice with data.

The cell below builds the "with filter" (`after`) and "without filter" (`before`) timelines from the same data (the latter by forcing every `moved` flag to `False`), so we can see exactly what the motion filter removes."""))

cells.append(code(r"""import dataclasses

after   = timeline_frame(xleap_timeline(store, line=LINE))                                     # filter ON
before  = timeline_frame(xleap_timeline(
    dataclasses.replace(store, frame=store.frame.assign(moved=False)), line=LINE))             # filter OFF
kvals   = store.wide_values(LINE.kact_pattern)
moved   = store.wide_moved(LINE.kact_pattern)
mom     = store.series(LINE.momentum_pv).reindex(after.index)
removed = before.index[before["xleap_on"] & ~after["xleap_on"]]

print(f"snapshots           : {len(after)}")
print(f"XLEAP-on (filtered) : {int(after['xleap_on'].sum())}")
print(f"moved cells total   : {int(store.frame['moved'].sum())}")
print(f"times any-moving    : {int(moved.any(axis=1).sum())} / {len(moved)}")
print(f"removed by filter   : {len(removed)}")"""))

cells.append(md(r"""### 5.1 Motion filter: rule and effect

**Motion rule (per Aaron).** For each nominal time an undulator has *moved* when the fractional spread of its raw samples inside a 5 s window exceeds a threshold $\tau$:
$$
\frac{\max - \min}{\lvert\mathrm{median}\rvert} > \tau \quad(\tau = 10^{-3}),\qquad \le 1\ \text{sample} \Rightarrow \text{stationary (thermal noise)}.
$$
If **any** undulator moved at a given time, that time is force-cleared: `xleap_on = False`, `taper = NaN`, `moving = True`, kept visible rather than dropped.

The figure overlays **before** (motion off, grey) and **after** (motion on, blue/orange); red crosses mark points the filter removed. A correct filter removes only slew/transition points and leaves the stable lasing blocks, and crucially the large spikes, untouched."""))

cells.append(code(r"""fig, (ax_t, ax_n) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
ax_t.plot(before.index, before["taper"], "o", ms=3, color="0.75", label="before (motion off)")
ax_t.plot(after.index,  after["taper"],  "o", ms=3, color="tab:blue", label="after (motion on)")
if len(removed):
    ax_t.plot(removed, before.loc[removed, "taper"], "x", color="red", ms=10, mew=2, label="removed by filter")
ax_t.set_ylabel("Taper [MeV/fs]"); ax_t.grid(alpha=0.3); ax_t.legend(loc="upper left")
ax_n.step(before.index, before["n_und"], where="mid", color="0.75", label="before")
ax_n.step(after.index,  after["n_und"],  where="mid", color="tab:orange", label="after")
ax_n.set_ylabel("Lasing undulators"); ax_n.set_xlabel("Date"); ax_n.grid(alpha=0.3); ax_n.legend(loc="upper left")
fig.suptitle(f"{LINE.name} June 2026: before vs after motion filter (tau = 1e-3)")
plt.show()"""))

cells.append(md(r"""**Observation.** The red crosses land only on the normal $\sim20$ MeV/fs band, at the edges of lasing blocks, where undulators are slewing in or out of a configuration. The large spikes carry no red cross: the motion filter does not touch them. That is the first clue that the spikes are not a motion phenomenon.

### 5.2 The largest spike is stationary, not moving

We drill into the single worst point. Its summary and full $K$ profile follow."""))

cells.append(code(r"""sp = after["taper"].idxmax()
display(pd.DataFrame({
    "time": [sp],
    "taper [MeV/fs]": [round(float(after.loc[sp, "taper"]), 1)],
    "beam [GeV/c]": [round(float(mom.loc[sp]), 3)],
    "n_und": [int(after.loc[sp, "n_und"])],
    "moving": [bool(after.loc[sp, "moving"])],
}))"""))

cells.append(code(r"""ks = kvals.loc[sp]
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(range(len(ks)), ks.values, "o-")
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks.index, rotation=90, fontsize=7)
ax.set_ylabel("K"); ax.set_xlabel("undulator")
ax.set_title(f"K profile at the worst spike ({sp}): clean ramp flanked by detuned undulators; all moved=False")
ax.grid(alpha=0.3); plt.show()"""))

cells.append(md(r"""**The profile tells the whole story.** A clean, monotonic $K$-ramp occupies the middle undulators (here $K\approx5.3$), but it is flanked by heavily detuned undulators ($K$ roughly 1 to 3.7) that are clearly parked, not lasing. Every undulator reports `moved = False`: the Ks are stationary. This is Aaron's "Ks in place, but the beam energy wasn't there yet." A 5 s motion window can never flag it, because there is no motion to flag. We verified separately that even a 10 times tighter threshold ($\tau=10^{-4}$) leaves this point `moving = False`.

### 5.3 The spikes correlate with off-nominal beam energy

If the spikes are a beam-energy artefact, they should occur when the machine is not at the soft line's normal operating energy. Splitting the XLEAP-on points by taper magnitude and summarising the beam momentum confirms it, and the scatter makes the $\gamma^3$ inflation visible."""))

cells.append(code(r"""on = after["xleap_on"]
tbl = pd.concat({
    "real (taper <= 100)":    mom[on & (after["taper"] <= 100)].describe(),
    "spurious (taper > 100)": mom[on & (after["taper"] >  100)].describe(),
}, axis=1)
display(tbl.round(2))"""))

cells.append(code(r"""fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.scatter(mom[on], after["taper"][on], s=14, color="tab:blue")
ax.axhline(100, color="red", ls="--", lw=1, label="taper = 100 MeV/fs")
ax.set_xlabel("beam momentum [GeV/c]"); ax.set_ylabel("taper [MeV/fs]")
ax.set_title("Taper vs beam energy: spikes cluster at off-nominal energy")
ax.grid(alpha=0.3); ax.legend()
plt.show()"""))

cells.append(md(r"""**Result.** The real XLEAP band sits in a tight 4 to 5 GeV/c cluster; the $>100$ MeV/fs spikes occur at 5 to 15.7 GeV/c (median about 8). Combined with the $\gamma^3$ scaling of Section 2.1, an ordinary $\Delta K\approx0.02$, $K\approx4$ to $5$ ramp inflates into hundreds of MeV/fs. The spikes are detection false positives in an off-nominal beam state, not motion, and not a taper-math error.

Note the overlap: the spurious set reaches down to 5 GeV/c and the real set up to 10 GeV/c, and XLEAP legitimately runs at 8 to 10 GeV. A flat energy cut or a flat taper ceiling would therefore clip real high-energy XLEAP; the correct discriminator is a resonance/energy-match condition (Section 6).

### 5.4 Motion-threshold study: $10^{-3}$ vs $10^{-4}$

David asked whether the motion threshold should be $10^{-3}$ or $10^{-4}$. Each value is baked into the `moved` column at fetch time, so the two rows below come from separate fetches and are recorded here as constants (they are not recomputed from the currently-loaded CSV). Both thresholds remove only real-band points, never the spikes."""))

cells.append(code(r"""sweep = pd.DataFrame([
    {"threshold": "1e-3 (recommended)", "removed": 25, "real_XLEAP_removed": 25,
     "isolated_transitions": 18, "in_runs": 6, "note": "only slew/transition points removed"},
    {"threshold": "1e-4",               "removed": 33, "real_XLEAP_removed": 33,
     "isolated_transitions": None, "in_runs": None, "note": "8 extra removals are sub-0.1%, i.e. thermal"},
])
display(sweep)"""))

cells.append(md(r"""**Decision: $\tau = 10^{-3}$.** Going to $10^{-4}$ removes 8 additional real XLEAP points, and by construction those 8 have fractional change between $10^{-4}$ and $10^{-3}$, i.e. sub-$0.1\%$, squarely in the thermal-noise regime the rule is meant to ignore. So $10^{-4}$ starts labelling thermal jitter as motion and eats real XLEAP; $10^{-3}$ removes only the clearly-moving points.

At $10^{-3}$ the 25 removals split into 18 isolated transitions (clean) and 6 that fall in short consecutive runs. The latter are real-band XLEAP points ($n_\text{und}=7$ to 15, taper 14 to 25) removed because an undulator crossed threshold at that moment, rule-consistent but a small, non-zero loss worth flagging:"""))

cells.append(code(r"""in_runs = pd.DataFrame([
    ("2026-06-10 12:30", 15, 23.6), ("2026-06-10 12:45", 15, 23.9),
    ("2026-06-13 23:30",  7, 23.5), ("2026-06-13 23:45", 14, 18.4),
    ("2026-06-14 12:00",  9, 24.5), ("2026-06-20 17:30", 14, 14.2),
], columns=["time", "n_und_before", "taper_before"])
display(in_runs)"""))

# ============================================================================
#  6. Discussion / open questions
# ============================================================================
cells.append(md(r"""## 6. Discussion and open questions

**What is settled.**
- The pipeline reconstructs XLEAP, the lasing undulators, and the taper, and reproduces the reference SXR results.
- The motion filter works: at $\tau=10^{-3}$ it removes 25 points (18 isolated transitions and 6 real-band points during genuine undulator motion) and loses no stable XLEAP.
- The high-taper spikes are not motion (stationary Ks at any threshold) but detection false positives at off-nominal beam energy, amplified by the $\gamma^3$ dependence.

**Open questions (for David).**
1. **Resonance / energy-match gate.** Reject "Ks tapered but beam not at the lasing energy." It must still admit real 8 to 10 GeV XLEAP, so it cannot be a flat energy or taper cap. Natural form: require the group's $K$ to be near the resonant $K$ for the measured $\gamma$ and target photon energy (needs $\lambda_u$ or photon energy; the reference cites SXR at 39 mm / 530 eV).
2. **The $\gamma^3$ dependence** in the taper formula: confirm it is physically correct as derived (Section 2.1); it is what turns an ordinary $\Delta K$ into $\sim730$ MeV/fs at 15.7 GeV/c.
3. **Detection threshold metric.** Detection currently thresholds the *absolute* step $\Delta K \ge 4\rho$ (matching the reference notebook); should it instead threshold the *relative* $\Delta K/K$?
4. **Motion threshold.** Recommend $10^{-3}$, justified by Section 5.4.
5. **Record motion magnitude.** Have the fetch store the fractional spread $(\max-\min)/\mathrm{median}$ per bin, not just the `moved` boolean, so borderline points (the 6 in Section 5.4) can be judged and the threshold tuned without re-fetching.

**Appendix.** The full unit and smoke suite lives in [`tests/test_taper.py`](tests/test_taper.py) (`python -m pytest tests/`): physics identities, store pivots, detection, the XLEAP timeline, the motion filter, and the energy-stability gate."""))

# ============================================================================
#  Assemble
# ============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# scripts/ lives one level below the parser root, where the notebook belongs.
out = Path(__file__).resolve().parent.parent / "XLEAP_detector.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({len(cells)} cells)")
