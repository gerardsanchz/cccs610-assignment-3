"""Session 9 (Synchrony), Basic stage: how the number of fireflies and the perception radius change synchrony.

Ports the notebook M09_sync.ipynb (Kuramoto-coupled fireflies in the unit square) into plain functions,
checks the port against a transcription of the notebook's own code, runs a grid of
5 flock sizes x 5 perception radii x 30 seeds with and without coupling, and writes
  results/m09-results.json   every number the report quotes
  figures/m09-heatmap.png, m09-timeseries.png, m09-snapshots.png

Run: /home/seba/.venvs/cccs610/bin/python -I code/m09_sync.py   (from the assignment-3 folder)
"""

import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
# -I leaves the script's own folder off sys.path, so the figure checker next to this file is added by hand.
sys.path.insert(0, str(HERE))
import legible as lg

ROOT = HERE.parent
RESULTS = ROOT / "results" / "m09-results.json"
FIGURES = ROOT / "figures"

# Constants of the notebook model.
COUPLING = 0.3
FREQ_LOW_DEG, FREQ_HIGH_DEG = 11.0, 17.0  # intrinsic frequency, degrees per step
STEPS = 200
TWO_PI = 2.0 * math.pi

# Experiment design.
COUNTS = [10, 25, 50, 100, 200]
RADII = [0.1, 0.2, 0.3, 0.4, 0.6]
SEEDS = list(range(30))
LAST_STEPS = 20  # r is averaged over the last 20 steps, t = STEPS-19 .. STEPS
SYNC_LEVEL = 0.9

# Cells drawn in the time-series figure and the two runs drawn as snapshots.
TIMESERIES_CELLS = [(10, 0.1), (50, 0.2), (50, 0.4)]
SNAPSHOT_CELLS = [(50, 0.1), (50, 0.4)]

# Figure geometry: drawn at the printed width, so a font size in code is the size on paper.
PRINT_W = 6.5
PRINT_DPI = 200
PAD = 0.04
# Okabe-Ito colours, distinguishable under the common colour-vision deficiencies.
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#555555"
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": PRINT_DPI,
})


# --- The model -------------------------------------------------------------------------------------------------


def initial_state(count, seed):
    """Positions, intrinsic frequencies (rad/step) and phases (rad) of one flock.

    One generator per (count, seed) draws positions, then frequencies, then phases, so every radius and both
    coupling settings start from the identical flock and differences between them are paired.
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((count, 2))
    freq = np.deg2rad(rng.uniform(FREQ_LOW_DEG, FREQ_HIGH_DEG, count))
    phase = np.deg2rad(rng.uniform(0.0, 360.0, count))
    return pos, freq, phase


def perception_graph(pos, radius):
    """Boolean adjacency: two flies sense each other when their distance is below the radius (strictly, as in the notebook)."""
    diff = pos[:, None, :] - pos[None, :, :]
    adj = np.sqrt((diff**2).sum(axis=2)) < radius
    np.fill_diagonal(adj, False)
    return adj


def graph_metrics(adj):
    """Largest-component share, isolated-fly count and mean degree of a perception graph."""
    count = adj.shape[0]
    label = np.full(count, -1)
    sizes = []
    for start in range(count):
        if label[start] >= 0:
            continue
        label[start] = len(sizes)
        stack, size = [start], 0
        while stack:
            node = stack.pop()
            size += 1
            for other in np.flatnonzero(adj[node] & (label < 0)):
                label[other] = label[start]
                stack.append(other)
        sizes.append(size)
    degree = adj.sum(axis=1)
    return {
        "lcc_share": max(sizes) / count,
        "n_components": len(sizes),
        "n_isolated": int((degree == 0).sum()),
        "mean_degree": float(degree.mean()),
    }


def order_parameter(phase):
    """Kuramoto order parameter r = |mean(exp(i*phase))|: 1 when all phases agree, about 1/sqrt(n) by chance."""
    return float(np.abs(np.exp(1j * phase).mean()))


def phase_increment(phase, freq, adj_float, degree, coupling):
    """Per-fly phase change for one step, all flies computed from the same phases.

    sin(other - own) summed over the neighbours is cos(own)*sum(sin) - sin(own)*sum(cos), a matrix-vector product.
    An isolated fly has an empty sum; its divisor is held at 1 so it keeps its own frequency (the notebook
    divides by a neighbour count that does not exist for it).
    """
    sine, cosine = np.sin(phase), np.cos(phase)
    effect = cosine * (adj_float @ sine) - sine * (adj_float @ cosine)
    return freq + coupling * effect / np.maximum(degree, 1.0)


def simulate(freq, phase0, adj, coupling, steps=STEPS):
    """Order parameter after each of 0..steps updates, and the final phases (synchronous update)."""
    adj_float = adj.astype(float)
    degree = adj_float.sum(axis=1)
    phase = phase0.copy()
    r = np.empty(steps + 1)
    r[0] = order_parameter(phase)
    for t in range(steps):
        phase = np.mod(
            phase + phase_increment(phase, freq, adj_float, degree, coupling), TWO_PI
        )
        r[t + 1] = order_parameter(phase)
    return r, phase


def notebook_wrap(angle):
    """The notebook's `rr`: a single add-back and a strict `>`, kept to count where it misbehaves."""
    while angle > TWO_PI:
        angle -= TWO_PI
    if angle < 0:
        angle += TWO_PI
    return angle


def simulate_sequential(freq, phase0, adj, coupling, steps=STEPS):
    """The notebook's update order: fly 0, 1, ... update in place, so later flies see earlier flies' new phases.

    Returns r after each step, the final phases and how many wrapped phases fell outside [0, 2pi).
    """
    neighbours = [np.flatnonzero(row) for row in adj]
    phase = phase0.copy()
    r = np.empty(steps + 1)
    r[0] = order_parameter(phase)
    out_of_range = 0
    for t in range(steps):
        for fly, nb in enumerate(neighbours):
            move = freq[fly]
            if nb.size:
                move += coupling / nb.size * np.sin(phase[nb] - phase[fly]).sum()
            value = notebook_wrap(phase[fly] + move)
            out_of_range += int(value < 0.0 or value >= TWO_PI)
            phase[fly] = value
        r[t + 1] = order_parameter(phase)
    return r, phase, out_of_range


def first_step_at(r, level=SYNC_LEVEL):
    """First step t (0 = before any update) with r >= level, or None."""
    hit = np.flatnonzero(r >= level)
    return int(hit[0]) if hit.size else None


# --- Checks of the port against the notebook's own code --------------------------------------------------------


class NotebookModel:
    """A transcription of the notebook's neighbors / buddies / kuramoto / sync cells in plain Python, as an oracle."""

    def __init__(self, pos, freq, phase, visual, coupling):
        count = len(freq)
        self.neighbors = defaultdict(set)
        for f1 in range(count):
            for f2 in range(f1 + 1, count):
                if math.dist(pos[f1], pos[f2]) < visual:
                    self.neighbors[f1].add(f2)
                    self.neighbors[f2].add(f1)
        self.buddies = {fly: len(self.neighbors[fly]) for fly in list(self.neighbors)}
        self.rf = [float(v) for v in freq]
        self.rp = [float(v) for v in phase]
        self.coupling = coupling
        self.count = count

    def kuramoto(self, fly):
        effect = sum(math.sin(self.rp[o] - self.rp[fly]) for o in self.neighbors[fly])
        return self.rf[fly] + (self.coupling / self.buddies[fly]) * effect

    def sync_step(self):
        for fly in range(self.count):
            self.rp[fly] = notebook_wrap(self.rp[fly] + self.kuramoto(fly))


def check_against_notebook():
    """Compare the port with the transcribed notebook on the same flock.

    Increment check: the notebook's kuramoto(fly) for every fly against the port's vector increment from the
    same phases. A fly without neighbours raises KeyError in the notebook (reproduced and counted); the port
    must return its own frequency there.
    Order check: 200 in-place notebook steps against the port's sequential update, final phases compared.
    """
    increment_cases, max_increment_diff = [], 0.0
    keyerrors = []
    for count, visual, seed in [
        (50, 0.4, 0),
        (100, 0.3, 1),
        (200, 0.6, 2),
        (25, 0.1, 0),
        (10, 0.1, 0),
    ]:
        pos, freq, phase = initial_state(count, seed)
        adj = perception_graph(pos, visual)
        model = NotebookModel(pos, freq, phase, visual, COUPLING)
        mine = phase_increment(
            phase, freq, adj.astype(float), adj.sum(axis=1).astype(float), COUPLING
        )
        compared = 0
        for fly in range(count):
            try:
                theirs = model.kuramoto(fly)
            except KeyError as err:
                keyerrors.append({
                    "count": count,
                    "radius": visual,
                    "seed": seed,
                    "fly": fly,
                    "key": int(err.args[0]),
                })
                assert mine[fly] == freq[fly], (
                    "isolated fly must keep its own frequency"
                )
                continue
            max_increment_diff = max(max_increment_diff, abs(theirs - float(mine[fly])))
            compared += 1
        increment_cases.append({
            "count": count,
            "radius": visual,
            "seed": seed,
            "flies_compared": compared,
        })
    assert max_increment_diff < 1e-12, (
        f"port differs from the notebook increment by {max_increment_diff}"
    )

    order_cases, max_phase_diff = [], 0.0
    for count, visual, seed in [(50, 0.4, 0), (100, 0.3, 1)]:
        pos, freq, phase = initial_state(count, seed)
        adj = perception_graph(pos, visual)
        assert graph_metrics(adj)["n_isolated"] == 0, (
            "order check needs a flock the notebook can run"
        )
        model = NotebookModel(pos, freq, phase, visual, COUPLING)
        for _ in range(STEPS):
            model.sync_step()
        _, final, _ = simulate_sequential(freq, phase, adj, COUPLING)
        diff = float(np.abs(np.array(model.rp) - final).max())
        max_phase_diff = max(max_phase_diff, diff)
        order_cases.append({
            "count": count,
            "radius": visual,
            "seed": seed,
            "max_abs_final_phase_diff": diff,
        })
    assert max_phase_diff < 1e-8, (
        f"sequential port differs from the notebook by {max_phase_diff}"
    )
    return {
        "increment_cases": increment_cases,
        "max_abs_increment_diff": max_increment_diff,
        "order_cases": order_cases,
        "max_abs_final_phase_diff_sequential": max_phase_diff,
        "keyerror_reproductions": keyerrors,
        "restore_examples": {
            "restore_360": notebook_restore_degrees(360),
            "restore_720": notebook_restore_degrees(720),
            "restore_minus_400": notebook_restore_degrees(-400),
        },
    }


def notebook_restore_degrees(angle):
    """The notebook's `restore`, verbatim: `>` instead of `>=`, and a single add-back."""
    while angle > 360:
        angle -= 360
    if angle < 0:
        angle += 360
    return angle


# --- Statistics ---------------------------------------------------------------------------------------------------


def average_ranks(values):
    """Ranks 1..n with ties given their mean rank (no scipy in the environment)."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(average_ranks(a), average_ranks(b))


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}


def run_cell(count, radius):
    """30 paired runs (coupled and no-coupling control) of one count x radius cell."""
    rows, coupled_r, control_r = [], [], []
    for seed in SEEDS:
        pos, freq, phase = initial_state(count, seed)
        adj = perception_graph(pos, radius)
        graph = graph_metrics(adj)
        r_on, _ = simulate(freq, phase, adj, COUPLING)
        r_off, _ = simulate(freq, phase, adj, 0.0)
        coupled_r.append(r_on)
        control_r.append(r_off)
        rows.append({
            "count": count,
            "radius": radius,
            "seed": seed,
            "r_final": float(r_on[-LAST_STEPS:].mean()),
            "first_step_r_ge_0_9": first_step_at(r_on),
            "r_final_control": float(r_off[-LAST_STEPS:].mean()),
            "first_step_r_ge_0_9_control": first_step_at(r_off),
            "lcc_share": graph["lcc_share"],
            "n_components": graph["n_components"],
            "n_isolated": graph["n_isolated"],
            "mean_degree": graph["mean_degree"],
        })
    return rows, np.array(coupled_r), np.array(control_r)


def cell_summary(rows, count, radius):
    def block(key_r, key_t):
        r = [row[key_r] for row in rows]
        t = [row[key_t] for row in rows if row[key_t] is not None]
        return {
            "r_final": describe(r),
            "share_runs_reaching_r_0_9": len(t) / len(rows),
            "n_runs_reaching_r_0_9": len(t),
            "median_first_step_r_0_9_among_reached": float(np.median(t)) if t else None,
            "share_runs_final_r_ge_0_9": float(np.mean([v >= SYNC_LEVEL for v in r])),
        }

    return {
        "count": count,
        "radius": radius,
        "coupled": block("r_final", "first_step_r_ge_0_9"),
        "control": block("r_final_control", "first_step_r_ge_0_9_control"),
        "chance_reference_1_over_sqrt_n": 1.0 / math.sqrt(count),
        "gain_mean_r_coupled_minus_control": float(
            np.mean([row["r_final"] - row["r_final_control"] for row in rows])
        ),
        "lcc_share": describe([row["lcc_share"] for row in rows]),
        "mean_isolated_flies": float(np.mean([row["n_isolated"] for row in rows])),
        "n_runs_with_isolated_fly": int(sum(row["n_isolated"] > 0 for row in rows)),
        "n_runs_connected": int(sum(row["lcc_share"] == 1.0 for row in rows)),
        "mean_degree": float(np.mean([row["mean_degree"] for row in rows])),
    }


def connectivity_analysis(runs):
    """Pooled association between final r and largest-component share, with the confounders measured beside it."""
    share = np.array([r["lcc_share"] for r in runs])
    r_on = np.array([r["r_final"] for r in runs])
    r_off = np.array([r["r_final_control"] for r in runs])
    degree = np.array([r["mean_degree"] for r in runs])
    counts = np.array([r["count"] for r in runs])
    radii = np.array([r["radius"] for r in runs])

    # Centre both variables inside each count x radius cell: what is left is variation between seeds only.
    share_c, r_c = share.copy(), r_on.copy()
    for n in COUNTS:
        for rad in RADII:
            m = (counts == n) & (radii == rad)
            share_c[m] -= share[m].mean()
            r_c[m] -= r_on[m].mean()
    within = pearson(share_c, r_c) if share_c.std() > 0 and r_c.std() > 0 else None

    connected = share == 1.0
    return {
        "n_runs": len(runs),
        "pearson_r_vs_lcc_share": pearson(r_on, share),
        "spearman_r_vs_lcc_share": spearman(r_on, share),
        "r_squared_pearson": pearson(r_on, share) ** 2,
        "pearson_r_control_vs_lcc_share": pearson(r_off, share),
        "spearman_r_control_vs_lcc_share": spearman(r_off, share),
        "pearson_gain_vs_lcc_share": pearson(r_on - r_off, share),
        "spearman_gain_vs_lcc_share": spearman(r_on - r_off, share),
        "pearson_r_vs_mean_degree": pearson(r_on, degree),
        "spearman_r_vs_mean_degree": spearman(r_on, degree),
        "pearson_lcc_share_vs_mean_degree": pearson(share, degree),
        "pearson_within_cell_r_vs_lcc_share": within,
        "n_runs_fully_connected": int(connected.sum()),
        "n_runs_not_connected": int((~connected).sum()),
        "mean_r_fully_connected": float(r_on[connected].mean()),
        "mean_r_not_connected": float(r_on[~connected].mean()),
        "n_fully_connected_with_final_r_below_0_9": int(
            (connected & (r_on < SYNC_LEVEL)).sum()
        ),
        "n_fully_connected_with_final_r_at_least_0_9": int(
            (connected & (r_on >= SYNC_LEVEL)).sum()
        ),
        "n_not_connected_with_final_r_at_least_0_9": int(
            (~connected & (r_on >= SYNC_LEVEL)).sum()
        ),
        "n_not_connected_with_final_r_below_0_9": int(
            (~connected & (r_on < SYNC_LEVEL)).sum()
        ),
        "min_r_among_fully_connected": float(r_on[connected].min()),
        "max_r_among_not_connected": float(r_on[~connected].max()),
    }


def update_order_sensitivity():
    """Does the notebook's in-place update order change the cell means? Coupled runs only, counts 10 and 50."""
    cells, out_of_range = [], 0
    for count in (10, 50):
        for radius in RADII:
            diffs, sync_r, seq_r = [], [], []
            for seed in SEEDS:
                pos, freq, phase = initial_state(count, seed)
                adj = perception_graph(pos, radius)
                r_sync, _ = simulate(freq, phase, adj, COUPLING)
                r_seq, _, bad = simulate_sequential(freq, phase, adj, COUPLING)
                out_of_range += bad
                sync_r.append(r_sync[-LAST_STEPS:].mean())
                seq_r.append(r_seq[-LAST_STEPS:].mean())
                diffs.append(abs(sync_r[-1] - seq_r[-1]))
            cells.append({
                "count": count,
                "radius": radius,
                "mean_r_synchronous": float(np.mean(sync_r)),
                "mean_r_sequential": float(np.mean(seq_r)),
                "max_abs_run_difference": float(max(diffs)),
            })
    return {
        "cells": cells,
        "max_abs_cell_mean_difference": max(
            abs(c["mean_r_synchronous"] - c["mean_r_sequential"]) for c in cells
        ),
        "max_abs_run_difference": max(c["max_abs_run_difference"] for c in cells),
        "notebook_wrap_results_outside_0_2pi": out_of_range,
    }


# --- Figures ------------------------------------------------------------------------------------------------------


def save_figure(fig, name):
    lg.assert_legible(fig, name)
    lg.assert_fits(fig, name, PRINT_W, PAD)
    fig.savefig(FIGURES / name, dpi=PRINT_DPI, bbox_inches="tight", pad_inches=PAD)
    plt.close(fig)


def draw_heatmap(ax, grid, title, cmap):
    image = ax.imshow(
        grid, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto"
    )
    ax.set_xticks(range(len(RADII)), [f"{v:g}" for v in RADII])
    ax.set_yticks(range(len(COUNTS)), [str(v) for v in COUNTS])
    ax.set_xlabel("Perception radius (unit square)")
    ax.set_title(title)
    for i in range(len(COUNTS)):
        for j in range(len(RADII)):
            rgba = cmap(grid[i, j])
            dark = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2] < 0.45
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if dark else "black",
                fontsize=7.5,
            )
    return image


def figure_heatmap(cells):
    on = np.array([
        [
            next(c for c in cells if c["count"] == n and c["radius"] == r)["coupled"][
                "r_final"
            ]["mean"]
            for r in RADII
        ]
        for n in COUNTS
    ])
    off = np.array([
        [
            next(c for c in cells if c["count"] == n and c["radius"] == r)["control"][
                "r_final"
            ]["mean"]
            for r in RADII
        ]
        for n in COUNTS
    ])
    fig, axes = plt.subplots(
        1, 2, figsize=(PRINT_W, 2.7), sharey=True, layout="constrained"
    )
    cmap = matplotlib.colormaps["viridis"]
    image = draw_heatmap(axes[0], on, "Coupled (coupling 0.3)", cmap)
    draw_heatmap(axes[1], off, "No coupling (radius has no effect)", cmap)
    axes[0].set_ylabel("Number of fireflies")
    fig.colorbar(
        image,
        ax=axes,
        label="Mean final r (last 20 steps, 30 seeds)",
        shrink=0.9,
        pad=0.02,
    )
    save_figure(fig, "m09-heatmap.png")


def figure_timeseries(coupled_r, control_r):
    fig, axes = plt.subplots(
        1, 3, figsize=(PRINT_W, 2.5), sharey=True, layout="constrained"
    )
    steps = np.arange(STEPS + 1)
    colours = [VERMILLION, GREEN, BLUE]
    for ax, cell, colour in zip(axes, TIMESERIES_CELLS, colours):
        lo, mid, hi = np.percentile(coupled_r[cell], [10, 50, 90], axis=0)
        ax.fill_between(steps, lo, hi, color=colour, alpha=0.25, linewidth=0)
        ax.plot(steps, mid, color=colour, linewidth=1.6)
        ax.plot(
            steps,
            np.median(control_r[cell], axis=0),
            color=GREY,
            linewidth=1.0,
            linestyle="--",
        )
        ax.axhline(SYNC_LEVEL, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(f"{cell[0]} flies, radius {cell[1]:g}")
        ax.set_xlabel("Step")
        ax.set_xlim(0, STEPS)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Order parameter r")
    handles = [
        Line2D([], [], color="black", linewidth=1.6, label="Coupled median"),
        Patch(facecolor="black", alpha=0.25, label="10-90% band over 30 seeds"),
        Line2D(
            [],
            [],
            color=GREY,
            linewidth=1.0,
            linestyle="--",
            label="No-coupling median",
        ),
        Line2D([], [], color="black", linewidth=0.6, linestyle=":", label="r = 0.9"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save_figure(fig, "m09-timeseries.png")


def representative_seed(rows, count, radius):
    """The seed whose final r is closest to the cell median, so the snapshot shows a typical run, not an extreme."""
    cell = [r for r in rows if r["count"] == count and r["radius"] == radius]
    median = float(np.median([r["r_final"] for r in cell]))
    return min(cell, key=lambda r: (abs(r["r_final"] - median), r["seed"]))["seed"]


def figure_snapshots(rows):
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_W, 3.5), layout="constrained")
    cmap = matplotlib.colormaps[
        "twilight"
    ]  # cyclic: phase 0 and 360 degrees share a colour
    chosen = []
    image = None
    for ax, (count, radius), label in zip(
        axes, SNAPSHOT_CELLS, ["Disconnected", "Connected"]
    ):
        seed = representative_seed(rows, count, radius)
        pos, freq, phase = initial_state(count, seed)
        adj = perception_graph(pos, radius)
        graph = graph_metrics(adj)
        r_trace, final = simulate(freq, phase, adj, COUPLING)
        i, j = np.nonzero(np.triu(adj))
        ax.add_collection(
            LineCollection(
                np.stack([pos[i], pos[j]], axis=1),
                colors="#999999",
                linewidths=0.5,
                alpha=0.5,
                zorder=1,
            )
        )
        isolated = adj.sum(axis=1) == 0
        for mask, marker, size in ((~isolated, "o", 26), (isolated, "D", 34)):
            image = ax.scatter(
                pos[mask, 0],
                pos[mask, 1],
                c=np.rad2deg(final[mask]),
                cmap=cmap,
                vmin=0,
                vmax=360,
                marker=marker,
                s=size,
                edgecolors="black",
                linewidths=0.7,
                zorder=2,
            )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(
            f"{label}: {count} flies, radius {radius:g}, seed {seed}\n"
            f"final r = {r_trace[-LAST_STEPS:].mean():.2f}, largest component {graph['lcc_share']:.0%}, "
            f"{graph['n_isolated']} isolated"
        )
        chosen.append({
            "count": count,
            "radius": radius,
            "seed": seed,
            "r_final": float(r_trace[-LAST_STEPS:].mean()),
            **graph,
        })
    fig.colorbar(
        image,
        ax=axes,
        label="Final phase (degrees)",
        ticks=[0, 90, 180, 270, 360],
        shrink=0.75,
        pad=0.02,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Fly with at least one neighbor",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Isolated fly",
        ),
        Line2D([], [], color="#999999", linewidth=0.8, label="Perception edge"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, "m09-snapshots.png")
    return chosen


# --- Main ---------------------------------------------------------------------------------------------------------


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.parent.mkdir(exist_ok=True)

    parity = check_against_notebook()

    runs, cells = [], []
    coupled_r, control_r = {}, {}
    for count in COUNTS:
        for radius in RADII:
            rows, r_on, r_off = run_cell(count, radius)
            runs.extend(rows)
            coupled_r[(count, radius)], control_r[(count, radius)] = r_on, r_off
            cells.append(cell_summary(rows, count, radius))

    connectivity = connectivity_analysis(runs)
    sensitivity = update_order_sensitivity()

    figure_heatmap(cells)
    figure_timeseries(coupled_r, control_r)
    snapshots = figure_snapshots(runs)

    timeseries = []
    for cell in TIMESERIES_CELLS:
        lo, mid, hi = np.percentile(coupled_r[cell], [10, 50, 90], axis=0)
        timeseries.append({
            "count": cell[0],
            "radius": cell[1],
            "coupled_median": mid.tolist(),
            "coupled_p10": lo.tolist(),
            "coupled_p90": hi.tolist(),
            "control_median": np.median(control_r[cell], axis=0).tolist(),
        })

    isolated_by_radius = {
        f"{radius:g}": {
            "n_runs": sum(r["radius"] == radius for r in runs),
            "n_runs_with_isolated_fly": sum(
                r["radius"] == radius and r["n_isolated"] > 0 for r in runs
            ),
        }
        for radius in RADII
    }
    results = {
        "constants": {
            "coupling": COUPLING,
            "intrinsic_frequency_deg_per_step": [FREQ_LOW_DEG, FREQ_HIGH_DEG],
            "steps": STEPS,
            "counts": COUNTS,
            "radii": RADII,
            "seeds": [SEEDS[0], SEEDS[-1]],
            "n_seeds_per_cell": len(SEEDS),
            "n_runs_per_coupling_setting": len(runs),
            "r_averaged_over_last_steps": LAST_STEPS,
            "sync_level": SYNC_LEVEL,
            "step_index": "step 0 is the initial state; step t is after t updates",
            "update": "synchronous (all flies from the same phases)",
            "snapshot_rule": "per snapshot cell, the seed whose final r is closest to the cell median (ties to the lower seed)",
            "control_note": "the no-coupling control ignores the graph, so for one count and seed it is identical at every radius",
            "neighbor_rule": "distance < radius, positions fixed, uniform random in the unit square",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "notebook_checks": parity,
        "notebook_defects": {
            "runs_where_notebook_raises_keyerror_isolated_fly": sum(
                r["n_isolated"] > 0 for r in runs
            ),
            "runs_total": len(runs),
            "by_radius": isolated_by_radius,
            "update_order_sensitivity": sensitivity,
        },
        "cells": cells,
        "connectivity": connectivity,
        "timeseries_cells": timeseries,
        "snapshots": snapshots,
        "runs": {key: [r[key] for r in runs] for key in runs[0]},
    }
    with RESULTS.open("w") as handle:
        json.dump(results, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
