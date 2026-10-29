"""CCCS 610 session 11, hands-on Basic stage: how three random-graph models break as vertices are removed.

Builds a small-world (Watts-Strogatz, the model in the session's example code), a preferential-attachment
(Barabasi-Albert) and a random (Erdos-Renyi G(n, m)) graph of the same order and nearly the same size, removes
vertices one at a time until none is left under two policies (uniform at random, and with probability
proportional to the current degree), and records the number of components and the order of the largest
component at every step, for 30 seeds. Writes results/m11-results.json and figures/m11-*.png.

Run:  /home/seba/.venvs/cccs610/bin/python -I code/m11_resist.py   (from the deliverable folder, or by absolute path)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display: figures are only written to files
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# -I removes the script's directory from sys.path, so the sibling helper is added explicitly.
sys.path.insert(0, str(HERE))
from legible import assert_fits, assert_legible

RESULTS = ROOT / "results" / "m11-results.json"
FIG_DIR = ROOT / "figures"

# --- Constants -------------------------------------------------------------------------------------------------
SEEDS = range(30)
N = 400  # networkx's spring_layout switches to a scipy sparse solver at 500 vertices; scipy is not installed
WS_K, WS_P = (
    4,
    0.3,
)  # the example's percolation graph G2: connected_watts_strogatz_graph(n, 4, 0.3)
BA_M = 2  # each new vertex attaches to 2 existing ones: about 2 edges per vertex, like WS with k = 4
# G(n, m) takes the edge count directly, so it is set to the Watts-Strogatz count n * k / 2.
ER_M = N * WS_K // 2
SHARES = (
    0.5,
    0.1,
)  # the largest component is "broken" below this share of the original order
AT_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5)
SNAPSHOT_SEED, SNAPSHOT_FRACTION, LAYOUT_SEED = 0, 0.3, 0
BAND = (10, 90)  # percentiles of the band in the curve figures
POLICIES = ("uniform", "degree")
FIG_WIDTH_IN, FIG_PAD_IN, FIG_DPI = 6.5, 0.05, 200
# Constrained layout fills the canvas, so the canvas is narrower than the print width: the cropped image plus its
# padding then stays within 6.5 in and no text is shrunk when the image is placed at that width.
FIG_CANVAS_IN = 6.38

# Okabe-Ito colors, distinguishable under the common color-vision deficiencies; line styles repeat the
# distinction for a grayscale print.
GENERATORS = {
    "small_world": {"label": "Small-world (WS)", "color": "#0072B2", "ls": "-"},
    "preferential_attachment": {
        "label": "Pref. attachment (BA)",
        "color": "#D55E00",
        "ls": "--",
    },
    "random": {"label": "Random (ER)", "color": "#009E73", "ls": ":"},
}
GEN_TITLE = {
    "small_world": "Small-world",
    "preferential_attachment": "Pref. attachment",
    "random": "Random",
}
POLICY_TITLE = {"uniform": "Uniform at random", "degree": "Proportional to degree"}


def build_graph(kind, seed):
    if kind == "small_world":
        return nx.connected_watts_strogatz_graph(N, WS_K, WS_P, seed=seed)
    if kind == "preferential_attachment":
        return nx.barabasi_albert_graph(N, BA_M, seed=seed)
    if kind == "random":
        return nx.gnm_random_graph(N, ER_M, seed=seed)
    raise ValueError(kind)


def simulate(graph, policy, rng):
    """Remove every vertex of a copy of graph one at a time.

    Returns the number of components, the order of the largest component (both indexed by the number of
    vertices already removed, 0..n) and the removal order.

    The degree policy draws a vertex with probability proportional to its CURRENT degree among the remaining
    vertices, recomputed after each removal: the attack a hub-targeting adversary can actually follow, since
    removals change who the hubs are. A vertex of degree 0 has probability 0, unless only isolated vertices
    remain, when the draw is uniform because the weights would otherwise all be zero.
    """
    g = graph.copy()
    n = g.number_of_nodes()
    ncomp = np.zeros(n + 1, dtype=np.int64)
    lcc = np.zeros(n + 1, dtype=np.int64)
    order = []
    for step in range(n + 1):
        sizes = [len(c) for c in nx.connected_components(g)]
        ncomp[step] = len(sizes)
        lcc[step] = max(sizes, default=0)
        if step == n:
            break
        nodes = list(g.nodes)
        if policy == "degree":
            weights = np.array([g.degree(v) for v in nodes], dtype=float)
            total = weights.sum()
            idx = (
                rng.choice(len(nodes), p=weights / total)
                if total > 0
                else rng.integers(len(nodes))
            )
        else:
            idx = rng.integers(len(nodes))
        order.append(nodes[idx])
        g.remove_node(nodes[idx])
    return ncomp, lcc, order


def first_below(lcc, n, share):
    """Fraction of vertices removed when the largest component first has fewer than share * n vertices."""
    below = lcc < share * n
    # lcc[n] == 0 is below any positive share, so a hit always exists.
    return int(np.argmax(below)) / n


def run_metrics(ncomp, lcc, n):
    peak = int(ncomp.max())
    return {
        "fraction_removed_lcc_below_50": first_below(lcc, n, 0.5),
        "fraction_removed_lcc_below_10": first_below(lcc, n, 0.1),
        "peak_components": peak,
        "fraction_removed_at_peak_components": int(np.argmax(ncomp))
        / n,  # first step reaching the peak
        # Trapezoid area of (lcc / n) against (removed / n). The largest component never exceeds the vertices
        # left, so the area is at most 1/2, reached by a graph that stays whole until the last vertex goes.
        "area_under_lcc_curve": float(np.trapezoid(lcc / n, np.arange(n + 1) / n)),
    }


def summarize(values):
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),  # sample sd across the 30 seeds
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def band(curves):
    lo, med, hi = np.percentile(curves, [BAND[0], 50, BAND[1]], axis=0)
    return lo, med, hi


def run_experiment():
    graphs_info = {
        k: {
            "edges": [],
            "largest_component_order": [],
            "components": [],
            "max_degree": [],
        }
        for k in GENERATORS
    }
    per_run = {k: {p: [] for p in POLICIES} for k in GENERATORS}
    curves = {k: {p: {"lcc": [], "ncomp": []} for p in POLICIES} for k in GENERATORS}
    snapshot = {}
    for gi, kind in enumerate(GENERATORS):
        for seed in SEEDS:
            graph = build_graph(kind, seed)
            assert graph.number_of_nodes() == N
            info = graphs_info[kind]
            info["edges"].append(graph.number_of_edges())
            comps = [len(c) for c in nx.connected_components(graph)]
            info["largest_component_order"].append(max(comps))
            info["components"].append(len(comps))
            info["max_degree"].append(max(d for _, d in graph.degree))
            for pi, policy in enumerate(POLICIES):
                # One stream per (generator, policy, seed), so no run depends on another's draws.
                rng = np.random.default_rng([seed, gi, pi])
                ncomp, lcc, order = simulate(graph, policy, rng)
                assert (
                    ncomp[N] == 0
                    and lcc[N] == 0
                    and (lcc <= N - np.arange(N + 1)).all()
                )
                per_run[kind][policy].append(run_metrics(ncomp, lcc, N))
                curves[kind][policy]["lcc"].append(lcc)
                curves[kind][policy]["ncomp"].append(ncomp)
                if seed == SNAPSHOT_SEED and policy == "degree":
                    snapshot[kind] = (graph, order)
    for kind in curves:
        for policy in POLICIES:
            for key in ("lcc", "ncomp"):
                curves[kind][policy][key] = np.array(curves[kind][policy][key])
    return graphs_info, per_run, curves, snapshot


def results_block(graphs_info, per_run, curves, snapshot):
    graphs = {}
    for kind, info in graphs_info.items():
        graphs[kind] = {
            "edges_per_seed": info["edges"],
            "edges": summarize(info["edges"]),
            "components_per_seed": info["components"],
            "largest_component_order_per_seed": info["largest_component_order"],
            "largest_component_order": summarize(info["largest_component_order"]),
            "max_degree": summarize(info["max_degree"]),
            "mean_degree_of_mean_edges": 2 * float(np.mean(info["edges"])) / N,
        }
    results = {}
    for kind in GENERATORS:
        results[kind] = {}
        for policy in POLICIES:
            runs = per_run[kind][policy]
            metrics = list(runs[0])
            at = {}
            for f in AT_FRACTIONS:
                k = round(f * N)
                at[str(f)] = {
                    "removed_vertices": k,
                    "lcc_fraction_of_original": summarize(
                        curves[kind][policy]["lcc"][:, k] / N
                    ),
                    "components": summarize(curves[kind][policy]["ncomp"][:, k]),
                }
            results[kind][policy] = {
                "per_seed": {m: [r[m] for r in runs] for m in metrics},
                "summary": {m: summarize([r[m] for r in runs]) for m in metrics},
                "at_fraction_removed": at,
            }
    # Both policies run on the same graph of each seed, so the difference is paired by seed.
    paired = {}
    for kind in GENERATORS:
        paired[kind] = {}
        for m in per_run[kind]["uniform"][0]:
            diff = [
                u[m] - d[m]
                for u, d in zip(per_run[kind]["uniform"], per_run[kind]["degree"])
            ]
            paired[kind][m] = summarize(diff)
    snaps = {}
    for kind, (graph, order) in snapshot.items():
        k = round(SNAPSHOT_FRACTION * N)
        after = graph.copy()
        after.remove_nodes_from(order[:k])
        sizes = [len(c) for c in nx.connected_components(after)]
        snaps[kind] = {
            "removed_vertices": k,
            "edges_before": graph.number_of_edges(),
            "edges_after": after.number_of_edges(),
            "max_degree_before": max(d for _, d in graph.degree),
            "max_degree_after": max(d for _, d in after.degree),
            "components_after": len(sizes),
            "largest_component_after": max(sizes),
            "isolated_vertices_after": sum(1 for _, d in after.degree if d == 0),
            "removed_max_original_degree": max(graph.degree(v) for v in order[:k]),
        }
    return graphs, results, paired, snaps


def save(fig, name):
    path = FIG_DIR / name
    assert_legible(fig, name)
    fig.canvas.draw()
    assert_fits(fig, name, FIG_WIDTH_IN, FIG_PAD_IN)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=FIG_PAD_IN)
    plt.close(fig)


def curve_figure(curves, key, ylabel, name, shared_ylim, yticks):
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_CANVAS_IN, 3.0), sharey=True, layout="constrained"
    )
    x = np.arange(N + 1) / N
    for ax, policy in zip(axes, POLICIES):
        for kind, style in GENERATORS.items():
            data = curves[kind][policy][key]
            data = data / N if key == "lcc" else data
            lo, med, hi = band(data)
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.2, linewidth=0)
            ax.plot(
                x,
                med,
                color=style["color"],
                ls=style["ls"],
                lw=1.6,
                label=style["label"],
            )
        if key == "lcc":
            for share in SHARES:
                ax.axhline(share, color="0.6", lw=0.6, zorder=0)
        ax.set_title(POLICY_TITLE[policy], fontsize=8.5)
        ax.set_xlabel("Fraction of vertices removed", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(*shared_ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel(ylabel, fontsize=8)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    save(fig, name)


def component_layout(graph):
    """spring_layout of the largest component, with every other component laid out on a ring around it.

    spring_layout with a fixed seed has a dense numpy solver that runs without scipy below 500 vertices, and
    force-directed placement shows hubs and rings, which a circular layout would hide. Its repulsion throws
    isolated vertices and small pieces far from the rest, which shrinks the largest component to a dot, so
    each small component gets its own place on a ring just outside it.
    """
    comps = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
    pos = nx.spring_layout(graph.subgraph(comps[0]), seed=LAYOUT_SEED)
    for i, comp in enumerate(comps[1:]):
        angle = 2 * np.pi * i / (len(comps) - 1)
        centre = 1.35 * np.array([np.cos(angle), np.sin(angle)])
        if len(comp) == 1:
            pos[next(iter(comp))] = centre
        else:
            sub = nx.spring_layout(graph.subgraph(comp), seed=LAYOUT_SEED, scale=0.08)
            pos.update({v: p + centre for v, p in sub.items()})
    return pos


def graphs_figure(snapshot):
    fig, axes = plt.subplots(2, 3, figsize=(FIG_CANVAS_IN, 4.4), layout="constrained")
    vmax = max(max(d for _, d in g.degree) for g, _ in snapshot.values())
    # Square-root scale: most degrees are 2 to 8 and the few hubs reach 50, so a linear scale would paint
    # every ordinary vertex the same color.
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    cmap = plt.get_cmap("viridis")  # perceptually uniform and color-blind safe
    k = round(SNAPSHOT_FRACTION * N)
    for col, kind in enumerate(GENERATORS):
        graph, order = snapshot[kind]
        pos = component_layout(graph)
        xy = np.array(list(pos.values()))
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = 0.04 * (hi - lo).max()
        removed = set(order[:k])
        kept = [v for v in graph if v not in removed]
        after = graph.subgraph(kept)
        for row, (g, title) in enumerate((
            (graph, f"{GEN_TITLE[kind]}, intact"),
            (after, f"{GEN_TITLE[kind]}, {int(SNAPSHOT_FRACTION * 100)}% removed"),
        )):
            ax = axes[row, col]
            nx.draw_networkx_edges(
                g, pos, ax=ax, edge_color="0.55", width=0.3, alpha=0.6
            )
            if row == 1:
                gone = np.array([pos[v] for v in sorted(removed)])
                ax.scatter(gone[:, 0], gone[:, 1], s=4, color="0.85", linewidths=0)
            nodes = list(g.nodes)
            xy_nodes = np.array([pos[v] for v in nodes])
            ax.scatter(
                xy_nodes[:, 0],
                xy_nodes[:, 1],
                s=6,
                linewidths=0,
                c=[g.degree(v) for v in nodes],
                cmap=cmap,
                norm=norm,
            )
            ax.set_title(title, fontsize=8)
            # Both rows of a column share one frame, so the vertices stay put when 30% are removed.
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal")
            ax.set_axis_off()
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.6,
        label="Degree (neighbors)",
        ticks=[0, 2, 5, 10, 20, 30, 40, 50],
    )
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.yaxis.label.set_size(8)
    fig.supxlabel(
        "Gray points in the lower row are the removed vertices.", fontsize=7.5
    )
    save(fig, "m11-graphs.png")


def main():
    graphs_info, per_run, curves, snapshot = run_experiment()
    graphs, results, paired, snaps = results_block(
        graphs_info, per_run, curves, snapshot
    )
    doc = {
        "meta": {
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "networkx": nx.__version__,
            "seed_first": SEEDS[0],
            "seed_last": SEEDS[-1],
            "seeds": len(SEEDS),
        },
        "constants": {
            "n": N,
            "ws_k": WS_K,
            "ws_p": WS_P,
            "ba_m": BA_M,
            "er_m": ER_M,
            "shares": list(SHARES),
            "at_fractions": list(AT_FRACTIONS),
            "snapshot_seed": SNAPSHOT_SEED,
            "snapshot_fraction": SNAPSHOT_FRACTION,
            "layout": f"networkx.spring_layout, dense numpy solver, seed {LAYOUT_SEED}",
            "curve_band_percentiles": list(BAND),
            "sd": "sample standard deviation across seeds (ddof = 1)",
        },
        "graphs": graphs,
        "results": results,
        "paired_difference_uniform_minus_degree": paired,
        "snapshot_degree_policy_seed_0": snaps,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 8})
    curve_figure(
        curves,
        "lcc",
        "Largest component (fraction of n)",
        "m11-largest.png",
        (0, 1.02),
        [0, 0.1, 0.5, 1.0],
    )
    peak = max(int(curves[k][p]["ncomp"].max()) for k in GENERATORS for p in POLICIES)
    curve_figure(
        curves,
        "ncomp",
        "Number of components",
        "m11-components.png",
        (0, peak * 1.05),
        None,
    )
    graphs_figure(snapshot)


if __name__ == "__main__":
    main()
