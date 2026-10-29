"""CCCS 610 session 10: the notebook's S-I-R epidemic on a scale-free network, with waning natural immunity.

A recovered vertex returns to susceptible each step with probability w (mean immunity 1/w steps), so
reinfections are possible; w = 0 is the notebook's class model. The script runs 30 seeds for each w, writes
results/m10-results.json (every number the report quotes) and three figures in figures/m10-*.png.

Run:  /home/seba/.venvs/cccs610/bin/python -I code/m10_epidemic.py     (from the assignment-3 folder)
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from legible import assert_fits, assert_legible

# Contagion and recovery rules and their probabilities are the notebook's own (its 200,000-vertex run).
ATTACH = 5  # connections each arriving vertex forms (barabasi_albert_graph m)
BETA = 0.15  # chance per step that one infected neighbor infects a susceptible vertex
GAMMA = 0.3  # chance per step that an infected vertex (not newly infected) recovers
N_BIG = 200_000  # the notebook's network
SEEDS_BIG = 20  # the notebook's round(0.0001 * 200000) initial infections
N_SMALL = 5_000  # network for the experiments: 40x smaller, so 180 runs finish in about a minute
SEEDS_SMALL = 2  # initial infections at N_SMALL: the notebook's 0.01% of 5,000 rounds to 0, and one
# vertex fizzles out in 4 of 30 trials, so two is the smallest count that always takes off
N_SENS = 20_000  # a larger network, used only to see how extinction depends on size
SENS_RUNS = 10
W_VALUES = [0.0, 0.2, 0.1, 0.05, 0.02, 0.01]
N_RUNS = 30  # seeds 0..29 per w; one seed fixes the graph, the initial infections and every random draw
# A run stops when no vertex is infected, or after MAX_STEPS. 1000 steps is ten mean immunity durations
# for the slowest nonzero w (1/0.01 = 100), so the first wave and its aftermath are over before the
# endemic window; the notebook's cap of 100 steps would cut that case off after one duration.
MAX_STEPS = 1000
NOTEBOOK_CAP = 100
ENDEMIC_WINDOW = (
    100  # the endemic level is the mean infected share over the last 100 steps
)
# Wave rule: a wave is a peak of the infected share of at least WAVE_MIN that is followed by a fall to
# at most WAVE_DROP (half) of its height, or by the end of the run. After a wave the share must rise
# by at least a factor 1/WAVE_DROP from its lowest point, and reach WAVE_MIN, before the next wave
# starts, so jitter of under a factor of two is not a wave. Larger swings on a low plateau still are,
# and the count depends on WAVE_MIN, so it is also reported at each WAVE_MIN_SENSITIVITY value.
# WAVE_MIN is 5% (one vertex in twenty infected at once), above the plateau of every w but the two
# smallest, where only the swings that reach 5% count.
WAVE_MIN = 0.05
WAVE_MIN_SENSITIVITY = [0.01, 0.02, 0.05, 0.10]
WAVE_DROP = 0.5
LOOP_RUNS = 30  # seeds compared against the notebook's loop code
STEPS_REPORTED = [0, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000]
FIG_W = 6.5  # inches: drawn at print width so font sizes are the printed sizes
PAD = 0.04
DPI = 200

# Okabe-Ito, distinguishable under the common forms of colour blindness.
COLOR = {
    0.0: "#000000",
    0.01: "#0072B2",
    0.02: "#56B4E9",
    0.05: "#009E73",
    0.1: "#E69F00",
    0.2: "#D55E00",
}


def make_graph(n, seed):
    """The notebook's generator (preferential attachment); an integer seed makes it reproducible."""
    return nx.barabasi_albert_graph(n, ATTACH, seed=seed)


def edge_arrays(graph):
    """Both directions of every edge, so a bincount over destinations counts infected neighbors."""
    e = np.array(graph.edges(), dtype=np.int64)
    return np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])


def simulate(n, src, dst, w, seed, n_seeds, max_steps):
    """One S-I-R-S run. Returns infected counts per step (0 after extinction), infections per vertex, and
    the step at which no vertex was infected (-1 when the run reached max_steps with infection present).

    States: 0 susceptible, 1 infected, 2 recovered (immune). Each step, in the notebook's order:
    contagion (a susceptible vertex with k infected neighbors is infected with probability 1-(1-BETA)^k,
    the same as one independent BETA draw per infected neighbor), recovery (infected vertices that were
    not infected this step), then waning (vertices already recovered at the start of the step). All
    three read the state at the start of the step, so a vertex never moves twice in one step.
    The three uniform draws per vertex come from one stream per seed, whatever w is, so runs that
    differ only in w share their random numbers.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=np.int8)
    first = rng.choice(n, n_seeds, replace=False)
    state[first] = 1
    infections = np.zeros(n, dtype=np.int64)
    infections[first] = 1  # the initial infection is a first infection
    counts = np.zeros(max_steps + 1, dtype=np.int64)
    counts[0] = n_seeds
    extinct_at = -1
    for t in range(1, max_steps + 1):
        infected = state == 1
        recovered = state == 2
        k = np.bincount(dst[infected[src]], minlength=n)
        u = rng.random((3, n))
        new = (state == 0) & (u[0] < 1.0 - (1.0 - BETA) ** k)
        recover = infected & (u[1] < GAMMA)
        wane = recovered & (u[2] < w)
        state[new] = 1
        state[recover] = 2
        state[wane] = 0
        infections[new] += 1
        counts[t] = int((state == 1).sum())
        if counts[t] == 0:
            extinct_at = t
            break
    return counts, infections, extinct_at


def count_waves(share, wave_min):
    """Number of waves in an infected-share series under the wave rule at wave_min and WAVE_DROP."""
    waves = 0
    in_wave = False
    peak = 0.0
    trough = float(share[0])
    for s in share:
        s = float(s)
        if in_wave:
            if s > peak:
                peak = s
            elif s <= WAVE_DROP * peak:
                waves += 1
                in_wave = False
                trough = s
        else:
            trough = min(trough, s)
            if s >= wave_min and s >= trough / WAVE_DROP:
                in_wave = True
                peak = s
    if in_wave:
        waves += 1
    return waves


def measures(counts, infections, extinct_at, n):
    share = counts / n
    out = {
        "peak_share": float(share.max()),
        "peak_step": int(share.argmax()),
        "infections_per_vertex": float(infections.mean()),
        "share_infected_twice_or_more": float((infections >= 2).mean()),
        "waves": count_waves(share, WAVE_MIN),
        "endemic_level": float(share[-ENDEMIC_WINDOW:].mean()),
        "extinct": extinct_at >= 0,
        "extinct_step": extinct_at,
    }
    for m in WAVE_MIN_SENSITIVITY:
        out[f"waves_if_min_share_{m:g}"] = count_waves(share, m)
    return out


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size)}


def summarize(runs):
    """Mean and sd of each measure over runs; extinction as a share, and the step among extinct runs."""
    out = {}
    for key in (
        "peak_share",
        "peak_step",
        "infections_per_vertex",
        "share_infected_twice_or_more",
        "waves",
        "endemic_level",
        *[f"waves_if_min_share_{m:g}" for m in WAVE_MIN_SENSITIVITY],
    ):
        out[key] = mean_sd([r[key] for r in runs])
    flags = [r["extinct"] for r in runs]
    out["share_extinct"] = float(np.mean(flags))
    out["n_extinct"] = int(np.sum(flags))
    steps = [r["extinct_step"] for r in runs if r["extinct"]]
    out["extinct_step_among_extinct"] = mean_sd(steps) if len(steps) > 1 else None
    out["endemic_level_among_surviving"] = (
        mean_sd([r["endemic_level"] for r in runs if not r["extinct"]])
        if sum(not f for f in flags) > 1
        else None
    )
    return out


def notebook_loop_model(graph, n_seeds, seed, max_steps):
    """The notebook's contagions() and recoveries() on Python sets, for checking simulate() at w = 0."""
    rnd = random.Random(seed)
    infected = set(rnd.sample(list(graph.nodes), n_seeds))
    recovered = set()
    newly = set()
    counts = [len(infected)]
    ever = set(infected)
    permitted = max_steps
    while len(infected) > 0 and permitted > 0:
        for v in infected:
            for u in graph.neighbors(v):
                if u in infected or u in recovered:
                    continue
                if rnd.random() < BETA:
                    newly.add(u)
        infected.update(newly)
        ever.update(newly)
        for v in infected:
            if v not in newly and rnd.random() < GAMMA:
                recovered.add(v)
        infected -= recovered
        newly = set()
        counts.append(len(infected))
        permitted -= 1
    return counts, len(ever)


def size_check():
    """Class model (w = 0) at the notebook's size with seed 0, against the experiment size."""
    graph = make_graph(N_BIG, 0)
    src, dst = edge_arrays(graph)
    counts, inf, ext = simulate(N_BIG, src, dst, 0.0, 0, SEEDS_BIG, NOTEBOOK_CAP)
    out = {"notebook_size": describe_curve(counts, inf, ext, N_BIG, SEEDS_BIG, 0)}
    graph = make_graph(N_SMALL, 0)
    src, dst = edge_arrays(graph)
    for n_seeds in (SEEDS_SMALL, SEEDS_BIG):
        counts, inf, ext = simulate(N_SMALL, src, dst, 0.0, 0, n_seeds, NOTEBOOK_CAP)
        out[f"experiment_size_{n_seeds}_initial"] = describe_curve(
            counts, inf, ext, N_SMALL, n_seeds, 0
        )
    return out


def describe_curve(counts, infections, extinct_at, n, n_seeds, seed):
    share = counts / n
    return {
        "n": n,
        "initial_infected": n_seeds,
        "seed": seed,
        "peak_share": float(share.max()),
        "peak_step": int(share.argmax()),
        "extinct_step": extinct_at,
        "infections_per_vertex": float(infections.mean()),
        "infected_share_steps_0_to_30": [float(x) for x in share[:31]],
    }


def loop_check(per_w_runs):
    """Notebook loop code against simulate() at w = 0, same graphs, independent random streams."""
    loop_peak, loop_step, loop_final = [], [], []
    vec_peak, vec_step, vec_final = [], [], []
    for seed in range(LOOP_RUNS):
        graph = make_graph(N_SMALL, seed)
        counts, ever = notebook_loop_model(graph, SEEDS_SMALL, seed, NOTEBOOK_CAP)
        share = np.array(counts) / N_SMALL
        loop_peak.append(float(share.max()))
        loop_step.append(int(share.argmax()))
        loop_final.append(ever / N_SMALL)
        run = per_w_runs[0.0][seed]
        vec_peak.append(run["peak_share"])
        vec_step.append(run["peak_step"])
        vec_final.append(run["infections_per_vertex"])
    return {
        "n": N_SMALL,
        "initial_infected": SEEDS_SMALL,
        "seeds": [0, LOOP_RUNS - 1],
        "loop_peak_share": mean_sd(loop_peak),
        "vectorized_peak_share": mean_sd(vec_peak),
        "loop_peak_step": mean_sd(loop_step),
        "vectorized_peak_step": mean_sd(vec_step),
        "loop_share_ever_infected": mean_sd(loop_final),
        "vectorized_share_ever_infected": mean_sd(vec_final),
    }


def run_grid(n, seeds, max_steps):
    """All runs of W_VALUES x seeds on one graph per seed; returns measures, infected-share curves and the
    per-vertex infection histograms pooled over seeds."""
    runs = {w: [] for w in W_VALUES}
    curves = {w: [] for w in W_VALUES}
    hist = dict.fromkeys(W_VALUES)
    for seed in seeds:
        graph = make_graph(n, seed)
        src, dst = edge_arrays(graph)
        for w in W_VALUES:
            counts, inf, ext = simulate(n, src, dst, w, seed, SEEDS_SMALL, max_steps)
            runs[w].append(measures(counts, inf, ext, n))
            curves[w].append(counts / n)
            h = np.bincount(inf)
            if hist[w] is None:
                hist[w] = h
            else:
                size = max(len(h), len(hist[w]))
                hist[w] = np.pad(hist[w], (0, size - len(hist[w]))) + np.pad(
                    h, (0, size - len(h))
                )
        print(f"  n={n} seed {seed} done", flush=True)
    return runs, {w: np.array(c) for w, c in curves.items()}, hist


def band(curves):
    return {
        q: np.percentile(curves, q, axis=0) for q in (10, 50, 90)
    }  # across seeds, at each step


def style():
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
    })


def save(fig, name):
    assert_legible(fig, name)
    assert_fits(fig, name, FIG_W, PAD)
    fig.savefig(ROOT / "figures" / name, dpi=DPI, bbox_inches="tight", pad_inches=PAD)
    plt.close(fig)


def figure_timeseries(curves):
    # Left: the first wave in detail (steps 0-100). Right: the whole run, where the plateau shows.
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, 3.0), sharey=True, constrained_layout=True
    )
    steps = np.arange(MAX_STEPS + 1)
    for ax, last in zip(axes, (100, MAX_STEPS)):
        for w in W_VALUES:
            b = band(curves[w])
            label = "0 (no waning)" if w == 0 else f"{w:g}"
            ax.fill_between(
                steps, b[10], b[90], color=COLOR[w], alpha=0.25, linewidth=0
            )
            ax.plot(steps, b[50], color=COLOR[w], label=f"w = {label}")
        ax.set_xlim(0, last)
        ax.set_ylim(0, 0.52)
        ax.set_xlabel("Step")
    axes[0].set_ylabel("Infected share of vertices")
    axes[0].set_title("Steps 0-100", loc="left")
    axes[1].set_title("Steps 0-1000", loc="left")
    axes[1].legend(loc="upper right", frameon=False, ncols=2)
    save(fig, "m10-timeseries.png")


def figure_summary(runs, summary):
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 4.0), constrained_layout=True)
    panels = [
        ("peak_share", "Peak infected share"),
        ("peak_step", "Step of the peak"),
        ("endemic_level", "Endemic level (steps 901-1000)"),
        ("waves", "Waves of 5% or more per run"),
        ("infections_per_vertex", "Infections per vertex"),
        ("share_extinct", "Share of runs extinct"),
    ]
    pos = np.arange(len(W_VALUES))
    jitter = np.linspace(-0.18, 0.18, N_RUNS)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for i, w in enumerate(W_VALUES):
            if key == "share_extinct":
                ax.bar(i, summary[w][key], color=COLOR[w], width=0.6)
                continue
            vals = [r[key] for r in runs[w]]
            ax.scatter(i + jitter, vals, s=4, color=COLOR[w], alpha=0.45, linewidths=0)
            m, sd = summary[w][key]["mean"], summary[w][key]["sd"]
            ax.errorbar(
                i + 0.27,
                m,
                yerr=sd,
                fmt="o",
                ms=3.2,
                color="black",
                capsize=2,
                elinewidth=0.8,
            )
        ax.set_title(title, loc="left")
        ax.set_xticks(pos, [f"{w:g}" for w in W_VALUES])
        ax.set_xlim(-0.6, len(W_VALUES) - 0.3)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Waning rate w")
    save(fig, "m10-summary.png")


def figure_reinfections(hist):
    chosen = [0.2, 0.05, 0.01]
    fig, axes = plt.subplots(
        1, 3, figsize=(FIG_W, 2.3), sharey=True, constrained_layout=True
    )
    top = max(len(hist[w]) for w in chosen)
    for ax, w in zip(axes, chosen):
        h = hist[w] / hist[w].sum()
        ax.bar(np.arange(len(h)), h, width=1.0, color=COLOR[w])
        ax.set_title(f"w = {w:g}", loc="left")
        ax.set_xlim(-1, top)
        ax.set_xlabel("Infections per vertex over a run")
    axes[0].set_ylabel("Share of vertices")
    save(fig, "m10-reinfections.png")


def pooled_stats(hist):
    out = {}
    for w, h in hist.items():
        total = int(h.sum())
        out[f"w_{w:g}"] = {
            "vertices": total,
            "counts_by_infections_0_upward": [int(x) for x in h],
            "share_never_infected": float(h[0] / total),
            "share_exactly_once": float(h[1] / total) if len(h) > 1 else 0.0,
            "share_ten_or_more": float(h[10:].sum() / total),
            "max_infections": int(len(h) - 1),
        }
    return out


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    style()
    seeds = range(N_RUNS)

    print("size check", flush=True)
    sizes = size_check()
    print("experiments", flush=True)
    runs, curves, hist = run_grid(N_SMALL, seeds, MAX_STEPS)
    summary = {w: summarize(runs[w]) for w in W_VALUES}
    print("loop check", flush=True)
    loop = loop_check(runs)
    print("size sensitivity", flush=True)
    s_runs, _, _ = run_grid(N_SENS, range(SENS_RUNS), MAX_STEPS)

    quant = {}
    for w in W_VALUES:
        b = band(curves[w])
        quant[f"w_{w:g}"] = {
            f"step_{t}": {
                "p10": float(b[10][t]),
                "median": float(b[50][t]),
                "p90": float(b[90][t]),
            }
            for t in STEPS_REPORTED
        }

    results = {
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "networkx": nx.__version__,
        },
        "constants": {
            "attachment_parameter_m": ATTACH,
            "contagion_probability_per_infected_neighbor": BETA,
            "recovery_probability": GAMMA,
            "n_notebook": N_BIG,
            "initial_infected_notebook": SEEDS_BIG,
            "n_experiments": N_SMALL,
            "initial_infected_experiments": SEEDS_SMALL,
            "n_sensitivity": N_SENS,
            "sensitivity_seeds": [0, SENS_RUNS - 1],
            "w_values": W_VALUES,
            "seeds": [0, N_RUNS - 1],
            "runs_per_w": N_RUNS,
            "max_steps": MAX_STEPS,
            "notebook_step_cap": NOTEBOOK_CAP,
            "endemic_window_steps": ENDEMIC_WINDOW,
            "wave_min_share": WAVE_MIN,
            "wave_min_share_sensitivity": WAVE_MIN_SENSITIVITY,
            "wave_drop_fraction": WAVE_DROP,
            "initial_infection_counts_as_first_infection": True,
            "sd_is_sample_sd_ddof": 1,
        },
        "size_check": sizes,
        "notebook_loop_code_check": loop,
        "summary_by_w": {f"w_{w:g}": summary[w] for w in W_VALUES},
        "infected_share_quantiles_by_w": quant,
        "infections_per_vertex_histogram_by_w": pooled_stats(hist),
        "size_sensitivity_n20000_by_w": {
            f"w_{w:g}": summarize(s_runs[w]) for w in W_VALUES
        },
        "runs_by_w": {
            f"w_{w:g}": {key: [r[key] for r in runs[w]] for key in runs[w][0]}
            for w in W_VALUES
        },
    }
    path = ROOT / "results" / "m10-results.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    figure_timeseries(curves)
    figure_summary(runs, summary)
    figure_reinfections(hist)
    print("wrote", path)


if __name__ == "__main__":
    main()
