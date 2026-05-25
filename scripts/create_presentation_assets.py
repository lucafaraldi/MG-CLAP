#!/usr/bin/env python3
"""Create slide-ready assets for MG-CLAP presentation slides 7-13.

The script only reads existing result files for numeric summaries. Conceptual
schematics use deterministic toy geometry and are labelled as schematics, not
measured data.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation"

AUDIO = "#2563eb"
TEXT = "#dc2626"
PROTO = "#f59e0b"
ADAPTER = "#7c3aed"
GREEN = "#059669"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
PANEL = "#f8fafc"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, path: Path, *, dpi: int = 220) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.015, 0.018, text, color=MUTED, fontsize=10, ha="left", va="bottom")


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str = PANEL,
    ec: str = "#cbd5e1",
    color: str = INK,
    lw: float = 1.5,
    fontsize: int = 13,
    weight: str = "normal",
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.2,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475569",
    lw: float = 2.0,
    rad: float = 0.0,
    text: str | None = None,
    text_offset: tuple[float, float] = (0, 0),
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, color=color, fontsize=12, ha="center", va="center")


def slide7_schematic() -> None:
    rng = np.random.default_rng(7)
    audio_center = np.array([1.25, 0.62])
    text_center = np.array([-1.05, -0.42])
    audio = rng.normal(scale=[0.34, 0.24], size=(90, 2)) + audio_center
    text = rng.normal(scale=[0.30, 0.22], size=(90, 2)) + text_center
    mu_a = audio.mean(axis=0)
    mu_t = text.mean(axis=0)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.scatter(audio[:, 0], audio[:, 1], s=55, alpha=0.72, c=AUDIO, edgecolor="white", linewidth=0.5, label="audio embeddings")
    ax.scatter(text[:, 0], text[:, 1], s=55, alpha=0.72, c=TEXT, edgecolor="white", linewidth=0.5, label="text embeddings")
    ax.scatter(*mu_a, s=260, marker="X", c=AUDIO, edgecolor="black", linewidth=1.3, zorder=5)
    ax.scatter(*mu_t, s=260, marker="X", c=TEXT, edgecolor="black", linewidth=1.3, zorder=5)
    ax.text(mu_a[0] + 0.1, mu_a[1] + 0.18, r"$\mu_a$", fontsize=24, color=AUDIO, fontweight="bold")
    ax.text(mu_t[0] - 0.38, mu_t[1] - 0.22, r"$\mu_t$", fontsize=24, color=TEXT, fontweight="bold")

    add_arrow(ax, tuple(mu_t), tuple(mu_a), color=INK, lw=2.8)
    mid = (mu_a + mu_t) / 2
    ax.text(mid[0], mid[1] + 0.34, r"$\Delta = \mu_a - \mu_t$", fontsize=24, color=INK, ha="center")
    ax.text(mid[0] + 0.03, mid[1] - 0.36, r"gap distance $||\Delta||_2$", fontsize=20, color=INK, ha="center")

    ax.set_title("Modality Gap: Audio and Text Occupy Offset Regions", pad=18)
    ax.text(
        0.5,
        0.94,
        "Conceptual schematic - no measured numeric result",
        transform=ax.transAxes,
        ha="center",
        color=MUTED,
        fontsize=13,
    )
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-1.35, 1.45)
    savefig(fig, OUT / "slide7_modality_gap_schematic.png")


def slide7_pca_if_available() -> None:
    sources = {
        "laion": ROOT / "results" / "figure_1" / "laion-audiocaps-val" / "gap_pca.png",
        "msclap": ROOT / "results" / "figure_1" / "msclap-audiocaps-val" / "gap_pca.png",
    }
    for backbone, source in sources.items():
        if not source.exists():
            continue
        image = mpimg.imread(source)
        fig, ax = plt.subplots(figsize=(13.333, 7.5))
        ax.imshow(image)
        ax.axis("off")
        title = "LAION-CLAP" if backbone == "laion" else "MSCLAP"
        fig.text(0.5, 0.965, f"{title} AudioCaps Validation PCA", ha="center", va="top", fontsize=24, color=INK, fontweight="bold")
        fig.text(0.5, 0.925, "Existing Figure 1 PCA from real cached embeddings", ha="center", va="top", fontsize=13, color=MUTED)
        add_footer(fig, f"Source: {source.relative_to(ROOT)}")
        savefig(fig, OUT / f"slide7_modality_gap_pca_{backbone}.png")


def load_landscape(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    if data.get("mode") != "real":
        raise ValueError(f"Refusing non-real Figure 3 landscape: {path}")
    rows: list[dict[str, float | str]] = []
    backbone = str(data["backbone"])
    for tau_text, sweep in data["results"].items():
        lambdas = np.asarray(sweep["lambdas"], dtype=float)
        gaps = np.asarray(sweep["gaps"], dtype=float)
        losses = np.asarray(sweep["losses"], dtype=float)
        idx = int(np.argmin(losses))
        rows.append(
            {
                "backbone": backbone,
                "tau": float(tau_text),
                "best_lambda": float(lambdas[idx]),
                "gap_distance_at_min_loss": float(gaps[idx]),
                "minimum_loss": float(losses[idx]),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def method_label(method: str) -> str:
    labels = {
        "zero_shot_clap": "zero-shot CLAP",
        "naive_adapter": "naive adapter",
        "mgp_only": "MGP only",
        "mgc_only": "MGC only",
        "mgp_mgc": "MGP + MGC",
    }
    return labels.get(method, method.replace("_", " "))


def backbone_label(backbone: str) -> str:
    return {"laion": "LAION", "msclap": "MSCLAP"}.get(backbone, backbone.upper())


def format_float(value: float, ndigits: int = 4) -> str:
    return f"{value:.{ndigits}f}"


def render_table(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    subtitle: str | None = None,
    highlight_method: str | None = None,
    scale_y: float = 1.45,
    fig_width: float = 13.333,
    fig_height: float = 7.5,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, scale_y)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("white")

    if highlight_method is not None and "method_key" in df.attrs:
        method_keys = df.attrs["method_key"]
        for i, method in enumerate(method_keys, start=1):
            if method == highlight_method:
                for col in range(len(df.columns)):
                    cell = table[(i, col)]
                    cell.set_facecolor("#dcfce7")
                    cell.set_edgecolor("#16a34a")
                    cell.set_text_props(weight="bold", color=INK)

    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=24, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=12.5, color=MUTED)
    savefig(fig, path)


def slide8_figure3_summary() -> None:
    paths = [
        ROOT / "results" / "figure_3" / "laion-audiocaps-val" / "3b_landscape.json",
        ROOT / "results" / "figure_3" / "msclap-audiocaps-val" / "3b_landscape.json",
    ]
    minima = pd.concat([load_landscape(path) for path in paths], ignore_index=True)
    minima = minima.sort_values(["backbone", "tau"]).reset_index(drop=True)
    minima.to_csv(OUT / "slide8_fig3_minima_table.csv", index=False)

    table_df = minima.copy()
    table_df["Backbone"] = table_df["backbone"].map(backbone_label)
    table_df["tau"] = table_df["tau"].map(lambda x: f"{x:g}")
    table_df["best lambda"] = table_df["best_lambda"].map(lambda x: f"{x:.2f}")
    table_df["gap at min loss"] = table_df["gap_distance_at_min_loss"].map(lambda x: f"{x:.4f}")
    table_df["min loss"] = table_df["minimum_loss"].map(lambda x: f"{x:.4f}")
    table_display = table_df[["Backbone", "tau", "best lambda", "gap at min loss", "min loss"]]
    render_table(
        table_display,
        OUT / "slide8_fig3_minima_table.png",
        title="Figure 3b Minima by Temperature",
        subtitle="Minimum InfoNCE loss selected from real AudioCaps validation landscapes",
        scale_y=1.35,
    )

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    for backbone, color, marker in [("laion", AUDIO, "o"), ("msclap", GREEN, "s")]:
        sub = minima[minima["backbone"] == backbone].sort_values("tau")
        ax.plot(
            sub["tau"],
            sub["gap_distance_at_min_loss"],
            marker=marker,
            markersize=8,
            linewidth=2.8,
            color=color,
            label=backbone_label(backbone),
        )
        for _, row in sub.iterrows():
            ax.text(
                row["tau"],
                row["gap_distance_at_min_loss"] + 0.025,
                f"$\\lambda$={row['best_lambda']:.2f}",
                fontsize=9.5,
                color=color,
                ha="center",
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"temperature $\tau$ (log scale)")
    ax.set_ylabel(r"gap distance at minimum loss $||\Delta||_2$")
    ax.set_title("Loss-preferred modality gap depends on temperature", pad=16)
    ax.grid(True, which="both", color=GRID, linewidth=0.9)
    ax.legend(frameon=False, loc="upper right")
    ymin, ymax = ax.get_ylim()
    ax.annotate(
        "lower $\\tau$ = sharper\ncontrastive pressure",
        xy=(0.01, ymax - 0.08 * (ymax - ymin)),
        xytext=(0.035, ymax - 0.22 * (ymax - ymin)),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.8},
        fontsize=14,
        color=INK,
        ha="left",
    )
    add_footer(fig, "Sources: results/figure_3/laion-audiocaps-val/3b_landscape.json; results/figure_3/msclap-audiocaps-val/3b_landscape.json")
    savefig(fig, OUT / "slide8_fig3_simplified_plot.png")

    explanation = """# Slide 8 explanation

Inputs are the real Figure 3b landscape JSON files for LAION-CLAP and MSCLAP on AudioCaps validation. For each backbone and temperature $\\tau$, the table selects the saved $\\lambda$ value with the lowest InfoNCE loss.

Why original Figure 3b can show two branches:

- The original plot uses gap distance on the x-axis, but the sweep parameter is $\\lambda$ along the modality-gap direction.
- Gap distance folds the $\\lambda$ sweep: moving toward the centroids and moving past the zero-gap point can yield similar distances.
- Those two sides need not have identical InfoNCE loss because the paired similarities and neighborhood geometry are not symmetric after the shift.

What the coloured dots mean:

- Each coloured curve corresponds to one temperature $\\tau$.
- The dot marks the saved sweep point with minimum InfoNCE loss for that temperature.

Safe claim:

- In these real AudioCaps CLAP landscapes, the loss-preferred gap depends on temperature. Lower temperatures can prefer a non-zero gap, while higher temperatures can move the minimum toward smaller gaps.

Claim to avoid:

- Do not claim this sweep proves the trained encoder will converge to exactly these gaps, or that downstream continual-learning performance is caused only by the modality-gap value. The sweep is a controlled loss-landscape probe, not a full training trajectory or causal ablation by itself.
"""
    (OUT / "slide8_fig3_explanation.md").write_text(explanation)


def slide10_method_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.5, 0.96, "MG-CLAP-lite Method", ha="center", va="top", fontsize=26, fontweight="bold", color=INK)

    y = 0.62
    boxes = [
        ((0.035, y), (0.13, 0.13), "ESC-50\naudio clip $x$", "#eff6ff", AUDIO),
        ((0.205, y), (0.17, 0.13), "frozen CLAP\naudio encoder", "#f1f5f9", "#64748b"),
        ((0.415, y), (0.14, 0.13), "cached audio\nembedding $a$", "#eff6ff", AUDIO),
        ((0.595, y), (0.18, 0.13), "low-rank residual\nadapter $g_\\theta$", "#f5f3ff", ADAPTER),
        ((0.825, y), (0.13, 0.13), "adapted\naudio $z$", "#ecfdf5", GREEN),
    ]
    centers = []
    for xy, wh, text, fc, ec in boxes:
        add_box(ax, xy, wh, text, fc=fc, ec=ec, fontsize=12.5, weight="bold" if "adapter" in text else "normal")
        centers.append((xy[0] + wh[0] / 2, xy[1] + wh[1] / 2))
    for left, right in zip(centers[:-1], centers[1:]):
        add_arrow(ax, (left[0] + 0.075, left[1]), (right[0] - 0.075, right[1]), color="#475569", lw=2.0)

    ax.text(0.29, 0.79, "CLAP encoders frozen", color="#475569", fontsize=13, ha="center")
    ax.text(0.685, 0.79, "adapter trainable", color=ADAPTER, fontsize=13, ha="center", fontweight="bold")

    text_box = ((0.50, 0.31), (0.25, 0.13))
    proto_box = ((0.50, 0.12), (0.25, 0.13))
    add_box(ax, text_box[0], text_box[1], "A. text classifier\n$z \\cdot t_c / \\tau$\nfrozen text embeddings", fc="#fff1f2", ec=TEXT, fontsize=12)
    add_box(ax, proto_box[0], proto_box[1], "B. audio prototype classifier\n$z \\cdot p_c / \\tau$\nprototypes stored/updated\nafter each task", fc="#fffbeb", ec=PROTO, fontsize=11.5)
    add_arrow(ax, (0.89, 0.62), (0.63, 0.44), color=TEXT, lw=2.0, rad=-0.08)
    add_arrow(ax, (0.89, 0.62), (0.63, 0.25), color=PROTO, lw=2.0, rad=0.08)

    combine_xy = (0.80, 0.22)
    add_box(ax, combine_xy, (0.17, 0.16), "$\\mathrm{logits}$\n$= z\\cdot t_c/\\tau$\n$+\\,\\beta z\\cdot p_c/\\tau$", fc="#f8fafc", ec=INK, fontsize=12)
    add_arrow(ax, (0.75, 0.375), (0.80, 0.315), color="#475569", lw=1.8)
    add_arrow(ax, (0.75, 0.185), (0.80, 0.265), color="#475569", lw=1.8)
    add_box(ax, (0.80, 0.035), (0.17, 0.10), "predicted class\namong seen classes", fc="#ecfdf5", ec=GREEN, fontsize=12)
    add_arrow(ax, (0.885, 0.22), (0.885, 0.135), color=GREEN, lw=2.0)

    add_box(ax, (0.05, 0.30), (0.32, 0.10), "Preservation\nnegative-similarity drift stopping", fc="#f1f5f9", ec="#64748b", fontsize=12)
    add_box(ax, (0.05, 0.13), (0.32, 0.10), "Compensation\naudio prototype classifier", fc="#fffbeb", ec=PROTO, fontsize=12)
    add_footer(fig, "Conceptual method diagram - no measured numeric result")
    savefig(fig, OUT / "slide10_mgclap_lite_method.png")


def preferred_probe(backbone: str) -> tuple[Path, Path]:
    phase2 = ROOT / "results" / "continual_mgclap_phase2" / backbone / "seed_0"
    if (phase2 / "preservation_probe.csv").exists():
        return phase2 / "preservation_probe.csv", phase2 / "summary.json"
    base = ROOT / "results" / "continual_mgclap" / backbone
    return base / "preservation_probe.csv", base / "summary.json"


def slide11_preservation_probe() -> None:
    probes: list[tuple[str, Path, Path]] = []
    for backbone in ["laion", "msclap"]:
        csv_path, summary_path = preferred_probe(backbone)
        if csv_path.exists() and summary_path.exists():
            probes.append((backbone, csv_path, summary_path))
    if not probes:
        return

    fig, axes = plt.subplots(1, len(probes), figsize=(13.333, 7.5), sharey=True)
    if len(probes) == 1:
        axes = [axes]
    for ax, (backbone, csv_path, summary_path) in zip(axes, probes):
        df = pd.read_csv(csv_path)
        summary = json.loads(summary_path.read_text())
        alpha = float(summary["args"]["alpha"])
        fold = int(sorted(df["fold"].unique())[0])
        sub = df[df["fold"] == fold].sort_values("epoch")
        selected = sub[sub["selected"].astype(bool)]
        if not selected.empty:
            e_star = int(selected.iloc[-1]["epoch"])
        else:
            e_star = int(summary["e_star_per_fold"][str(fold)])

        color = AUDIO if backbone == "laion" else GREEN
        ax.plot(sub["epoch"], sub["drift"], marker="o", linewidth=2.5, markersize=7, color=color)
        ax.axhline(alpha, linestyle="--", linewidth=2.0, color=TEXT, label=rf"$\alpha={alpha:g}$")
        ax.axvline(e_star, linestyle="-", linewidth=2.0, color=INK, alpha=0.85, label=rf"$e^*={e_star}$")
        ax.scatter([e_star], [float(sub[sub["epoch"] == e_star]["drift"].iloc[0])], s=110, color=color, edgecolor=INK, zorder=5)
        ax.set_title(f"{backbone_label(backbone)} seed 0, fold {fold}")
        ax.set_xlabel("epoch")
        ax.grid(True, color=GRID)
        ax.legend(frameon=False, loc="upper left")
        ax.text(
            0.03,
            0.94,
            r"$D_e = |\mathrm{neg}_e-\mathrm{neg}_0| / \max(|\mathrm{neg}_0|,\epsilon)$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            color=INK,
        )
    axes[0].set_ylabel(r"relative negative-similarity drift $D_e$")
    fig.suptitle("Preservation Probe: Stop Before Negative-Similarity Drift Exceeds Threshold", fontsize=24, fontweight="bold", y=0.98)
    sources = "; ".join(str(csv.relative_to(ROOT)) for _, csv, _ in probes)
    add_footer(fig, f"Sources: {sources}")
    savefig(fig, OUT / "slide11_preservation_probe.png")

    explanation = """# Slide 11 explanation

The preservation probe uses saved `preservation_probe.csv` files. Phase 2 seed-0 probes are preferred when available; otherwise the base continual MG-CLAP probes are used.

The plotted drift is the saved relative negative-similarity drift:

`D_e = |neg_e - neg_0| / max(|neg_0|, epsilon)`

The dashed horizontal line is `alpha` from the corresponding `summary.json`. The vertical line is the selected epoch `e_star`, read from the CSV `selected` flag when present.

Interpretation:

- Preservation is implemented as early stopping against negative-class similarity drift.
- The adapter can improve positive alignment while stopping before negative similarities move too far from the frozen CLAP geometry.
- This plot is a diagnostic for the selected run/fold, not an aggregate performance result.
"""
    (OUT / "slide11_preservation_probe_explanation.md").write_text(explanation)


def slide12_prototype_compensation() -> None:
    rng = np.random.default_rng(12)
    class_specs = {
        "dog": {"center": np.array([-1.7, 0.75]), "color": "#2563eb"},
        "rain": {"center": np.array([0.15, -0.7]), "color": "#059669"},
        "siren": {"center": np.array([1.65, 0.65]), "color": "#dc2626"},
    }
    text_offsets = {"dog": np.array([-0.15, 0.62]), "rain": np.array([0.55, -0.40]), "siren": np.array([0.30, 0.55])}
    test_z = class_specs["siren"]["center"] + np.array([-0.38, -0.12])

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_title("Prototype Compensation Adds an Audio-Space Class Anchor", pad=18)
    for name, spec in class_specs.items():
        center = spec["center"]
        color = spec["color"]
        pts = rng.normal(scale=[0.22, 0.16], size=(24, 2)) + center
        ax.scatter(pts[:, 0], pts[:, 1], s=55, color=color, alpha=0.42, edgecolor="white", linewidth=0.5)
        ax.scatter(center[0], center[1], marker="D", s=260, color=PROTO, edgecolor=INK, linewidth=1.2, zorder=5)
        ax.text(center[0], center[1] - 0.32, rf"$p_{{\mathrm{{{name}}}}}$", fontsize=16, color=INK, ha="center", fontweight="bold")
        text_xy = center + text_offsets[name]
        ax.scatter(text_xy[0], text_xy[1], marker="s", s=190, color="white", edgecolor=color, linewidth=2.6, zorder=5)
        ax.text(text_xy[0], text_xy[1] + 0.20, rf"$t_{{\mathrm{{{name}}}}}$", fontsize=16, color=color, ha="center", fontweight="bold")
        ax.text(center[0], center[1] + 0.26, name, fontsize=14, color=color, ha="center", fontweight="bold")

    ax.scatter(test_z[0], test_z[1], marker="*", s=360, color="#111827", edgecolor="white", linewidth=1.2, zorder=6)
    ax.text(test_z[0] - 0.05, test_z[1] - 0.33, "test audio $z$", fontsize=16, color=INK, ha="center", fontweight="bold")
    siren_center = class_specs["siren"]["center"]
    siren_text = siren_center + text_offsets["siren"]
    add_arrow(ax, tuple(test_z), tuple(siren_text), color=TEXT, lw=2.0, text=r"$z\cdot t_c$", text_offset=(-0.10, 0.12))
    add_arrow(ax, tuple(test_z), tuple(siren_center), color=PROTO, lw=2.4, text=r"$z\cdot p_c$", text_offset=(0.03, -0.16))
    ax.text(
        0.5,
        0.08,
        r"$\mathrm{logits}(x,c) = z\cdot t_c/\tau + \beta\, z\cdot p_c/\tau$",
        transform=ax.transAxes,
        fontsize=24,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.12", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    ax.text(0.5, 0.92, "Conceptual 2D embedding schematic - no measured numeric result", transform=ax.transAxes, ha="center", color=MUTED, fontsize=13)
    ax.set_xlim(-2.55, 2.45)
    ax.set_ylim(-1.55, 1.85)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    savefig(fig, OUT / "slide12_audio_prototype_compensation.png")


def slide13_continual_setup() -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.5, 0.965, "Continual-Learning Evaluation Setup", ha="center", va="top", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.5, 0.925, "ESC-50 split into 10 tasks x 5 classes", ha="center", va="top", fontsize=14, color=MUTED)

    y = 0.58
    x0 = 0.055
    gap = 0.012
    width = (0.89 - 9 * gap) / 10
    for task in range(1, 11):
        x = x0 + (task - 1) * (width + gap)
        fc = "#eff6ff" if task <= 2 else "#f8fafc"
        ec = AUDIO if task <= 2 else "#cbd5e1"
        add_box(ax, (x, y), (width, 0.16), f"Task {task}\n+5 classes", fc=fc, ec=ec, fontsize=10.8, weight="bold" if task <= 2 else "normal")
        ax.text(x + width / 2, y - 0.045, f"{task * 5}", ha="center", va="top", fontsize=12, color=INK)
        if task < 10:
            add_arrow(ax, (x + width, y + 0.08), (x + width + gap, y + 0.08), color="#94a3b8", lw=1.5)

    ax.text(0.055, y - 0.095, "seen classes after task k", ha="left", va="top", fontsize=12.5, color=MUTED)
    ax.text(0.50, y + 0.22, "Task 1: 5 classes    Task 2: +5 classes    ...    Task 10: 50 classes", ha="center", va="center", fontsize=16, color=INK)

    eval_box = (0.13, 0.25)
    add_box(
        ax,
        eval_box,
        (0.74, 0.16),
        "Evaluation after task $k$: classify among all seen classes $C_{\\leq k}$",
        fc="#ecfdf5",
        ec=GREEN,
        fontsize=17,
        weight="bold",
    )
    add_arrow(ax, (0.50, y), (0.50, eval_box[1] + 0.16), color=GREEN, lw=2.0)
    add_box(ax, (0.13, 0.08), (0.34, 0.10), "no task ID at test time", fc="#fff7ed", ec="#f97316", fontsize=14, weight="bold")
    add_box(ax, (0.53, 0.08), (0.34, 0.10), "future unseen classes are not evaluated before they appear", fc="#f1f5f9", ec="#64748b", fontsize=12.5)
    add_footer(fig, "Conceptual setup diagram - no measured numeric result")
    savefig(fig, OUT / "slide13_continual_setup.png")


def beta4_source() -> Path:
    requested = ROOT / "results" / "phase2" / "fixed_beta4_main_table.csv"
    fallback = ROOT / "results" / "fixed_beta4_main_table.csv"
    if requested.exists():
        return requested
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing beta-4 table: {requested} or {fallback}")


def format_mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.1f} +/- {100 * std:.1f}"


def slide13_tables() -> None:
    source = beta4_source()
    raw = pd.read_csv(source)
    wanted = ["zero_shot_clap", "naive_adapter", "mgp_only", "mgc_only", "mgp_mgc"]
    order = {method: i for i, method in enumerate(wanted)}
    raw = raw[raw["method"].isin(wanted)].copy()
    raw["method_order"] = raw["method"].map(order)
    raw = raw.sort_values(["backbone", "method_order"]).reset_index(drop=True)

    full = pd.DataFrame(
        {
            "Backbone": raw["backbone"].map(backbone_label),
            "Method": raw["method"].map(method_label),
            "Runs": raw["runs"].astype(int).astype(str),
            "Avg acc (%)": [format_mean_std(m, s) for m, s in zip(raw["Avg_mean"], raw["Avg_std"])],
            "Last acc (%)": [format_mean_std(m, s) for m, s in zip(raw["Last_mean"], raw["Last_std"])],
            "Forgetting (%)": [format_mean_std(m, s) for m, s in zip(raw["forgetting_proxy_mean"], raw["forgetting_proxy_std"])],
        }
    )
    full.attrs["method_key"] = raw["method"].tolist()
    render_table(
        full,
        OUT / "slide13_main_results_table_fixed_beta4.png",
        title="MG-CLAP-lite Main Results (Fixed beta = 4)",
        subtitle=f"Real ESC-50 continual-learning results. Source: {source.relative_to(ROOT)}",
        highlight_method="mgp_mgc",
        scale_y=1.55,
    )

    compact = pd.DataFrame(
        {
            "Backbone": raw["backbone"].map(backbone_label),
            "Method": raw["method"].map(method_label),
            "Avg (%)": [format_mean_std(m, s) for m, s in zip(raw["Avg_mean"], raw["Avg_std"])],
            "Last (%)": [format_mean_std(m, s) for m, s in zip(raw["Last_mean"], raw["Last_std"])],
        }
    )
    compact.attrs["method_key"] = raw["method"].tolist()
    render_table(
        compact,
        OUT / "slide13_main_results_table_compact.png",
        title="Main Results: Avg and Last Accuracy",
        subtitle=f"Fixed beta = 4. Source: {source.relative_to(ROOT)}",
        highlight_method="mgp_mgc",
        scale_y=1.65,
    )


def main() -> None:
    configure_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)

    slide7_schematic()
    slide7_pca_if_available()
    slide8_figure3_summary()
    slide10_method_diagram()
    slide11_preservation_probe()
    slide12_prototype_compensation()
    slide13_continual_setup()
    slide13_tables()

    generated = sorted(path.name for path in OUT.iterdir() if path.is_file())
    print("Generated presentation assets:")
    for name in generated:
        print(f"  - {OUT / name}")


if __name__ == "__main__":
    main()
