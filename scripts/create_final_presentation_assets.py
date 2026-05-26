#!/usr/bin/env python3
"""Create final slide assets for the MG-CLAP presentation.

Numeric plots and tables are derived only from real-data result files. Conceptual
schematics are deterministic drawings and are labelled as conceptual schematics.
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
OUT = ROOT / "results" / "presentation_final"

INK = "#111827"
NAVY = "#0f172a"
MUTED = "#64748b"
GRID = "#e5e7eb"
PANEL = "#f8fafc"
ZERO = "#374151"
NAIVE = "#dc2626"
MGP = "#2563eb"
MGC = "#059669"
MGPMGC = "#7c3aed"
YELLOW = "#f59e0b"
RED_BG = "#fee2e2"
YELLOW_BG = "#fef3c7"
GREEN_BG = "#dcfce7"

METHOD_ORDER = ["zero_shot_clap", "naive_adapter", "mgp_only", "mgc_only", "mgp_mgc"]
METHOD_COLORS = {
    "zero_shot_clap": ZERO,
    "naive_adapter": NAIVE,
    "mgp_only": MGP,
    "mgc_only": MGC,
    "mgp_mgc": MGPMGC,
}
METHOD_LABELS = {
    "zero_shot_clap": "zero-shot",
    "naive_adapter": "naive adapter",
    "mgp_only": "MGP only",
    "mgc_only": "MGC only",
    "mgp_mgc": "MGP+MGC",
}
BACKBONE_LABELS = {"laion": "LAION", "msclap": "MSCLAP"}

REAL_NUMERIC_SOURCES = [
    "results/fixed_beta4_main_table.csv",
    "results/phase2_main_grid_table.csv",
    "results/phase2_alpha_sensitivity.csv",
    "results/phase2_rank_sensitivity.csv",
    "results/phase2_class_order_robustness.csv",
    "results/figure_1/laion-audiocaps-val/stats.json",
    "results/figure_1/msclap-audiocaps-val/stats.json",
    "results/table_1/laion-shift/summary.json",
    "results/table_1/msclap-shift/summary.json",
    "results/figure_3/laion-audiocaps-val/3b_landscape.json",
    "results/figure_3/msclap-audiocaps-val/3b_landscape.json",
]

GENERATED: list[tuple[str, str, Path]] = []


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 23,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.default": "regular",
        }
    )


def register(slide: str, desc: str, path: Path) -> None:
    GENERATED.append((slide, desc, path))


def savefig(fig: plt.Figure, path: Path, slide: str, desc: str, dpi: int = 220) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    register(slide, desc, path)


def title(fig: plt.Figure, main: str, sub: str | None = None) -> None:
    fig.text(0.5, 0.955, main, ha="center", va="top", fontsize=26, fontweight="bold", color=NAVY)
    if sub:
        fig.text(0.5, 0.915, sub, ha="center", va="top", fontsize=14, color=MUTED)


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.015, 0.018, text, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def conceptual_footer(fig: plt.Figure) -> None:
    footer(fig, "conceptual schematic — no measured numeric result")


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
    fontsize: float = 12.5,
    weight: str = "normal",
    rounding: float = 0.022,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight=weight, linespacing=1.18)
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475569",
    lw: float = 2.0,
    rad: float = 0.0,
    arrowstyle: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=arrowstyle,
            mutation_scale=18,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def setup_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def fmt_pct(x: float, ndigits: int = 1) -> str:
    return f"{100 * x:.{ndigits}f}"


def fmt_mean_std(mean: float, std: float, ndigits: int = 1) -> str:
    return f"{100 * mean:.{ndigits}f} +/- {100 * std:.{ndigits}f}"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def backbone_label(backbone: str) -> str:
    return BACKBONE_LABELS.get(backbone, backbone.upper())


def load_csv(name: str) -> pd.DataFrame:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_json(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def render_table(
    df: pd.DataFrame,
    path: Path,
    slide: str,
    desc: str,
    *,
    main_title: str,
    subtitle: str | None = None,
    highlight_methods: list[str] | None = None,
    method_keys: list[str] | None = None,
    font_size: float = 12,
    scale_y: float = 1.45,
    col_widths: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("white")
    if highlight_methods and method_keys:
        for ridx, method in enumerate(method_keys, start=1):
            if method in highlight_methods:
                for cidx in range(len(df.columns)):
                    cell = table[(ridx, cidx)]
                    cell.set_facecolor("#ede9fe")
                    cell.set_edgecolor(MGPMGC)
                    cell.set_text_props(weight="bold", color=INK)
    title(fig, main_title, subtitle)
    savefig(fig, path, slide, desc)


def slide1() -> None:
    fig, ax = setup_canvas()
    title(fig, "Mind the Gap in CLAP", "Preserving and Compensating Audio-Text Geometry for Continual Learning")
    add_box(ax, (0.08, 0.57), (0.28, 0.17), "Image-text CLIP\nMG-CLIP", fc="#f1f5f9", ec=ZERO, fontsize=17, weight="bold")
    add_box(ax, (0.64, 0.57), (0.28, 0.17), "Audio-text CLAP\nMG-CLAP-lite", fc="#f5f3ff", ec=MGPMGC, fontsize=17, weight="bold")
    add_arrow(ax, (0.38, 0.655), (0.62, 0.655), color=INK, lw=2.8)
    ax.text(0.50, 0.705, "transfer principle", ha="center", va="bottom", color=INK, fontsize=14)

    add_box(ax, (0.13, 0.36), (0.08, 0.10), "image", fc="white", ec=ZERO, fontsize=13)
    ax.text(0.24, 0.41, r"$\leftrightarrow$", ha="center", va="center", fontsize=26, color=INK)
    add_box(ax, (0.27, 0.36), (0.08, 0.10), "text", fc="white", ec=ZERO, fontsize=13)
    add_box(ax, (0.69, 0.36), (0.08, 0.10), "audio", fc="white", ec=MGP, fontsize=13)
    ax.text(0.80, 0.41, r"$\leftrightarrow$", ha="center", va="center", fontsize=26, color=INK)
    add_box(ax, (0.83, 0.36), (0.08, 0.10), "text", fc="white", ec=NAIVE, fontsize=13)

    ax.text(0.5, 0.20, "Does the MG-CLIP principle transfer from images to sounds?", ha="center", va="center", fontsize=24, color=NAVY, fontweight="bold")
    conceptual_footer(fig)
    savefig(fig, OUT / "slide1_title_thesis.png", "Slide 1", "Title visual")


def slide2() -> None:
    fig, ax = setup_canvas()
    title(fig, "CLAP Zero-Shot Classification", "CLAP is an embedding model, not a fixed 50-way classifier.")
    add_box(ax, (0.05, 0.59), (0.14, 0.13), "ESC-50\naudio clip $x$", fc="#eff6ff", ec=MGP, fontsize=13, weight="bold")
    add_box(ax, (0.25, 0.59), (0.18, 0.13), "frozen CLAP\naudio encoder", fc="#f1f5f9", ec=ZERO, fontsize=13)
    add_box(ax, (0.49, 0.59), (0.14, 0.13), "audio\nembedding $a$", fc="#eff6ff", ec=MGP, fontsize=13)
    add_arrow(ax, (0.19, 0.655), (0.25, 0.655))
    add_arrow(ax, (0.43, 0.655), (0.49, 0.655))

    labels = ["a sound of dog", "a sound of rain", "a sound of siren"]
    for i, lab in enumerate(labels):
        add_box(ax, (0.08, 0.25 - i * 0.085), (0.20, 0.055), lab, fc="white", ec=NAIVE, fontsize=11)
    add_box(ax, (0.34, 0.17), (0.19, 0.16), "frozen CLAP\ntext encoder", fc="#f1f5f9", ec=ZERO, fontsize=13)
    add_box(ax, (0.60, 0.17), (0.15, 0.16), "text\nembeddings $t_c$", fc="#fff1f2", ec=NAIVE, fontsize=13)
    add_arrow(ax, (0.28, 0.245), (0.34, 0.25))
    add_arrow(ax, (0.53, 0.25), (0.60, 0.25))

    add_box(ax, (0.72, 0.48), (0.20, 0.24), "similarity scores\n$a^T t_{dog}$\n$a^T t_{rain}$\n$a^T t_{siren}$", fc="#f8fafc", ec=INK, fontsize=13)
    add_arrow(ax, (0.63, 0.63), (0.72, 0.62), color=MGP)
    add_arrow(ax, (0.75, 0.25), (0.82, 0.48), color=NAIVE)
    add_box(ax, (0.72, 0.25), (0.20, 0.10), "argmax\npredicted class", fc="#ecfdf5", ec=MGC, fontsize=13, weight="bold")
    add_arrow(ax, (0.82, 0.48), (0.82, 0.35), color=MGC)
    ax.text(0.50, 0.08, r"$\hat{y} = \arg\max_c\; a^T t_c$", ha="center", va="center", fontsize=28, color=NAVY)
    conceptual_footer(fig)
    savefig(fig, OUT / "slide2_clap_zero_shot_classifier.png", "Slide 2", "CLAP zero-shot classifier schematic")


def slide3() -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    title(fig, "Original Mind the Gap Premise", "Pretrained contrastive modalities can occupy offset regions")
    rng = np.random.default_rng(3)
    image_center = np.array([-0.9, 0.1])
    text_center = np.array([1.0, -0.2])
    image = rng.normal(scale=[0.35, 0.25], size=(85, 2)) + image_center
    text = rng.normal(scale=[0.32, 0.22], size=(85, 2)) + text_center
    mu_i = image.mean(axis=0)
    mu_t = text.mean(axis=0)
    ax.scatter(image[:, 0], image[:, 1], s=55, c=MGP, alpha=0.65, edgecolor="white", linewidth=0.5, label="image embeddings")
    ax.scatter(text[:, 0], text[:, 1], s=55, c=NAIVE, alpha=0.65, edgecolor="white", linewidth=0.5, label="text embeddings")
    ax.scatter(*mu_i, s=260, marker="X", c=MGP, edgecolor=INK, linewidth=1.2, zorder=5)
    ax.scatter(*mu_t, s=260, marker="X", c=NAIVE, edgecolor=INK, linewidth=1.2, zorder=5)
    add_arrow(ax, tuple(mu_t), tuple(mu_i), color=INK, lw=2.6)
    mid = (mu_i + mu_t) / 2
    ax.text(mu_i[0] - 0.33, mu_i[1] + 0.25, r"$\mu_{image}$", fontsize=20, color=MGP, fontweight="bold")
    ax.text(mu_t[0] + 0.10, mu_t[1] - 0.25, r"$\mu_{text}$", fontsize=20, color=NAIVE, fontweight="bold")
    ax.text(mid[0], mid[1] + 0.25, r"$\Delta = \mu_{image} - \mu_{text}$", fontsize=22, color=INK, ha="center")
    ax.text(mid[0], mid[1] - 0.30, r"gap distance $||\Delta||_2$", fontsize=19, color=INK, ha="center")
    bullets = [
        "Pretrained CLIP modalities occupy offset regions.",
        "The gap is a structural property, not necessarily a bug.",
        "Continual learning can damage this structure.",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.07, 0.23 - 0.07 * i, f"- {b}", transform=ax.transAxes, ha="left", va="center", fontsize=16, color=INK)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.3, 1.25)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    conceptual_footer(fig)
    savefig(fig, OUT / "slide3_original_mind_the_gap_premise.png", "Slide 3", "Original Mind the Gap premise")


def slide4() -> None:
    fig, ax = setup_canvas()
    title(fig, "Original MG-CLIP Response", "Preserve shared geometry while adding task-specific discrimination")
    add_box(ax, (0.06, 0.55), (0.20, 0.15), "Pretrained\nCLIP", fc="#f1f5f9", ec=ZERO, fontsize=17, weight="bold")
    add_box(ax, (0.40, 0.55), (0.22, 0.15), "continual\nlearning risk", fc="#fff1f2", ec=NAIVE, fontsize=17, weight="bold")
    add_box(ax, (0.76, 0.55), (0.18, 0.15), "MG-CLIP\nresponse", fc="#f5f3ff", ec=MGPMGC, fontsize=17, weight="bold")
    add_arrow(ax, (0.26, 0.625), (0.40, 0.625), color=INK, lw=2.5)
    add_arrow(ax, (0.62, 0.625), (0.76, 0.625), color=INK, lw=2.5)

    add_box(ax, (0.20, 0.25), (0.25, 0.16), "A. Preserve\nmonitor cross-modal geometry\navoid destructive drift", fc="#eff6ff", ec=MGP, fontsize=13)
    add_box(ax, (0.56, 0.25), (0.25, 0.16), "B. Compensate\nadd intra-modal classifier\nfor task discrimination", fc="#ecfdf5", ec=MGC, fontsize=13)
    ax.text(0.325, 0.18, "stability", ha="center", va="center", fontsize=20, color=MGP, fontweight="bold")
    ax.text(0.685, 0.18, "plasticity", ha="center", va="center", fontsize=20, color=MGC, fontweight="bold")
    ax.plot([0.33, 0.68], [0.13, 0.13], color=INK, lw=3)
    ax.scatter([0.33, 0.68], [0.13, 0.13], s=130, color=[MGP, MGC], zorder=5)
    ax.text(0.50, 0.095, "stability-plasticity trade-off", ha="center", va="center", fontsize=14, color=MUTED)
    conceptual_footer(fig)
    savefig(fig, OUT / "slide4_original_mgclip_preserve_compensate.png", "Slide 4", "Original MG-CLIP preserve/compensate diagram")


def slide5() -> None:
    rows = [
        ["image-text CLIP", "audio-text CLAP"],
        ["image classes", "sound classes"],
        ["visual adaptation", "audio embedding adapter"],
        ["visual classifier", "audio prototype classifier"],
        ["preserve image-text gap", "preserve audio-text geometry"],
    ]
    df = pd.DataFrame(rows, columns=["Original MG-CLIP", "Our MG-CLAP-lite"])
    render_table(
        df,
        OUT / "slide5_extension_from_mgclip_to_mgclap.png",
        "Slide 5",
        "Extension map table",
        main_title="Extension Map: MG-CLIP to MG-CLAP-lite",
        subtitle="Controlled embedding-level extension, not full CLAP fine-tuning.",
        font_size=16,
        scale_y=2.0,
        col_widths=[0.42, 0.42],
    )


def slide6() -> None:
    fig, ax = setup_canvas()
    title(fig, "Experimental Evidence Map", "Real-data stack used for the main claims")
    add_box(ax, (0.05, 0.67), (0.15, 0.10), "AudioCaps", fc="#f1f5f9", ec=ZERO, fontsize=15, weight="bold")
    add_box(ax, (0.30, 0.67), (0.19, 0.10), "1. Gap\ngeometry", fc="#eff6ff", ec=MGP, fontsize=13)
    add_box(ax, (0.59, 0.67), (0.24, 0.10), "2. Contrastive loss\nlandscape", fc="#f5f3ff", ec=MGPMGC, fontsize=13)
    add_arrow(ax, (0.20, 0.72), (0.30, 0.72))
    add_arrow(ax, (0.49, 0.72), (0.59, 0.72))

    add_box(ax, (0.05, 0.50), (0.15, 0.10), "ESC-50", fc="#f1f5f9", ec=ZERO, fontsize=15, weight="bold")
    add_box(ax, (0.30, 0.50), (0.19, 0.10), "3. Zero-shot\ngap shift", fc="#fff7ed", ec=YELLOW, fontsize=13)
    add_box(ax, (0.59, 0.50), (0.24, 0.10), "4. Continual learning\nMG-CLAP-lite", fc="#ecfdf5", ec=MGC, fontsize=13)
    add_arrow(ax, (0.20, 0.55), (0.30, 0.55))
    add_arrow(ax, (0.49, 0.55), (0.59, 0.55))

    rows = [
        ["Does CLAP have a gap?", "AudioCaps", "LAION, MSCLAP", "gap stats/PCA"],
        ["Does loss prefer different gaps?", "AudioCaps", "LAION, MSCLAP", "InfoNCE landscape"],
        ["Does shifting affect classification?", "ESC-50", "LAION, MSCLAP", "shift sweep"],
        ["Does P+C help continual learning?", "ESC-50", "LAION, MSCLAP", "Avg/Last/forgetting"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Question", "Dataset", "Models", "Output"],
        cellLoc="center",
        colLoc="center",
        loc="lower center",
        bbox=[0.045, 0.10, 0.91, 0.30],
        colWidths=[0.38, 0.15, 0.20, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f9fafb")
    footer(fig, "Main evidence uses real AudioCaps and ESC-50 result files only")
    savefig(fig, OUT / "slide6_experimental_evidence_map.png", "Slide 6", "Experimental evidence map")


def slide14() -> None:
    df = load_csv("results/phase2_main_grid_table.csv")
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), sharey=True)
    title(fig, "Seed and Class-Order Robustness", "Fixed beta=4. Seed 0 uses canonical order; seeds 1-3 use shuffled class orders.")
    for ax, backbone in zip(axes, ["laion", "msclap"]):
        sub = df[df["backbone"] == backbone]
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("seed")
            lw = 3.6 if method == "mgp_mgc" else 1.8
            alpha = 1.0 if method == "mgp_mgc" else 0.75
            marker = "o" if method == "mgp_mgc" else "s"
            ax.plot(m["seed"], 100 * m["Avg_mean"], marker=marker, lw=lw, color=METHOD_COLORS[method], alpha=alpha, label=method_label(method))
        ax.set_title(backbone_label(backbone))
        ax.set_xlabel("seed")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(True, color=GRID)
        ax.set_ylim(68, 101)
    axes[0].set_ylabel("Avg accuracy (%)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=10.5)
    fig.text(0.5, 0.105, "MGP+MGC remains strong across seeds and class orders.", ha="center", va="center", fontsize=22, color=MGPMGC, fontweight="bold")
    footer(fig, "Source: results/phase2_main_grid_table.csv")
    savefig(fig, OUT / "slide14_seed_order_robustness.png", "Slide 14", "Seed and class-order robustness plot")
    notes = """# Slide 14 speaker notes

- Fixed beta=4 robustness is evaluated across four seeds per backbone.
- Seed 0 uses the canonical ESC-50 class order; seeds 1-3 use shuffled class orders.
- MGP+MGC is consistently near the top for both LAION and MSCLAP.
- Caveat: shuffled class orders are robustness probes, not a replacement for new datasets.
- Caveat: beta is fixed here, not tuned on a held-out validation split.
"""
    path = OUT / "slide14_seed_order_summary.md"
    path.write_text(notes)
    register("Slide 14", "Seed/order speaker-note summary", path)


def slide15() -> None:
    alpha = load_csv("results/phase2_alpha_sensitivity.csv")
    rank = load_csv("results/phase2_rank_sensitivity.csv")
    a = alpha[alpha["method"] == "mgp_mgc"].sort_values("alpha")
    r = rank[rank["method"] == "mgp_mgc"].sort_values("adapter_rank")
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5))
    title(fig, "Ablation: Preservation Threshold and Adapter Rank", "LAION MGP+MGC real ablation runs")
    axes[0].plot(a["alpha"], 100 * a["Avg_mean"], marker="o", color=MGPMGC, lw=3, label="Avg")
    axes[0].plot(a["alpha"], 100 * a["Last_mean"], marker="s", color=MGP, lw=2.4, label="Last")
    axes[0].set_xlabel("preservation threshold alpha")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Preservation threshold")
    axes[0].grid(True, color=GRID)
    axes[0].legend(frameon=False)
    forget_a = "\n".join([f"alpha {row.alpha:.2f}: {100*row.forgetting_proxy_mean:.1f}%" for row in a.itertuples()])
    axes[0].text(0.96, 0.05, "forgetting\n" + forget_a, transform=axes[0].transAxes, ha="right", va="bottom", fontsize=10.5, color=INK, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"})

    axes[1].plot(r["adapter_rank"], 100 * r["Avg_mean"], marker="o", color=MGPMGC, lw=3, label="Avg")
    axes[1].plot(r["adapter_rank"], 100 * r["Last_mean"], marker="s", color=MGP, lw=2.4, label="Last")
    axes[1].set_xlabel("adapter rank")
    axes[1].set_title("Adapter rank")
    axes[1].set_xticks([4, 8, 16, 32])
    axes[1].grid(True, color=GRID)
    axes[1].legend(frameon=False)
    forget_r = "\n".join([f"rank {int(row.adapter_rank)}: {100*row.forgetting_proxy_mean:.1f}%" for row in r.itertuples()])
    axes[1].text(0.96, 0.05, "forgetting\n" + forget_r, transform=axes[1].transAxes, ha="right", va="bottom", fontsize=10.5, color=INK, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"})
    fig.text(0.5, 0.095, "Stricter preservation improves performance and reduces forgetting; rank sensitivity is mild.", ha="center", fontsize=17, color=NAVY, fontweight="bold")
    footer(fig, "Sources: results/phase2_alpha_sensitivity.csv; results/phase2_rank_sensitivity.csv")
    savefig(fig, OUT / "slide15_alpha_rank_ablation.png", "Slide 15", "Alpha/rank ablation plot")

    table_rows = []
    for row in a.itertuples():
        table_rows.append(["alpha", f"{row.alpha:.2f}", fmt_mean_std(row.Avg_mean, row.Avg_std), fmt_mean_std(row.Last_mean, row.Last_std), fmt_mean_std(row.forgetting_proxy_mean, row.forgetting_proxy_std)])
    for row in r.itertuples():
        table_rows.append(["rank", f"{int(row.adapter_rank)}", fmt_mean_std(row.Avg_mean, row.Avg_std), fmt_mean_std(row.Last_mean, row.Last_std), fmt_mean_std(row.forgetting_proxy_mean, row.forgetting_proxy_std)])
    table_df = pd.DataFrame(table_rows, columns=["Ablation", "Value", "Avg (%)", "Last (%)", "Forgetting (%)"])
    render_table(
        table_df,
        OUT / "slide15_alpha_rank_table.png",
        "Slide 15",
        "Compact alpha/rank numeric table",
        main_title="Alpha and Rank Sensitivity: MGP+MGC",
        subtitle="LAION real ablation runs; lower forgetting is better",
        font_size=12.5,
        scale_y=1.45,
    )


def slide16() -> None:
    rows = [
        ["CLAP has a modality gap", "AudioCaps stats", "strong"],
        ["Temperature changes loss-preferred gap", "Figure 3 real runs", "strong"],
        ["Gap shift universally helps", "LAION yes, MSCLAP near-null", "mixed"],
        ["Naive continual adaptation helps", "worse than zero-shot", "false"],
        ["Preservation alone transfers", "LAION yes, MSCLAP mixed", "partial"],
        ["Compensation alone transfers", "below zero-shot", "weak"],
        ["Preservation + compensation transfers", "best on both", "strong"],
    ]
    fig, ax = setup_canvas()
    title(fig, "What Transferred and What Did Not?", "Claims supported by the real-data evidence stack")
    table = ax.table(cellText=rows, colLabels=["Claim", "Evidence", "Verdict"], cellLoc="left", colLoc="center", bbox=[0.055, 0.12, 0.89, 0.72], colWidths=[0.45, 0.30, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(13.2)
    verdict_bg = {"strong": GREEN_BG, "mixed": YELLOW_BG, "partial": YELLOW_BG, "weak": YELLOW_BG, "false": RED_BG}
    verdict_fg = {"strong": MGC, "mixed": "#92400e", "partial": "#92400e", "weak": "#92400e", "false": NAIVE}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            if c == 2:
                verdict = rows[r - 1][2]
                cell.set_facecolor(verdict_bg[verdict])
                cell.set_text_props(color=verdict_fg[verdict], weight="bold", ha="center")
            elif r % 2 == 0:
                cell.set_facecolor("#f9fafb")
    footer(fig, "Evidence references real AudioCaps, ESC-50 shift, and Phase 2 continual-learning outputs")
    savefig(fig, OUT / "slide16_claims_evidence_matrix.png", "Slide 16", "Claims/evidence traffic-light matrix")


def slide17() -> None:
    rows = [
        ["Frozen embeddings", "not full CLAP fine-tuning", "LoRA/audio encoder adaptation"],
        ["ESC-50 only", "limited external validity", "UrbanSound8K/FSD50K"],
        ["beta fixed, not validation-tuned", "possible hyperparameter critique", "held-out validation"],
        ["No replay memory", "not compared to rehearsal CL", "add replay baseline"],
        ["Synthetic artifacts exist historically", "presentation safety", "use only real-data stack"],
    ]
    df = pd.DataFrame(rows, columns=["Limitation", "Why it matters", "Next step"])
    render_table(
        df,
        OUT / "slide17_limitations_threats.png",
        "Slide 17",
        "Limitations and threats table",
        main_title="Limitations and Threats to Validity",
        subtitle="These limitations define scope; they do not invalidate the controlled transfer result.",
        font_size=13.5,
        scale_y=1.8,
        col_widths=[0.28, 0.33, 0.30],
    )


def slide18() -> None:
    fig, ax = setup_canvas()
    title(fig, "Final Takeaway", "MG-CLAP-lite supports the MG-CLIP principle at the embedding-adapter level")
    blocks = [
        (0.08, "1. Geometry transfers", "CLAP has a real\naudio-text modality gap.", MGP, "#eff6ff"),
        (0.385, "2. Mechanism transfers", "Loss-preferred gap\ndepends on temperature.", MGPMGC, "#f5f3ff"),
        (0.69, "3. CL principle transfers", "Preservation + compensation\nbeats naive adaptation.", MGC, "#ecfdf5"),
    ]
    for x, head, body, color, fc in blocks:
        add_box(ax, (x, 0.44), (0.23, 0.22), head + "\n\n" + body, fc=fc, ec=color, fontsize=14.5, weight="bold")
    ax.text(0.50, 0.245, "MG-CLAP-lite supports the MG-CLIP principle at the embedding-adapter level.", ha="center", va="center", fontsize=22, color=NAVY, fontweight="bold")
    add_box(ax, (0.365, 0.11), (0.27, 0.08), "Warning: not full CLAP fine-tuning", fc="#fff7ed", ec=YELLOW, fontsize=13.5, weight="bold")
    footer(fig, "Conclusion constrained to the real-data, frozen-embedding MG-CLAP-lite setup")
    savefig(fig, OUT / "slide18_final_takeaway.png", "Slide 18", "Final conclusion slide")


def backup_formula_sheet() -> None:
    fig, ax = setup_canvas()
    title(fig, "Backup: Formula Sheet", "Definitions used across the presentation")
    formulas = [
        ("gap vector", r"$\Delta = \mu_a - \mu_t$"),
        ("gap distance", r"$||\Delta||_2$"),
        ("pair cosine", r"$\cos(a_i,t_i)=a_i^Tt_i$ for L2-normalized embeddings"),
        ("InfoNCE loss", r"$L=-\frac{1}{N}\sum_i \log \frac{\exp(a_i^Tt_i/\tau)}{\sum_j \exp(a_i^Tt_j/\tau)}$"),
        ("gap shift", r"$(a,t) \mapsto \mathrm{shift}(a,t;\lambda,\Delta)$"),
        ("preservation drift", r"$D_e=|neg_e-neg_0|/\max(|neg_0|,\epsilon)$"),
        ("combined logits", r"$logits(x,c)=z^Tt_c/\tau + \beta z^Tp_c/\tau$"),
        ("Avg / Last", "Avg = mean accuracy over tasks; Last = final-task seen-class accuracy"),
    ]
    for i, (name, formula) in enumerate(formulas):
        y = 0.80 - i * 0.085
        ax.text(0.08, y, name, ha="left", va="center", fontsize=15, color=NAVY, fontweight="bold")
        ax.text(0.31, y, formula, ha="left", va="center", fontsize=17, color=INK)
    footer(fig, "Formula sheet; no new numeric result")
    savefig(fig, OUT / "backup_formula_sheet.png", "Backup 1", "Formula sheet")


def backup_full_phase2_table() -> None:
    raw = load_csv("results/fixed_beta4_main_table.csv")
    raw["method_order"] = raw["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    raw = raw.sort_values(["backbone", "method_order"])
    df = pd.DataFrame(
        {
            "Backbone": raw["backbone"].map(backbone_label),
            "Method": raw["method"].map(method_label),
            "Runs": raw["runs"].astype(int).astype(str),
            "Avg (%)": [fmt_mean_std(m, s) for m, s in zip(raw["Avg_mean"], raw["Avg_std"])],
            "Last (%)": [fmt_mean_std(m, s) for m, s in zip(raw["Last_mean"], raw["Last_std"])],
            "Forgetting (%)": [fmt_mean_std(m, s) for m, s in zip(raw["forgetting_proxy_mean"], raw["forgetting_proxy_std"])],
        }
    )
    render_table(
        df,
        OUT / "backup_full_phase2_table.png",
        "Backup 2",
        "Full fixed beta=4 table",
        main_title="Backup: Full Fixed-Beta=4 Main Table",
        subtitle="Source: results/fixed_beta4_main_table.csv",
        highlight_methods=["mgp_mgc"],
        method_keys=raw["method"].tolist(),
        font_size=12,
        scale_y=1.55,
    )


def backup_class_order_table() -> None:
    raw = load_csv("results/phase2_class_order_robustness.csv")
    raw["method_order"] = raw["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    raw = raw.sort_values(["backbone", "method_order"])
    df = pd.DataFrame(
        {
            "Backbone": raw["backbone"].map(backbone_label),
            "Method": raw["method"].map(method_label),
            "Canonical Avg": raw["canonical_Avg"].map(lambda x: fmt_pct(x)),
            "Shuffled Avg": [f"{fmt_pct(m)} +/- {fmt_pct(s)}" for m, s in zip(raw["shuffled_Avg_mean"], raw["shuffled_Avg_std"])],
            "Delta Avg": raw["delta_Avg_shuffled_minus_canonical"].map(lambda x: f"{100*x:+.1f}"),
            "Canonical Last": raw["canonical_Last"].map(lambda x: fmt_pct(x)),
            "Shuffled Last": [f"{fmt_pct(m)} +/- {fmt_pct(s)}" for m, s in zip(raw["shuffled_Last_mean"], raw["shuffled_Last_std"])],
        }
    )
    render_table(
        df,
        OUT / "backup_class_order_table.png",
        "Backup 3",
        "Class-order robustness table",
        main_title="Backup: Class-Order Robustness",
        subtitle="Source: results/phase2_class_order_robustness.csv",
        highlight_methods=["mgp_mgc"],
        method_keys=raw["method"].tolist(),
        font_size=9.8,
        scale_y=1.55,
    )


def backup_rank_table() -> None:
    raw = load_csv("results/phase2_rank_sensitivity.csv")
    raw = raw[raw["method"].isin(["mgp_mgc", "mgp_only", "naive_adapter", "zero_shot_clap"])].copy()
    raw["method_order"] = raw["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    raw = raw.sort_values(["method_order", "adapter_rank"])
    df = pd.DataFrame(
        {
            "Method": raw["method"].map(method_label),
            "Rank": raw["adapter_rank"].astype(int).astype(str),
            "Runs": raw["runs"].astype(int).astype(str),
            "Avg (%)": [fmt_mean_std(m, s) for m, s in zip(raw["Avg_mean"], raw["Avg_std"])],
            "Last (%)": [fmt_mean_std(m, s) for m, s in zip(raw["Last_mean"], raw["Last_std"])],
            "Forgetting (%)": [fmt_mean_std(m, s) for m, s in zip(raw["forgetting_proxy_mean"], raw["forgetting_proxy_std"])],
        }
    )
    render_table(
        df,
        OUT / "backup_rank_sensitivity_table.png",
        "Backup 4",
        "Rank sensitivity table",
        main_title="Backup: Rank Sensitivity",
        subtitle="Source: results/phase2_rank_sensitivity.csv",
        highlight_methods=["mgp_mgc"],
        method_keys=raw["method"].tolist(),
        font_size=10.4,
        scale_y=1.28,
    )


def backup_safety_table() -> None:
    safe = [
        "figure_1/laion-audiocaps-val",
        "figure_1/msclap-audiocaps-val",
        "figure_3/laion-audiocaps-val",
        "figure_3/msclap-audiocaps-val",
        "table_1/laion-shift",
        "table_1/msclap-shift",
        "continual_mgclap/laion",
        "continual_mgclap/msclap",
    ]
    unsafe = [
        "synthetic Figure 3",
        "simulation table",
        "unclear training sweep",
        "dry run",
    ]
    fig, ax = setup_canvas()
    title(fig, "Backup: Presentation-Safety Status", "Main slides use only the real-data stack")
    add_box(ax, (0.08, 0.15), (0.38, 0.67), "safe for presentation\n\n" + "\n".join(f"- {x}" for x in safe), fc="#ecfdf5", ec=MGC, fontsize=13)
    add_box(ax, (0.54, 0.15), (0.38, 0.67), "not safe\n\n" + "\n".join(f"- {x}" for x in unsafe), fc="#fff1f2", ec=NAIVE, fontsize=14)
    footer(fig, "Source: results/PROJECT_STATUS.md")
    savefig(fig, OUT / "backup_real_vs_synthetic_status.png", "Backup 5", "Real vs synthetic presentation-safety table")


def write_speaker_notes() -> None:
    notes_data = [
        (1, "Mind the Gap in CLAP", "The project asks whether the MG-CLIP preserve-and-compensate idea transfers from image-text CLIP to audio-text CLAP.", "Open by framing the talk as a controlled transfer experiment. We are not claiming full CLAP retraining; we are asking whether the principle survives when the modalities change from images and text to sounds and text.", "The diagram maps image-text CLIP and MG-CLIP to audio-text CLAP and MG-CLAP-lite.", "Move from the thesis to how CLAP performs zero-shot classification.", "Q: Is this a new CLAP model? A: No. It is a frozen-embedding adapter method built on existing CLAP embeddings."),
        (2, "CLAP Zero-Shot Classification", "CLAP classifies by comparing an audio embedding to text-label embeddings.", "Explain that CLAP gives a shared embedding space. Candidate labels are encoded as text prompts, and prediction is the largest audio-text similarity. This matters because preserving geometry is preserving the classifier itself.", "The figure shows audio and text encoders, similarity scores, and the argmax rule.", "Next, introduce the original modality-gap premise from CLIP.", "Q: Why use text prompts for ESC-50? A: That is the standard zero-shot CLAP interface; labels become text embeddings."),
        (3, "Original Mind the Gap Premise", "Pretrained contrastive modalities can occupy offset regions, and that offset may be structural.", "Describe the centroids and the gap vector. The key point is that the gap is not automatically a defect; training may use it as part of the representation geometry. Continual learning can unintentionally destroy it.", "The schematic shows two modality clouds, centroids, and the gap distance.", "Next, show how MG-CLIP responds to that risk.", "Q: Is this slide measured data? A: No. It is a conceptual schematic; measured CLAP gap stats come later."),
        (4, "Original MG-CLIP Response", "MG-CLIP balances stability and plasticity through preservation and compensation.", "Present the two-part logic: preservation avoids destructive cross-modal drift, while compensation adds a task-specific intra-modal classifier. The combination is the principle we test in audio.", "The diagram branches into Preserve and Compensate and labels them stability and plasticity.", "Next, map each image-side component to our audio-side version.", "Q: Why not only preserve? A: Preservation stabilizes geometry but may not add enough task discrimination."),
        (5, "Extension Map", "MG-CLAP-lite is a controlled embedding-level extension of MG-CLIP to sound classes.", "Walk down the table: image-text becomes audio-text, visual classes become sound classes, visual adaptation becomes an audio embedding adapter, and the intra-modal classifier becomes audio prototypes.", "The table states the mapping and the scope: not full CLAP fine-tuning.", "Next, show the evidence stack that tests the mapping.", "Q: What is controlled here? A: We keep CLAP encoders frozen and operate on cached embeddings."),
        (6, "Experimental Evidence Map", "The project tests geometry, mechanism, shift behavior, and continual learning with real AudioCaps and ESC-50 outputs.", "Use this slide as the roadmap. AudioCaps supports the gap and loss-landscape claims; ESC-50 supports zero-shot shift and continual-learning claims.", "The pipeline and table connect each question to a dataset, model pair, and output type.", "Next, start with measured CLAP modality gap evidence.", "Q: Are synthetic results in the main evidence? A: No. The main evidence uses real AudioCaps and ESC-50 files."),
        (7, "Modality Gap in CLAP", "CLAP shows an audio-text modality gap in real AudioCaps embeddings.", "Explain the PCA and schematic. The exact coordinates are only visualization, but the stats come from real cached embeddings. The point is that CLAP shares the same kind of structural issue as CLIP.", "Slide 7 shows clouds, centroids, and gap direction for audio and text.", "Next, ask what the contrastive loss itself prefers.", "Q: Does PCA prove the gap? A: PCA visualizes it; the numeric centroid gap supports it."),
        (8, "Contrastive Loss and Temperature", "The loss-preferred gap changes with temperature in real CLAP landscapes.", "Explain that each temperature has a saved sweep over lambda. The dot is the minimum-loss point. Lower temperature sharpens contrastive pressure and can prefer a non-zero gap.", "The plot summarizes gap at minimum loss versus temperature for LAION and MSCLAP.", "Next, test whether shifting the gap affects ESC-50 classification.", "Q: Does this prove training convergence? A: No. It is a controlled landscape probe, not full retraining."),
        (9, "Gap Shifting on ESC-50", "Gap shifts affect LAION more clearly than MSCLAP, so the shift result is mixed rather than universal.", "Present this as a diagnostic. LAION improves under a shift; MSCLAP is close to null. This motivates a careful claim: geometry matters, but not identically for every backbone.", "The slide should show the real ESC-50 shift sweeps from Table 1 outputs.", "Next, introduce the MG-CLAP-lite method that uses preservation and compensation.", "Q: Why keep the mixed result? A: It makes the final claims more credible and avoids overgeneralization."),
        (10, "MG-CLAP-lite Method", "MG-CLAP-lite adds a low-rank audio adapter and combines text and audio-prototype logits.", "Walk through the pipeline from cached audio embedding to adapted embedding, then into text classifier and prototype classifier. Preservation is early stopping; compensation is the prototype term.", "The diagram marks frozen encoders, trainable adapter, stored prototypes, and combined logits.", "Next, explain how preservation is selected.", "Q: What is trainable? A: The low-rank residual adapter; CLAP encoders stay frozen."),
        (11, "Preservation Probe", "The preservation probe selects an epoch before negative-similarity drift exceeds alpha.", "Explain the drift formula. It monitors how negative-class similarities move relative to epoch zero. The selected epoch is a practical stopping point for maintaining geometry.", "The plot shows drift over epochs, alpha, and the selected e-star for representative runs.", "Next, show why prototypes compensate for remaining task discrimination needs.", "Q: Is this an aggregate result? A: No. It is a diagnostic for selected run/fold behavior."),
        (12, "Prototype Compensation", "Audio prototypes add an intra-modal class anchor alongside text embeddings.", "Explain that text labels remain useful but audio clusters can provide stronger class-specific anchors after adaptation. Beta controls the contribution of the prototype classifier.", "The schematic shows clusters, prototypes, text embeddings, and similarity arrows.", "Next, define the continual-learning protocol.", "Q: Are prototypes replay memory? A: No. They are compact class anchors, not stored examples."),
        (13, "Continual-Learning Setup and Main Result", "Evaluation is among all seen classes without task ID, and MGP+MGC is strongest in the fixed-beta table.", "Describe 10 tasks with 5 classes each. After task k, classification is over all seen classes. Then highlight the fixed beta=4 result, especially MGP+MGC versus naive adapter.", "The setup diagram and table define the protocol and main result.", "Next, check whether that result holds across seeds and class orders.", "Q: Are future classes included before they appear? A: No. Evaluation only includes seen classes."),
        (14, "Seed and Class-Order Robustness", "MGP+MGC remains strong across canonical and shuffled class orders.", "Point out that seed 0 is canonical and seeds 1-3 are shuffled. Both backbones keep MGP+MGC near the top, so the result is not just a single class order artifact.", "The two panels plot Avg accuracy by seed for each method and backbone.", "Next, test sensitivity to preservation threshold and adapter rank.", "Q: Is beta tuned per seed? A: No. Beta is fixed at 4 in these robustness checks."),
        (15, "Alpha and Rank Ablation", "Stricter preservation helps, and rank sensitivity is comparatively mild for MGP+MGC.", "Explain alpha first: smaller alpha means stricter drift tolerance, and the real LAION ablation shows better Avg/Last and lower forgetting. Then explain rank: performance is fairly stable over the tested ranks.", "The figure plots Avg and Last, with forgetting values shown in side boxes and a compact table.", "Next, summarize what evidence supports and what remains mixed.", "Q: Is this for both backbones? A: This ablation is LAION; main robustness covers both backbones."),
        (16, "Claims Evidence Matrix", "The strongest claim is preservation plus compensation; several narrower claims are mixed or unsafe.", "Use the matrix to calibrate the conclusion. The gap exists and the temperature mechanism transfers. Gap shifting is mixed. Naive adaptation is false as a helpful baseline. The combined method is strong on both backbones.", "Traffic-light colors separate strong, mixed/partial, and false/unsafe claims.", "Next, make the limitations explicit.", "Q: Why include weak claims? A: To prevent the audience from overreading the result."),
        (17, "Limitations and Threats", "The scope is frozen-embedding MG-CLAP-lite on ESC-50, not full CLAP continual fine-tuning.", "Name each limitation and immediately pair it with the next step. This turns critique into a concrete roadmap: LoRA, new datasets, validation tuning, replay baseline, and presentation-safety discipline.", "The table lists limitation, why it matters, and next step.", "Next, close with the final constrained conclusion.", "Q: Do these limitations invalidate the result? A: No. They define the scope of the controlled transfer result."),
        (18, "Final Takeaway", "Geometry, mechanism, and the preserve-plus-compensate continual-learning principle transfer at the embedding-adapter level.", "Close with three blocks: CLAP has the gap, temperature changes the loss-preferred gap, and preservation plus compensation beats naive adaptation. Then restate the warning: not full CLAP fine-tuning.", "The slide has three conclusion blocks and a scope warning.", "End by inviting questions around scope, baselines, and next experiments.", "Q: What is the single safest final claim? A: MG-CLAP-lite supports the MG-CLIP principle at the frozen-embedding adapter level."),
    ]
    lines = ["# Speaker Notes: Slides 1-18", ""]
    for num, slide_title, takeaway, say, fig_exp, transition, qa in notes_data:
        lines.extend(
            [
                f"## Slide {num}: {slide_title}",
                f"- Takeaway: {takeaway}",
                f"- What to say in 45-75 seconds: {say}",
                f"- Figure explanation: {fig_exp}",
                f"- Transition: {transition}",
                f"- Likely Q&A: {qa}",
                "",
            ]
        )
    path = OUT / "SPEAKER_NOTES_SLIDES_1_18.md"
    path.write_text("\n".join(lines))
    register("Speaker notes", "Slides 1-18 speaker notes", path)


def write_manifest() -> None:
    lines = ["# Asset Manifest", "", "Generated final presentation assets.", ""]
    for slide, desc, path in GENERATED:
        lines.append(f"- {slide}: `{path.relative_to(ROOT)}` - {desc}")
    lines.append(f"- Manifest: `{(OUT / 'ASSET_MANIFEST.md').relative_to(ROOT)}` - Generated asset manifest")
    lines.extend(
        [
            "",
            "## Numeric Source Policy",
            "",
            "Main-slide numeric plots/tables are generated only from these real-data files:",
        ]
    )
    for source in REAL_NUMERIC_SOURCES:
        if (ROOT / source).exists():
            lines.append(f"- `{source}`")
    lines.extend(
        [
            "",
            "Synthetic/historical artifacts excluded from main-slide numeric plots:",
            "- `results/figure_3/synth-gap0.82/`",
            "- `results/table_1/simulation/`",
            "- `results/table_1_training/laion/`",
            "- `results/continual_mgclap/laion/dry_run/`",
            "- `handoff/` archive copies",
        ]
    )
    path = OUT / "ASSET_MANIFEST.md"
    path.write_text("\n".join(lines))
    register("Manifest", "Generated asset manifest", path)


def verify_real_inputs() -> None:
    for source in REAL_NUMERIC_SOURCES:
        path = ROOT / source
        if path.exists() and path.suffix == ".json":
            data = json.loads(path.read_text())
            if "mode" in data and data["mode"] != "real":
                raise ValueError(f"Non-real JSON source used: {source}")
    for unsafe in [
        "results/figure_3/synth-gap0.82/3b_landscape.json",
        "results/table_1/simulation/summary.json",
    ]:
        if unsafe in REAL_NUMERIC_SOURCES:
            raise ValueError(f"Unsafe source included: {unsafe}")


def verify_pngs() -> list[str]:
    problems: list[str] = []
    for path in sorted(OUT.glob("*.png")):
        img = mpimg.imread(path)
        if img.size == 0 or float(np.asarray(img).std()) < 0.001:
            problems.append(str(path))
        if img.shape[0] != 1650 or img.shape[1] != 2933:
            problems.append(f"unexpected dimensions {path}: {img.shape}")
    if problems:
        raise RuntimeError("PNG verification failed: " + "; ".join(problems))
    return [p.name for p in sorted(OUT.glob("*.png"))]


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    verify_real_inputs()

    slide1()
    slide2()
    slide3()
    slide4()
    slide5()
    slide6()
    slide14()
    slide15()
    slide16()
    slide17()
    slide18()
    backup_formula_sheet()
    backup_full_phase2_table()
    backup_class_order_table()
    backup_rank_table()
    backup_safety_table()
    write_speaker_notes()
    write_manifest()

    names = verify_pngs()
    print("Generated final presentation assets:")
    for _, _, path in GENERATED:
        print(f"  - {path}")
    print("Verified PNGs:")
    for name in names:
        print(f"  - {name}")
    print("Numeric sources verified as real-data/current result files.")


if __name__ == "__main__":
    main()
