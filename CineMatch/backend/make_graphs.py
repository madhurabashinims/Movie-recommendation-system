import sys, os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── OUTPUT FOLDER ─────────────────────────────────────────────
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── COLORS ────────────────────────────────────────────────────
ORANGE = "#f97316"
GREEN  = "#16a34a"
BLUE   = "#2563eb"
TEAL   = "#0891b2"
GOLD   = "#ca8a04"
RED    = "#dc2626"
LGRAY  = "#d1d5db"

# ── GLOBAL STYLE ──────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        # Paper-grade font: Linux Libertine / fallback to Palatino / DejaVu Serif
        "font.family":        "serif",
        "font.serif":         ["Palatino Linotype", "Palatino", "Book Antiqua",
                               "Linux Libertine", "DejaVu Serif", "Times New Roman"],
        "font.size":          9,

        "axes.titlesize":     10,
        "axes.titleweight":   "bold",
        "axes.titlepad":      10,

        "axes.labelsize":     9,
        "axes.labelweight":   "regular",

        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,

        "axes.linewidth":     0.8,

        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "grid.color":         "#cccccc",

        "figure.dpi":         220,

        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",

        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   3,
        "ytick.major.size":   3,

        "legend.fontsize":    8,
        "legend.title_fontsize": 8,
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "#cccccc",
        "legend.borderpad":   0.5,
        "legend.labelspacing":0.35,
        "legend.handlelength":1.4,
        "legend.handletextpad":0.5,
    })

def _out(name):
    """Return output path inside graphs/ folder."""
    return os.path.join(OUT_DIR, os.path.basename(name))

def _save(fig, name):
    path = _out(name)
    pdf  = path.replace(".png", ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=220)
    fig.savefig(pdf,  bbox_inches="tight")   # vector version
    plt.close(fig)
    print(f"  {path}")

# ── SHARED LEGEND HELPER ──────────────────────────────────────
def _place_legend_below(ax, handles, ncol=3, y_offset=-0.22):
    """
    Place a legend below the axes, anchored to the bottom-centre.
    y_offset is in axes-fraction units (negative = below x-axis).
    """
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        borderaxespad=0.,
        framealpha=0.95,
        edgecolor="#cccccc",
    )

# ══════════════════════════════════════════════════════════════
# GRAPH 1 — Recommender Ablation
def graph1_recommender(data, out="graph1_recommender_ablation.png"):
    labels = [r["label"] for r in data]
    p10    = [r["p10"]   for r in data]
    ndcg   = [r["ndcg"]  for r in data]

    x = np.arange(len(labels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(9, 5))
    # Extra bottom margin for the legend
    fig.subplots_adjust(top=0.88, bottom=0.28)

    b1 = ax.bar(x - w/2, p10,  w, color=BLUE, alpha=0.88, zorder=3)
    b2 = ax.bar(x + w/2, ndcg, w, color=TEAL, alpha=0.88, zorder=3)

    # Highlight optimal config
    for bar in [b1[-1], b2[-1]]:
        bar.set_edgecolor(GOLD)
        bar.set_linewidth(1.8)

    # Value labels — small, above each bar
    for bar, v in zip(list(b1) + list(b2), p10 + ndcg):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.0006,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0.20, 0.305)
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))

    ax.set_title(
    "Multimodal Fusion Ablation",
    loc="center",
    fontsize=10, fontweight="bold", pad=10
)



    _place_legend_below(ax, handles=[
        mpatches.Patch(color=BLUE, label="Precision@10"),
        mpatches.Patch(color=TEAL, label="NDCG@10"),
        mpatches.Patch(edgecolor=GOLD, facecolor="none", lw=2, label="Optimal config"),
    ], ncol=3, y_offset=-0.26)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════
# GRAPH 2 — Ultra-Specific Search ILD
def graph2_ultra_ild(rows, base_cov, full_cov, out="graph2_ultra_ild.png"):
    if not rows:
        return

    labels = [r["label"]    for r in rows]
    base   = [r["base_ild"] for r in rows]
    full   = [r["full_ild"] for r in rows]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, n * 0.65)))
    fig.subplots_adjust(left=0.30, right=0.78, top=0.88, bottom=0.20)

    y = np.arange(n)
    h = 0.30

    ax.barh(y - h/2, base, h, color=ORANGE, alpha=0.88, label="Baseline",  zorder=3)
    ax.barh(y + h/2, full, h, color=GREEN,  alpha=0.88, label="Proposed",  zorder=3)

    x_max = max(max(base), max(full))
    for i, (b, f) in enumerate(zip(base, full)):
        ax.text(f + x_max * 0.012, i + h/2, f"{f:.3f}", va="center", fontsize=7.5)
        ax.text(b + x_max * 0.012, i - h/2, f"{b:.3f}", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Intra-List Diversity (ILD)")
    ax.set_title("Ultra-Specific Search — Diversity Comparison", loc="center")

    _place_legend_below(ax, handles=[
        mpatches.Patch(color=ORANGE, label="Baseline"),
        mpatches.Patch(color=GREEN,  label="Proposed"),
    ], ncol=2, y_offset=-0.18)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════
# GRAPH 3 — Negation Handling
def graph3_negation(rows, out="graph3_negation.png"):
    rows = [r for r in rows if r.get("has_excl")]
    if not rows:
        return

    labels = [r["label"]        for r in rows]
    base   = [r["base_ner"]*100 for r in rows]
    full   = [r["full_ner"]*100 for r in rows]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, n * 0.80)))
    fig.subplots_adjust(left=0.30, right=0.78, top=0.88, bottom=0.20)

    y = np.arange(n)
    h = 0.30

    ax.barh(y - h/2, base, h, color=ORANGE, alpha=0.88, label="Baseline", zorder=3)
    ax.barh(y + h/2, full, h, color=GREEN,  alpha=0.88, label="Proposed", zorder=3)

    x_max = max(max(base), max(full))
    for i, (b, f) in enumerate(zip(base, full)):
        ax.text(f + x_max * 0.015, i + h/2, f"{f:.0f}%", va="center", fontsize=8)
        ax.text(b + x_max * 0.015, i - h/2, f"{b:.0f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Negation Enforcement Rate (%)")
    ax.set_title("Negation Handling Performance", loc="center")

    _place_legend_below(ax, handles=[
        mpatches.Patch(color=ORANGE, label="Baseline"),
        mpatches.Patch(color=GREEN,  label="Proposed"),
    ], ncol=2, y_offset=-0.18)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════
# GRAPH 4 — Chatbot Diversity
def graph4_chatbot(rows, base_cov, full_cov, out="graph4_chatbot.png"):
    if not rows:
        return

    labels = [r["label"]    for r in rows]
    base   = [r["base_ild"] for r in rows]
    full   = [r["full_ild"] for r in rows]

    x = np.arange(len(labels))
    w = 0.30

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.subplots_adjust(top=0.88, bottom=0.26)

    ax.bar(x - w/2, base, w, color=ORANGE, alpha=0.88, label="Baseline", zorder=3)
    ax.bar(x + w/2, full, w, color=GREEN,  alpha=0.88, label="Proposed", zorder=3)

    y_max = max(max(base), max(full))
    for i, (b, f) in enumerate(zip(base, full)):
        ax.text(i + w/2,  f + y_max * 0.012, f"{f:.3f}", ha="center", fontsize=7.5)
        ax.text(i - w/2,  b + y_max * 0.012, f"{b:.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ILD")
    ax.set_title("Emotion-Aware Chatbot — Diversity Comparison", loc="center")

    _place_legend_below(ax, handles=[
        mpatches.Patch(color=ORANGE, label="Baseline"),
        mpatches.Patch(color=GREEN,  label="Proposed"),
    ], ncol=2, y_offset=-0.22)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════
# GRAPH 5 — WILT Preference Separation
def graph5_wilt(wilt, out="graph5_wilt.png"):
    if not wilt:
        return

    liked    = wilt["liked_avg"]
    disliked = wilt["disliked_avg"]
    labels   = wilt["labels"]

    x = np.arange(len(labels))
    w = 0.30

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.subplots_adjust(top=0.88, bottom=0.26)

    ax.bar(x - w/2, liked,    w, color=GREEN, alpha=0.88, label=r"Liked  ($\geq$4 stars)", zorder=3)
    ax.bar(x + w/2, disliked, w, color=RED,   alpha=0.88, label=r"Disliked ($\leq$2 stars)", zorder=3)

    # Gap annotation — drawn AFTER bars so positions are known
    y_max = max(max(liked), max(disliked))
    for i, (l, d) in enumerate(zip(liked, disliked)):
        gap = abs(l - d)
        top = max(l, d)
        # Bracket slightly above the taller bar
        bracket_y = top + y_max * 0.04
        ax.annotate(
            "",
            xy  =(i + w/2, bracket_y),
            xytext=(i - w/2, bracket_y),
            arrowprops=dict(arrowstyle="<->", color="#444444",
                            lw=0.9, shrinkA=0, shrinkB=0),
        )
        ax.text(i, bracket_y + y_max * 0.025,
                f"Δ={gap:.2f}",
                ha="center", va="bottom", fontsize=7, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Compatibility Score")
    ax.set_ylim(0, y_max * 1.22)
    ax.set_title("Preference Separation (WILT)", loc="center")

    _place_legend_below(ax, handles=[
        mpatches.Patch(color=GREEN, label=r"Liked  ($\geq$4 stars)"),
        mpatches.Patch(color=RED,   label=r"Disliked ($\leq$2 stars)"),
    ], ncol=2, y_offset=-0.22)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════
# GRAPH 6 — System Comparison Matrix
def graph6_matrix(out="graph6_matrix.png"):
    systems  = ["LightGCN", "SGL", "Multimodal", "ConvRec",
                "ContrastMM", "BIGRec", "Ours"]
    features = ["CF", "Text", "Visual", "Emotion", "Negation", "LLM"]

    mat = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.88, bottom=0.16)

    cmap = matplotlib.colors.ListedColormap(["#f8d7da", "#d4edda"])
    ax.imshow(mat.T, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Cell text
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            symbol = r"$\checkmark$" if mat[i, j] else r"$\times$"
            color  = "#155724" if mat[i, j] else "#721c24"
            ax.text(i, j, symbol, ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    ax.set_xticks(range(len(systems)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(systems, fontsize=8.5)
    ax.set_yticklabels(features, fontsize=8.5)

    # Highlight "Ours" column
    for j in range(len(features)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (len(systems)-1 - 0.49, j - 0.49), 0.98, 0.98,
            linewidth=1.5, edgecolor=GOLD, facecolor="none",
            boxstyle="square,pad=0", zorder=5,
        ))

    ax.set_title("System Capability Comparison", loc="center")

    # Grid lines between cells
    for x in np.arange(-0.5, len(systems), 1):
        ax.axvline(x, color="#cccccc", lw=0.5)
    for y in np.arange(-0.5, len(features), 1):
        ax.axhline(y, color="#cccccc", lw=0.5)

    ax.tick_params(length=0)

    _place_legend_below(ax, handles=[
        mpatches.Patch(facecolor="#d4edda", edgecolor="#999", label="Supported"),
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#999", label="Not supported"),
        mpatches.Patch(edgecolor=GOLD, facecolor="none", lw=1.5, label="Our system"),
    ], ncol=3, y_offset=-0.18)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_results.json")
    args = parser.parse_args()

    with open(args.results) as f:
        r = json.load(f)

    _style()

    print(f"Writing graphs to ./{OUT_DIR}/")
    graph1_recommender(r["recommender"])
    graph2_ultra_ild(r["ultra"]["rows"], r["ultra"]["base_cov"], r["ultra"]["full_cov"])
    graph3_negation(r["ultra"]["rows"])
    graph4_chatbot(r["chatbot"]["rows"], r["chatbot"]["base_cov"], r["chatbot"]["full_cov"])
    graph5_wilt(r.get("wilt"))
    graph6_matrix()
    print("Done.")

if __name__ == "__main__":
    main()