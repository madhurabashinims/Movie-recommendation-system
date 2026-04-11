"""
CineMatch — Clean Graph Generation
====================================
Drop-in replacement for the graph section of comprehensive_eval.py.

All legends are placed OUTSIDE the plot area using bbox_to_anchor so they
never overlap bars. Call generate_all_graphs(results) at the end of main().

Usage inside comprehensive_eval.py:
    from graphs_clean import generate_all_graphs
    generate_all_graphs(
        rec_ablation   = rec_ablation,
        ultra_rows     = ultra_rows,
        ner_rows_only  = ner_rows_only,
        chat_rows      = chat_rows,
        base_cov_ultra = base_cov_ultra,
        full_cov_ultra = full_cov_ultra,
        base_cov_chat  = base_cov_chat,
        full_cov_chat  = full_cov_chat,
        wilt_data      = wilt_data,
    )

Or run standalone for demo with hardcoded sample data:
    python graphs_clean.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── colour palette ─────────────────────────────────────────────────────────────
ORANGE  = "#f97316"   # baseline
GREEN   = "#16a34a"   # proposed
BLUE    = "#2563eb"   # precision
TEAL    = "#0891b2"   # ndcg
GOLD    = "#d97706"   # highlight border
RED     = "#dc2626"   # disliked scores
LGRAY   = "#e5e7eb"   # grid / subtle fills


# ── shared style ───────────────────────────────────────────────────────────────
def _set_style():
    plt.rcParams.update({
        "font.family":          "DejaVu Sans",
        "font.size":            10,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.alpha":           0.25,
        "grid.linestyle":       "--",
        "figure.dpi":           160,
        "axes.facecolor":       "white",
        "figure.facecolor":     "white",
        "savefig.facecolor":    "white",
        "savefig.edgecolor":    "none",
    })


def _save(fig, name):
    fig.savefig(name, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 1 — Hybrid Recommender Ablation (bar chart, vertical)
# ══════════════════════════════════════════════════════════════════════════════
def graph1_recommender(rec_ablation, out="graph1_recommender_ablation.png"):
    """
    rec_ablation: list of (label, alpha, beta, gamma, P@10, NDCG@10)
    """
    labels    = [r[0] for r in rec_ablation]
    p10_vals  = [r[4] for r in rec_ablation]
    ndcg_vals = [r[5] for r in rec_ablation]

    # Extra top margin to fit legend above the bars
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(top=0.78)   # shrink axes top so legend has room

    x = np.arange(len(labels))
    w = 0.32

    b1 = ax.bar(x - w/2, p10_vals,  w, color=BLUE,  alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, ndcg_vals, w, color=TEAL,  alpha=0.85, zorder=3)

    # Gold border on the optimal (last) bar pair
    for bar in [b1[-1], b2[-1]]:
        bar.set_edgecolor(GOLD)
        bar.set_linewidth(2.5)

    # Value labels above each bar
    for bar, v in zip(list(b1) + list(b2), p10_vals + ndcg_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.0008,
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=7.5, color="#374151"
        )

    # Improvement arrow from NCF-only to Optimal (P@10)
    ax.annotate(
        "",
        xy     = (x[-1] - w/2, p10_vals[-1] + 0.003),
        xytext = (x[0]  - w/2, p10_vals[0]  + 0.003),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.3),
        zorder=5,
    )
    pct = (p10_vals[-1] - p10_vals[0]) / p10_vals[0] * 100
    ax.text(
        (x[0] + x[-1]) / 2 - w / 2,
        p10_vals[-1] + 0.006,
        f"+{pct:.1f}%",
        ha="center", fontsize=8.5, color="red", fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0.20, 0.295)
    ax.set_title(
        "Hybrid Recommender — Ablation Study: Modality Fusion Configurations\n"
        "Evaluated on 1 000 users · MovieLens 25M",
        fontsize=11, fontweight="bold", pad=12
    )

    # Legend placed ABOVE the axes, outside the plot
    legend_handles = [
        mpatches.Patch(color=BLUE, label="Precision@10"),
        mpatches.Patch(color=TEAL, label="NDCG@10"),
        mpatches.Patch(edgecolor=GOLD, facecolor="none", lw=2, label="Optimal config"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),   # above the axes
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor=LGRAY,
    )

    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 2 — Ultra-Specific ILD: Baseline vs Proposed (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
def graph2_ultra_ild(ultra_rows, base_cov, full_cov,
                     out="graph2_ultra_ild.png"):
    """
    ultra_rows: list of dicts with keys label, base_ild, full_ild
    """
    if not ultra_rows:
        print("  [SKIP] graph2 — no ultra_rows data")
        return

    labels    = [r["label"] for r in ultra_rows]
    base_ilds = [r["base_ild"] for r in ultra_rows]
    full_ilds = [r["full_ild"] for r in ultra_rows]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(9, max(4.5, n * 0.72)))
    fig.subplots_adjust(right=0.72)   # leave right margin for legend

    yp = np.arange(n)
    bar_h = 0.32

    ax.barh(yp - bar_h/2, base_ilds, bar_h, color=ORANGE, alpha=0.82,
            label="Baseline (plain SBERT, no novelties)")
    ax.barh(yp + bar_h/2, full_ilds, bar_h, color=GREEN,  alpha=0.87,
            label="Proposed (N4 + N5 + N6)")

    # Value labels at end of each bar
    x_max = max(full_ilds + base_ilds)
    for i, (b, f) in enumerate(zip(base_ilds, full_ilds)):
        ax.text(f + 0.008, i + bar_h/2, f"{f:.3f}",
                va="center", fontsize=8.5, color=GREEN, fontweight="bold")
        ax.text(b + 0.008, i - bar_h/2, f"{b:.3f}",
                va="center", fontsize=8.5, color=ORANGE)

    ax.set_yticks(yp)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Intra-List Diversity (ILD)", fontsize=10)
    ax.set_xlim(0, x_max + 0.18)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, lw=1.2)
    ax.text(0.502, n - 0.6, "ILD = 0.5", fontsize=7.5, color="gray", va="top")

    ax.set_title(
        "Ultra-Specific Search — Recommendation Diversity (ILD)\n"
        "Higher ILD = more diverse, less repetitive results",
        fontsize=11, fontweight="bold", pad=10
    )

    # Legend to the RIGHT of the axes, outside
    ax.legend(
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.01, 0.65),
        frameon=True,
        framealpha=0.95,
        edgecolor=LGRAY,
    )

    # Coverage annotation below legend, right side
    ax.text(
        1.01, 0.25,
        f"Catalog coverage\nBaseline: {base_cov:.1%}\nProposed: {full_cov:.1%}",
        transform=ax.transAxes,
        ha="left", va="center", fontsize=8.5, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=LGRAY, lw=0.8),
    )

    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 3 — Negation Enforcement Rate (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
def graph3_negation(ner_rows_only, out="graph3_negation_enforcement.png"):
    """
    ner_rows_only: list of dicts with keys label, base_ner (0–1), full_ner (0–1)
    """
    if not ner_rows_only:
        print("  [SKIP] graph3 — no NER rows")
        return

    labels    = [r["label"] for r in ner_rows_only]
    base_ners = [r["base_ner"] * 100 for r in ner_rows_only]
    full_ners = [r["full_ner"] * 100 for r in ner_rows_only]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(9, max(4.0, n * 0.82)))
    fig.subplots_adjust(right=0.70)

    yp    = np.arange(n)
    bar_h = 0.32

    ax.barh(yp - bar_h/2, base_ners, bar_h, color=ORANGE, alpha=0.82,
            label="Baseline (no negation handling)")
    ax.barh(yp + bar_h/2, full_ners, bar_h, color=GREEN,  alpha=0.87,
            label="Proposed (N4: negation subtraction)")

    for i, (b, f) in enumerate(zip(base_ners, full_ners)):
        ax.text(f + 0.8, i + bar_h/2, f"{f:.0f}%",
                va="center", fontsize=9, color=GREEN, fontweight="bold")
        ax.text(b + 0.8, i - bar_h/2, f"{b:.0f}%",
                va="center", fontsize=9, color=ORANGE)

    ax.set_yticks(yp)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Negation Enforcement Rate (%)", fontsize=10)
    ax.set_xlim(0, 122)
    ax.axvline(100, color="gray", linestyle=":", alpha=0.4, lw=1)

    # Mean improvement annotation
    avg_base = np.mean(base_ners)
    avg_full = np.mean(full_ners)
    ax.text(
        0.98, 0.04,
        f"Mean:  Baseline {avg_base:.0f}%  →  Proposed {avg_full:.0f}%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=LGRAY, lw=0.8),
    )

    ax.set_title(
        "N4: Negation-Aware Vector Subtraction — Enforcement Rate\n"
        "% of top-10 results that correctly exclude the specified genres",
        fontsize=11, fontweight="bold", pad=10
    )

    ax.legend(
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor=LGRAY,
    )

    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 4 — Chatbot ILD: Baseline vs Proposed (vertical bar)
# ══════════════════════════════════════════════════════════════════════════════
def graph4_chatbot(chat_rows, base_cov, full_cov,
                   out="graph4_chatbot_ild.png"):
    if not chat_rows:
        print("  [SKIP] graph4 — no chat_rows data")
        return

    labels = [r["label"] for r in chat_rows]
    base_c = [r["base_ild"] for r in chat_rows]
    full_c = [r["full_ild"] for r in chat_rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.subplots_adjust(top=0.76)

    x_c = np.arange(len(labels))
    w   = 0.32

    ax.bar(x_c - w/2, base_c, w, color=ORANGE, alpha=0.82,
           label="Baseline (plain SBERT)")
    ax.bar(x_c + w/2, full_c, w, color=GREEN,  alpha=0.87,
           label="Proposed (N7: Emotion Pipeline)")

    y_top = max(full_c + base_c)
    for i, (b, f) in enumerate(zip(base_c, full_c)):
        ax.text(i + w/2, f + 0.006, f"{f:.3f}",
                ha="center", fontsize=8.5, color=GREEN, fontweight="bold")
        ax.text(i - w/2, b + 0.006, f"{b:.3f}",
                ha="center", fontsize=8.5, color=ORANGE)

    ax.set_xticks(x_c)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, y_top + 0.14)
    ax.set_ylabel("Intra-List Diversity (ILD)", fontsize=10)

    ax.set_title(
        "Emotion-Aware Chatbot — Recommendation Diversity by Emotional Context\n"
        "Baseline (plain SBERT) vs Proposed (N7: 7-Class Emotion-to-Genre Pipeline)",
        fontsize=11, fontweight="bold", pad=12
    )

    # Legend ABOVE axes
    ax.legend(
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor=LGRAY,
    )

    # Coverage — bottom right, inside axes but in empty space
    ax.text(
        0.98, 0.97,
        f"Catalog coverage  |  Baseline: {base_cov:.1%}   Proposed: {full_cov:.1%}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#6b7280",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LGRAY, lw=0.8),
    )

    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 5 — Will I Like This? Score Separation
# ══════════════════════════════════════════════════════════════════════════════
def graph5_wilt(wilt_data, out="graph5_wilt_separation.png"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.subplots_adjust(top=0.76)

    liked    = wilt_data["liked_avg"]
    disliked = wilt_data["disliked_avg"]
    sep      = wilt_data["separation"]
    x_labels = wilt_data["labels"]
    x_w      = np.arange(len(x_labels))
    w        = 0.28

    ax.bar(x_w - w/2, liked,    w, color=GREEN, alpha=0.87,
           label="Avg score — movies user liked (≥ 4★)")
    ax.bar(x_w + w/2, disliked, w, color=RED,   alpha=0.78,
           label="Avg score — movies user disliked (≤ 2★)")

    y_top = max(liked) + 0.14
    for i, (l, d, s) in enumerate(zip(liked, disliked, sep)):
        ax.text(i - w/2, l + 0.01, f"{l:.2f}",
                ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
        ax.text(i + w/2, d + 0.01, f"{d:.2f}",
                ha="center", fontsize=10.5, color=RED, fontweight="bold")
        # Double-headed arrow between bar tops
        ax.annotate(
            "",
            xy     = (i + w/2, d + 0.025),
            xytext = (i - w/2, l + 0.025),
            arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.3),
        )
        ax.text(i, max(l, d) + 0.055, f"gap = {s:.2f}",
                ha="center", fontsize=9.5, fontweight="bold", color="#374151")

    ax.set_xticks(x_w)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Compatibility Score", fontsize=10)
    ax.set_title(
        "Will I Like This? — Score Separation\n"
        "Text-only baseline vs Text + Poster (proposed)",
        fontsize=11, fontweight="bold", pad=12
    )

    # Legend ABOVE axes
    ax.legend(
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor=LGRAY,
    )

    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 6 — Feature Comparison Matrix (heatmap)
# ══════════════════════════════════════════════════════════════════════════════
def graph6_comparison_matrix(out="graph6_comparison_matrix.png"):
    systems = [
        "LightGCN\n2020", "SGL\n2021", "Multimodal\n2022",
        "Conv.Rec\n2022", "Contrast.MM\n2023", "BIGRec\n2024",
        "CineMatch\n(Ours)",
    ]
    features = [
        "CF / NCF", "Text embed.", "Visual feat.",
        "Emotion\naware", "Negation\nhandling", "LLM\nexplain",
    ]
    mat = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.imshow(mat.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(systems)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_yticklabels(features, fontsize=10)

    for i in range(len(systems)):
        for j in range(len(features)):
            sym   = "✓" if mat[i, j] else "✗"
            color = "white" if mat[i, j] else "#6b7280"
            ax.text(i, j, sym, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    # Gold border around CineMatch column
    ax.add_patch(plt.Rectangle(
        (len(systems) - 1.5, -0.5), 1, len(features),
        lw=3, edgecolor=GOLD, facecolor="none", zorder=5,
    ))

    # Column separator lines
    for i in range(len(systems) - 1):
        ax.axvline(i + 0.5, color="white", lw=0.8, alpha=0.4)
    for j in range(len(features) - 1):
        ax.axhline(j + 0.5, color="white", lw=0.8, alpha=0.4)

    ax.spines[:].set_visible(False)
    ax.set_title(
        "CineMatch vs Related Work (2020–2024): Feature Comparison\n"
        "✓ = Supported   ✗ = Not supported   (gold border = our system)",
        fontsize=11, fontweight="bold", pad=10
    )

    fig.tight_layout(pad=1.5)
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def generate_all_graphs(
    rec_ablation,
    ultra_rows,
    ner_rows_only,
    chat_rows,
    base_cov_ultra,
    full_cov_ultra,
    base_cov_chat,
    full_cov_chat,
    wilt_data,
):
    _set_style()
    print("\nGenerating graphs...")
    graph1_recommender(rec_ablation)
    graph2_ultra_ild(ultra_rows, base_cov_ultra, full_cov_ultra)
    graph3_negation(ner_rows_only)
    graph4_chatbot(chat_rows, base_cov_chat, full_cov_chat)
    graph5_wilt(wilt_data)
    graph6_comparison_matrix()
    print("All graphs saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO  (python graphs_clean.py)
#  Uses the same hardcoded values that were in the original comprehensive_eval
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _set_style()

    demo_rec_ablation = [
        ("NCF only\n(baseline)",          1.00, 0.00, 0.00, 0.2497, 0.2709),
        ("NCF + Text\n(no poster)",        0.70, 0.30, 0.00, 0.2505, 0.2730),
        ("Text-dominant\n(0.5/0.3/0.2)",   0.50, 0.30, 0.20, 0.2336, 0.2530),
        ("Visual-boost\n(0.65/0.2/0.15)",  0.65, 0.20, 0.15, 0.2505, 0.2730),
        ("Optimal\n(0.6/0.25/0.15)",       0.60, 0.25, 0.15, 0.2583, 0.2791),
    ]

    # Sample ultra rows — replace with real output from comprehensive_eval
    demo_ultra_rows = [
        {"label": "Animation+Animals",       "base_ild": 0.421, "full_ild": 0.563, "has_excl": False, "base_ner": None, "full_ner": None},
        {"label": "Thriller (not Horror)",   "base_ild": 0.388, "full_ild": 0.541, "has_excl": True,  "base_ner": 1.0,  "full_ner": 1.0 },
        {"label": "Romance (not Drama)",     "base_ild": 0.345, "full_ild": 0.512, "has_excl": True,  "base_ner": 0.5,  "full_ner": 1.0 },
        {"label": "SciFi (no Romance/Comedy)","base_ild": 0.402, "full_ild": 0.558, "has_excl": True, "base_ner": 0.5,  "full_ner": 1.0 },
        {"label": "Family Drama (no Action)", "base_ild": 0.371, "full_ild": 0.527, "has_excl": True, "base_ner": 1.0,  "full_ner": 1.0 },
        {"label": "Kids Adventure",           "base_ild": 0.389, "full_ild": 0.545, "has_excl": False,"base_ner": None, "full_ner": None},
        {"label": "Psych Thriller (no Comedy)","base_ild": 0.361,"full_ild": 0.511,"has_excl": True,  "base_ner": 0.8,  "full_ner": 1.0 },
        {"label": "Sports Drama",             "base_ild": 0.412, "full_ild": 0.569, "has_excl": False,"base_ner": None, "full_ner": None},
    ]
    demo_ner_rows = [r for r in demo_ultra_rows if r["has_excl"]]

    demo_chat_rows = [
        {"label": "Soft Grief",  "base_ild": 0.41, "full_ild": 0.58},
        {"label": "Anger",       "base_ild": 0.38, "full_ild": 0.55},
        {"label": "Happy",       "base_ild": 0.44, "full_ild": 0.62},
        {"label": "Deep Grief",  "base_ild": 0.39, "full_ild": 0.57},
        {"label": "Bored",       "base_ild": 0.43, "full_ild": 0.60},
        {"label": "Stressed",    "base_ild": 0.40, "full_ild": 0.56},
    ]

    demo_wilt = {
        "labels":       ["Text-only\n(baseline)", "Text + Poster\n(proposed)"],
        "liked_avg":    [0.52, 0.61],
        "disliked_avg": [0.44, 0.38],
        "separation":   [0.08, 0.23],
    }

    generate_all_graphs(
        rec_ablation   = demo_rec_ablation,
        ultra_rows     = demo_ultra_rows,
        ner_rows_only  = demo_ner_rows,
        chat_rows      = demo_chat_rows,
        base_cov_ultra = 0.031,
        full_cov_ultra = 0.031,
        base_cov_chat  = 0.018,
        full_cov_chat  = 0.018,
        wilt_data      = demo_wilt,
    )