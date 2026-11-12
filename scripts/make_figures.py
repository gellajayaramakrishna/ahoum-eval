"""Renders reports/figures/routing_curve.png from reports/lite_metrics.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "reports", "lite_metrics.json")))["routing_A"]
x = [100 * p["llm_share"] for p in r]
ink, muted, grid, surf = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7, 4), facecolor=surf)
ax.set_facecolor(surf)
series = [("MAE_on_student_rows", "Uncertainty routing", "#2a78d6"),
          ("MAE_random_rows_same_size", "Random routing", "#eb6834")]
for key, label, c in series:
    y = [p[key] for p in r]
    ax.plot(x, y, color=c, lw=2, marker="o", ms=8, mec=surf, mew=2, label=label, zorder=3)
    ax.annotate(label, (x[0], y[0]), xytext=(8, 0), textcoords="offset points", va="center", color=ink, fontsize=9)
ax.set_xlabel("Facets sent to the LLM judge (%)", color=muted)
ax.set_ylabel("MAE of facets kept by the lite judge", color=muted)
ax.set_title("Lite judge error on the facets it keeps (5-fold, unseen conversations)", color=ink, fontsize=11, loc="left")
ax.set_ylim(0, 0.55)
ax.set_xlim(-3, 60)
ax.grid(axis="y", color=grid, lw=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(grid)
ax.tick_params(colors=muted)
ax.legend(frameon=False, loc="lower left", labelcolor=ink)
fig.tight_layout()
out = os.path.join(ROOT, "reports", "figures", "routing_curve.png")
fig.savefig(out, dpi=160, facecolor=surf)
print("wrote", out)
