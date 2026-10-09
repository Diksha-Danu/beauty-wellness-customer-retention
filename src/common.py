"""Shared paths, plotting style and helpers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 7
for p in (RAW, PROC, FIG, REP, MODELS):
    p.mkdir(parents=True, exist_ok=True)

NAVY, CORAL, SAGE, SAND, GREY = "#2C3E50", "#E07A5F", "#81B29A", "#F2CC8F", "#95A5A6"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlecolor": NAVY, "figure.dpi": 150,
})


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def update_json(name, new):
    """Merge `new` into reports/<name>.json."""
    path = REP / name
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(new)
    path.write_text(json.dumps(cur, indent=1, default=str))
    return cur


def flowchart(steps, fname, title, cols=4):
    rows = (len(steps) + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 2.6, rows * 1.9))
    ax.axis("off"); ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    pos = []
    for i, (head, sub) in enumerate(steps):
        r, c = divmod(i, cols)
        c = c if r % 2 == 0 else cols - 1 - c
        x, y = c + .07, rows - r - .93
        ax.add_patch(FancyBboxPatch((x, y), .86, .8, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=[NAVY, "#3D5A73"][i % 2], ec="none"))
        ax.text(x + .43, y + .62, head, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text(x + .43, y + .28, sub, ha="center", va="center", color="#ECF0F1", fontsize=7)
        pos.append((x + .43, y + .4))
    for a, b in zip(pos[:-1], pos[1:]):
        if abs(a[1] - b[1]) < .01:
            d = .45 if b[0] > a[0] else -.45
            ax.annotate("", xy=(b[0] - d, b[1]), xytext=(a[0] + d, a[1]),
                        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.8))
        else:
            ax.annotate("", xy=(b[0], b[1] + .42), xytext=(a[0], a[1] - .42),
                        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.8))
    ax.set_title(title, loc="left")
    save_fig(fname)
