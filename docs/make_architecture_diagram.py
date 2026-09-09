"""
make_architecture_diagram.py
-----------------------------
Generates docs/architecture.png — a visual overview of the project's
data + ML pipeline, from raw text to predicted category.

Run with:
    python docs/make_architecture_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Style config
# ---------------------------------------------------------------------------
STAGE_COLOR = "#1f4e79"
STAGE_FACE = "#eaf1fb"
ACCENT = "#2e75b6"
TEXT_COLOR = "#1a1a1a"
BG_COLOR = "#ffffff"

fig, ax = plt.subplots(figsize=(13, 6.5), dpi=200)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")

# Title
ax.text(6.5, 6.15, "Multi-Class News Classifier — Architecture",
        fontsize=18, fontweight="bold", ha="center", color=STAGE_COLOR)
ax.text(6.5, 5.75, "Bag of Words + Classical Machine Learning Pipeline",
        fontsize=11, ha="center", color="#555555", style="italic")


def draw_box(x, y, w, h, title, subtitle, face=STAGE_FACE, edge=STAGE_COLOR, title_size=11):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=TEXT_COLOR, zorder=3)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=8.3, color="#444444", zorder=3, wrap=True)


def draw_arrow(x1, y, x2, label=None):
    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=2, color=ACCENT, zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        ax.text((x1 + x2) / 2, y + 0.22, label, ha="center", fontsize=8,
                color="#555555")


# ---------------------------------------------------------------------------
# Main pipeline (top row)
# ---------------------------------------------------------------------------
y0 = 3.6
box_h = 1.35
box_w = 1.85
gap = 0.55

stages = [
    ("Raw News\nText", "CSV: text, category\n(news_dataset.csv)"),
    ("Text\nPreprocessing", "lowercase, remove\npunctuation & stopwords,\nstemming"),
    ("Bag of Words", "CountVectorizer\n(unigrams + bigrams)"),
    ("ML Classifier", "Naive Bayes /\nLogistic Regression /\nLinear SVM"),
    ("Predicted\nCategory", "Business, Sports, Tech,\nEntertainment, Politics"),
]

x = 0.4
xs = []
for title, subtitle in stages:
    xs.append(x)
    draw_box(x, y0, box_w, box_h, title, subtitle)
    x += box_w + gap

for i in range(len(xs) - 1):
    x1 = xs[i] + box_w
    x2 = xs[i + 1]
    draw_arrow(x1 + 0.05, y0 + box_h / 2, x2 - 0.05)

# ---------------------------------------------------------------------------
# Bottom row: supporting components
# ---------------------------------------------------------------------------
y1 = 1.0
sub_w = 2.55
sub_h = 1.15
sub_gap = 0.5

sub_stages = [
    ("train.py", "Trains & compares models,\nsaves best model + vectorizer"),
    ("models/*.pkl", "Saved model, vectorizer\n& label artifacts (joblib)"),
    ("predict.py / app.py", "CLI script & Streamlit\napp for live predictions"),
]

sx = 2.5
sxs = []
for title, subtitle in sub_stages:
    sxs.append(sx)
    draw_box(sx, y1, sub_w, sub_h, title, subtitle,
              face="#fdf3e7", edge="#c07c1a", title_size=10.5)
    sx += sub_w + sub_gap

for i in range(len(sxs) - 1):
    x1 = sxs[i] + sub_w
    x2 = sxs[i + 1]
    draw_arrow(x1 + 0.05, y1 + sub_h / 2, x2 - 0.05)

# Connect training flow up to bag-of-words/model stage
ax.annotate(
    "", xy=(xs[3] + box_w / 2, y0), xytext=(sxs[0] + sub_w / 2, y1 + sub_h),
    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.5, linestyle="dashed"),
)
ax.text((xs[3] + box_w / 2 + sxs[0] + sub_w / 2) / 2 + 0.3, (y0 + y1 + sub_h) / 2,
        "trains on", fontsize=8, color="#777777", style="italic")

ax.annotate(
    "", xy=(sxs[2] + sub_w / 2, y1 + sub_h), xytext=(xs[4] + box_w / 2, y0),
    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.5, linestyle="dashed"),
)
ax.text((sxs[2] + sub_w / 2 + xs[4] + box_w / 2) / 2 - 0.35, (y0 + y1 + sub_h) / 2 - 0.35,
        "loads &\npredicts", fontsize=8, color="#777777", style="italic")

# Footer
ax.text(6.5, 0.25, "Multi-Class-News-Classifier  |  scikit-learn  •  pandas  •  Streamlit",
        ha="center", fontsize=8.5, color="#888888")

plt.tight_layout()
out_path = "docs/architecture.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved architecture diagram to {out_path}")
