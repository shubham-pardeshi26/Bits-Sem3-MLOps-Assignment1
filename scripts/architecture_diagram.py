"""Draw the CardioRisk system architecture to reports/figures/architecture.png.

Run: .venv/bin/python scripts/architecture_diagram.py
Plain matplotlib on a 100 x 62 data-unit canvas, so the figure is reproducible without a
diagramming tool. Box widths are sized for DejaVu Sans, matplotlib's default font.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Same palette as src/cardiorisk/eda.py.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, SECONDARY = "#0b0b0b", "#52514e"
PANEL, EXTERNAL, ARTIFACT = "#f4f4f2", "#e6e5e1", "#fde6dc"
OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

STYLES = {
    "component": ("white", BLUE),
    "artifact": (ARTIFACT, ORANGE),
    "external": (EXTERNAL, GRAY),
}
# (centre, width) of the four columns shared by the data row and the serving row.
COLS = ((8.5, 12), (25, 16), (44.5, 18), (64, 16))
CI_X, CI_W = 87.5, 19
TEXT = {"ha": "center", "va": "center", "zorder": 3}


def panel(ax, x0, y0, x1, y1, title):
    shape = {"boxstyle": "round,pad=0,rounding_size=1", "fc": PANEL, "ec": "none", "zorder": 0}
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, **shape))
    ax.text(
        x0 + 1, y1 - 1.1, title, ha="left", va="top", fontsize=8.5, weight="bold", color=SECONDARY
    )


def box(ax, x, y, w, title, detail="", kind="component", h=7.4):
    """A box centred on (x, y); returns its edge midpoints for arrows."""
    face, edge = STYLES[kind]
    shape = {"boxstyle": "round,pad=0,rounding_size=0.8", "fc": face, "ec": edge, "lw": 1.1}
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, zorder=2, **shape))
    lines = detail.count("\n") + 1 if detail else 0
    top = y + (1.9 + 1.2 * lines) / 2  # title line 1.5 units, gap 0.4, detail lines 1.2 each
    ax.text(x, top - 0.75, title, fontsize=8.5, weight="bold", color=INK, **TEXT)
    if detail:
        y_detail = top - 1.9 - 0.6 * lines
        ax.text(x, y_detail, detail, fontsize=7, color=SECONDARY, linespacing=1.25, **TEXT)
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}


def col_box(ax, col, y, title, detail="", kind="component"):
    x, w = COLS[col]
    return box(ax, x, y, w, title, detail, kind)


def arrow(ax, start, end, color=SECONDARY, rad=0.0, dashed=False, label="", label_at=None):
    style = {"arrowstyle": "-|>", "mutation_scale": 9, "connectionstyle": f"arc3,rad={rad}"}
    style |= {"color": color, "lw": 1.1, "ls": (0, (3, 2)) if dashed else "-"}
    ax.add_patch(FancyArrowPatch(start, end, shrinkA=0, shrinkB=0, zorder=1, **style))
    if label:
        ax.text(*label_at, label, fontsize=7, color=color, **TEXT)


def shift(point, dx):
    """A point on a box edge, dx units along the edge from its midpoint."""
    return (point[0] + dx, point[1])


def main():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set(xlim=(0, 100), ylim=(0, 62), aspect="equal")
    ax.axis("off")

    # Data and training
    title = "Data and training  (make data, make train; rerun by the CI train job)"
    panel(ax, 1, 35.5, 73.5, 61.5, title)
    uci = col_box(ax, 0, 51.5, "UCI archive", "4 hospitals\n920 rows", "external")
    data = col_box(ax, 1, 51.5, "cardiorisk.data", "download, SHA-256\ncheck, dedupe\n918 rows")
    csv = col_box(
        ax, 2, 51.5, "Clean dataset", "data/processed/\nheart_disease_clean.csv", "artifact"
    )
    detail = "4 model families\nGridSearchCV 5-fold\nhold-out + LOSO"
    train = col_box(ax, 3, 51.5, "cardiorisk.train", detail)
    reg = box(ax, 10, 41, 15, "MLflow registry", "cardiorisk-classifier\n@champion")
    detail = "mlflow.db + mlruns/\ncardiorisk-heart-disease: 4 + 85 nested runs"
    track = box(ax, 36.75, 41, 33.5, "MLflow tracking", detail)
    detail = "cardiorisk_model.joblib\n+ model_card.json"
    model = col_box(ax, 3, 41, "Serving artifact", detail, "artifact")
    arrow(ax, uci["r"], data["l"])
    arrow(ax, data["r"], csv["l"])
    arrow(ax, csv["r"], train["l"])
    arrow(ax, shift(train["b"], -5), track["r"])
    arrow(ax, track["l"], reg["r"])
    arrow(ax, train["b"], model["t"], ORANGE)

    # CI/CD
    panel(ax, 76, 27, 99, 61.5, "CI/CD  (GitHub Actions)")
    jobs = [
        ("lint + test", "ruff, pytest"),
        ("train", "model + MLflow artifacts"),
        ("docker", "build image + smoke test"),
        ("deploy-kind", "kind cluster: smoke +\nPrometheus target check"),
        ("publish", "push to main only"),
    ]
    ci = [box(ax, CI_X, 55.2 - i * 6.2, CI_W, *job, h=5.2) for i, job in enumerate(jobs)]
    for upper, lower in pairwise(ci):
        arrow(ax, upper["b"], lower["t"])
    detail = "GHCR by default\nsha-<7> + latest"
    registry = box(ax, CI_X, 22, CI_W, "Container registry", detail, "external", h=6.6)
    arrow(ax, ci[-1]["b"], registry["t"], ORANGE)
    arrow(ax, model["r"], ci[2]["l"], ORANGE)

    # Serving
    title = "Serving on Kubernetes  (namespace cardiorisk; overlays minikube, kind, cloud)"
    panel(ax, 1, 14.5, 73.5, 32.5, title)
    note = (
        "image source: minikube image build (minikube), kind load of the docker job's image (CI),"
    )
    note += " registry pull (cloud)"
    ax.text(2, 29.3, note, ha="left", va="top", fontsize=7, color=SECONDARY)
    client = col_box(ax, 0, 22, "Clients", "POST /predict\nGET /docs", "external")
    ingress = col_box(ax, 1, 22, "Ingress", "nginx class\nhttp://127.0.0.1/")
    service = col_box(ax, 2, 22, "Service", "LoadBalancer :8081\n(minikube tunnel)")
    detail = "cardiorisk-api pods\nFastAPI + uvicorn\nnon-root, HPA 2-5"
    pods = col_box(ax, 3, 22, "Deployment", detail)
    arrow(ax, client["r"], ingress["l"], BLUE)
    arrow(ax, ingress["r"], service["l"], BLUE)
    arrow(ax, service["r"], pods["l"], BLUE)
    arrow(ax, shift(client["b"], 2), shift(service["b"], -4), BLUE, rad=0.25)
    arrow(ax, registry["l"], pods["r"], ORANGE, label="pull\n(cloud)", label_at=(75.2, 19.9))

    # Monitoring and logging, in the same columns as the serving row
    panel(ax, 1, 0.5, 73.5, 12.5, "Monitoring and logging")
    graf = col_box(ax, 1, 5.5, "Grafana", "provisioned dashboard\nCardioRisk API")
    prom = col_box(ax, 2, 5.5, "Prometheus", "Kubernetes pod discovery\nscrapes /metrics")
    detail = "stdout per request\nand prediction\n(kubectl logs)"
    logs = col_box(ax, 3, 5.5, "JSON logs", detail)
    arrow(ax, shift(pods["b"], 3), shift(logs["t"], 3), dashed=True)
    label = {"label": "scrape /metrics", "label_at": (47, 13.5)}
    arrow(ax, shift(prom["t"], 5), shift(pods["b"], -4), dashed=True, **label)
    arrow(ax, prom["l"], graf["r"])

    # Legend, under the registry
    kinds = ["component", "artifact", "external"]
    for i, kind in enumerate(kinds):
        y = 15 - i * 2.1
        box(ax, 79, y, 4, "", kind=kind, h=1.5)
        ax.text(82.5, y, kind, va="center", fontsize=7, color=SECONDARY)
    flows = [
        (SECONDARY, False, "data / CI flow"),
        (ORANGE, False, "model and image"),
        (BLUE, False, "request path"),
        (SECONDARY, True, "logs and metrics"),
    ]
    for i, (color, dashed, text) in enumerate(flows):
        y = 8.4 - i * 2.1
        arrow(ax, (77, y), (81, y), color, dashed=dashed)
        ax.text(82.5, y, text, va="center", fontsize=7, color=SECONDARY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
