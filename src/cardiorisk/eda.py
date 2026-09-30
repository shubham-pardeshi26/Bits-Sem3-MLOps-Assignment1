"""Exploratory figures for the cleaned four-site table.

Each plot_* function takes the DataFrame from cardiorisk.data.load_processed() and returns a
matplotlib Figure; `python -m cardiorisk.eda` saves them as reports/figures/eda_<name>.png.
Figures use the object API instead of pyplot, so nothing piles up in pyplot's global registry and a
notebook never renders one twice (it needs `%matplotlib inline`, which pyplot would otherwise
trigger, to show a returned Figure as an image).
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cardiorisk import config
from cardiorisk.data import load_processed
from cardiorisk.features import INPUT_FEATURES, add_clinical_features

# One look for every project figure (the training plots import these). The class pair is
# colour-blind safe (simulated CVD delta E 25) and both colours clear 3:1 on the surface.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CLASS_COLORS = {0: BLUE, 1: ORANGE}
CLASS_NAMES = {0: "No heart disease", 1: "Heart disease"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "cardiorisk_blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "cardiorisk_diverging", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c23636"]
)
SAVE = {"dpi": 150, "bbox_inches": "tight"}

sns.set_theme(
    style="whitegrid",
    palette=[BLUE, ORANGE],
    rc={
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e1e0d9",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "figure.titlesize": "x-large",
        "legend.frameon": False,
    },
)

SITE_NAMES = {
    "cleveland": "Cleveland",
    "hungarian": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}
# column -> (panel title, axis label with unit, histogram bin width)
DISTRIBUTIONS = {
    "age": ("Age", "age (years)", 5),
    "trestbps": ("Resting blood pressure", "trestbps (mm Hg)", 10),
    "chol": ("Serum cholesterol", "chol (mg/dl)", 25),
    "thalach": ("Maximum heart rate on exercise", "thalach (bpm)", 10),
    "oldpeak": ("Exercise-induced ST depression", "oldpeak (mm)", 0.5),
    "hr_achieved_frac": ("Share of age-predicted max heart rate", "thalach / (220 - age)", 0.05),
}
# column -> (panel title, UCI code -> label), codes as documented in heart-disease.names
CATEGORIES = {
    "cp": (
        "Chest pain type",
        {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    ),
    "exang": ("Exercise-induced angina", {0: "No", 1: "Yes"}),
    "slope": ("Slope of peak exercise ST segment", {1: "Upsloping", 2: "Flat", 3: "Downsloping"}),
    "thal": ("Thallium stress test", {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}),
    "sex": ("Sex", {0: "Female", 1: "Male"}),
    "restecg": ("Resting ECG", {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}),
}
NOT_RECORDED = "Not recorded"


def _subtitle(ax, text: str) -> None:
    ax.set_title(text, fontweight="normal", color=SECONDARY)


def _class_legend(fig: Figure) -> None:
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in CLASS_COLORS]
    fig.legend(handles=handles, loc="outside lower center", ncols=2)


def plot_class_balance(df: pd.DataFrame) -> Figure:
    """Class counts overall, next to the per-site split that the pooled numbers hide."""
    fig = Figure(figsize=(11, 4.2), layout="constrained")
    ax_all, ax_site = fig.subplots(1, 2, width_ratios=[1, 2.2])

    counts = df[config.TARGET].value_counts().reindex(list(CLASS_COLORS), fill_value=0)
    bars = ax_all.bar(
        [CLASS_NAMES[c] for c in counts.index],
        counts,
        width=0.55,
        color=list(CLASS_COLORS.values()),
    )
    ax_all.bar_label(bars, labels=[f"{n} ({n / len(df):.0%})" for n in counts], padding=3)
    ax_all.set(title=f"All sites (n = {len(df)})", ylabel="Patients", ylim=(0, 1.2 * counts.max()))
    ax_all.grid(axis="x", visible=False)

    share = pd.crosstab(df["site"], df[config.TARGET], normalize="index")
    share = share.reindex(index=list(SITE_NAMES), columns=list(CLASS_COLORS), fill_value=0)
    sizes = df["site"].value_counts()
    ticks = [f"{SITE_NAMES[s]}\nn = {sizes.get(s, 0)}" for s in share.index]
    left = np.zeros(len(share))
    for c, text_color in zip(CLASS_COLORS, ("white", INK), strict=True):
        pct = 100 * share[c].to_numpy()
        bars = ax_site.barh(
            ticks, pct, left=left, height=0.6, color=CLASS_COLORS[c], edgecolor=SURFACE, lw=1
        )
        labels = [f"{p:.0f}%" if p >= 8 else "" for p in pct]  # a label must fit its segment
        ax_site.bar_label(bars, labels=labels, label_type="center", color=text_color)
        left += pct
    ax_site.invert_yaxis()
    ax_site.set(title="Share of patients by site", xlabel="Patients (%)", xlim=(0, 100))
    ax_site.grid(axis="y", visible=False)

    low, high = (100 * share[1]).min(), (100 * share[1]).max()
    fig.suptitle(
        f"Class balance: {100 * counts[1] / len(df):.0f}% with heart disease overall, "
        f"{low:.0f}% to {high:.0f}% depending on the hospital"
    )
    _class_legend(fig)
    return fig


def plot_missingness_by_site(df: pd.DataFrame) -> Figure:
    """Missing share of every model input per site: each hospital skipped different tests."""
    missing = df[INPUT_FEATURES].isna()
    pct = 100 * missing.groupby(df["site"]).mean().reindex(list(SITE_NAMES)).rename(SITE_NAMES)
    pct.loc["All sites"] = 100 * missing.mean()
    annot = pct.map(lambda v: "" if v == 0 else "<1" if v < 0.5 else f"{v:.0f}")

    fig = Figure(figsize=(11, 3.8), layout="constrained")
    ax = fig.subplots()
    sns.heatmap(
        pct,
        mask=pct.eq(0),  # complete cells stay blank, so only the gaps draw the eye
        annot=annot,
        fmt="",
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor=SURFACE,
        cbar_kws={"label": "Missing (%)"},
        ax=ax,
    )
    ax.axhline(len(SITE_NAMES), color=SURFACE, lw=6)  # set the pooled row apart
    ax.set(xlabel="Model input (UCI attribute)", ylabel="")
    ax.tick_params(length=0)
    ax.grid(False)
    fig.suptitle("Missing values by site (% of patients)")
    _subtitle(
        ax,
        "ca, thal and slope were rarely recorded outside Cleveland; "
        "Switzerland never recorded chol",
    )
    return fig


def plot_numeric_distributions(df: pd.DataFrame) -> Figure:
    """Continuous inputs per class, each class scaled to 100% so unequal class sizes compare."""
    data = add_clinical_features(df).join(df[config.TARGET])
    fig = Figure(figsize=(12, 7), layout="constrained")
    axes = fig.subplots(2, 3).flat
    for ax, (col, (title, xlabel, width)) in zip(axes, DISTRIBUTIONS.items(), strict=True):
        sns.histplot(
            data,
            x=col,
            hue=config.TARGET,
            palette=CLASS_COLORS,
            stat="percent",
            common_norm=False,
            binwidth=width,
            element="step",
            alpha=0.15,
            linewidth=1.5,
            legend=False,
            ax=ax,
        )
        missing = data[col].isna().mean()
        ax.set(title=title, xlabel=f"{xlabel}  ·  {missing:.0%} missing" if missing else xlabel)
        ax.set_ylabel("Share of class (%)" if ax.get_subplotspec().is_first_col() else "")
    fig.suptitle("Distributions by diagnosis")
    _class_legend(fig)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Pearson correlations (point-biserial for binary inputs) over pairwise-complete rows."""
    columns = ["age", "trestbps", "chol", "thalach", "oldpeak", "sex", "fbs", "exang"]
    corr = (
        df[[*columns, config.TARGET]]
        .corr()
        .rename(index={config.TARGET: "disease"}, columns={config.TARGET: "disease"})
    )
    corr = corr.iloc[1:, :-1]  # lower triangle without the uninformative diagonal
    fig = Figure(figsize=(7.8, 6.6), layout="constrained")
    ax = fig.subplots()
    sns.heatmap(
        corr,
        mask=np.triu(np.ones(corr.shape, dtype=bool), k=1),
        annot=True,
        fmt=".2f",
        cmap=DIVERGING,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        linecolor=SURFACE,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        ax=ax,
    )
    ax.tick_params(length=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.grid(False)
    fig.suptitle("Correlation between inputs and with disease")
    _subtitle(ax, "Pearson r on pairwise-complete rows (point-biserial for binary inputs)")
    return fig


def plot_categorical_rates(df: pd.DataFrame) -> Figure:
    """Disease rate per category with patient counts; unrecorded values get their own bar."""
    overall = 100 * df[config.TARGET].mean()
    fig = Figure(figsize=(13, 7), layout="constrained")
    for ax, (col, (title, codes)) in zip(fig.subplots(2, 3).flat, CATEGORIES.items(), strict=True):
        level = df[col].map(codes).fillna(NOT_RECORDED)
        stats = df.groupby(level)[config.TARGET].agg(["mean", "size"])
        stats = stats.reindex([*codes.values(), NOT_RECORDED]).dropna()
        ax.barh(
            stats.index,
            100 * stats["mean"],
            height=0.6,
            color=[GRAY if label == NOT_RECORDED else ORANGE for label in stats.index],
        )
        for y, (m, n) in enumerate(stats.values):  # a column right of 100%, clear of the line
            ax.text(104, y, f"{m:.0%}  (n = {n:.0f})", va="center")
        ax.axvline(overall, color=INK, lw=1)
        ax.set(
            title=title, xlim=(0, 145), xticks=range(0, 101, 25), xlabel="With heart disease (%)"
        )
        ax.set_ylim(len(codes) + 0.5, -0.5)  # same bar thickness in every panel, first code on top
        ax.grid(axis="y", visible=False)
    handles = [
        Patch(color=ORANGE, label="Recorded value"),
        Patch(color=GRAY, label=NOT_RECORDED),
        Line2D([], [], color=INK, lw=1, label=f"All patients ({overall:.0f}%)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    fig.suptitle("Heart disease rate by category")
    return fig


def plot_max_hr_vs_age(df: pd.DataFrame) -> Figure:
    """Peak exercise heart rate against the 220 - age estimate, the basis of hr_achieved_frac."""
    data = df.dropna(subset=["thalach"])
    achieved = add_clinical_features(data)["hr_achieved_frac"]  # the model's own definition
    fig = Figure(figsize=(9, 6), layout="constrained")
    ax = fig.subplots()
    for c in CLASS_COLORS:
        group = data[data[config.TARGET] == c]
        below = (achieved[group.index] < 0.85).mean()
        ax.scatter(
            group["age"],
            group["thalach"],
            s=24,
            color=CLASS_COLORS[c],
            alpha=0.8,
            edgecolors=SURFACE,
            linewidths=0.6,
            label=f"{CLASS_NAMES[c]}: {below:.0%} below 85% of predicted",
        )
    ages = np.array([data["age"].min() - 2, data["age"].max() + 2])
    ax.plot(ages, 220 - ages, color=INK, lw=1.5, label="220 - age (age-predicted maximum)")
    ax.plot(
        ages,
        0.85 * (220 - ages),
        color=INK,
        lw=1.2,
        ls=(0, (5, 3)),
        label="85% of predicted maximum (hr_achieved_frac = 0.85)",
    )
    ax.set(xlabel="Age (years)", ylabel="Maximum heart rate achieved (bpm)", xlim=tuple(ages))
    fig.suptitle("Maximum heart rate on exercise vs age")
    _subtitle(
        ax, "Patients with heart disease more often fall short of their age-predicted maximum"
    )
    fig.legend(loc="outside lower center", ncols=2, markerscale=1.6)
    return fig


FIGURES = {
    "class_balance": plot_class_balance,
    "missingness_by_site": plot_missingness_by_site,
    "numeric_distributions": plot_numeric_distributions,
    "correlation_heatmap": plot_correlation_heatmap,
    "categorical_rates": plot_categorical_rates,
    "max_hr_vs_age": plot_max_hr_vs_age,
}


def main() -> None:
    df = load_processed()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, plot in FIGURES.items():
        path = config.FIGURES_DIR / f"eda_{name}.png"
        plot(df).savefig(path, **SAVE)  # object-API figures need no close()
        print(path)


if __name__ == "__main__":
    main()
