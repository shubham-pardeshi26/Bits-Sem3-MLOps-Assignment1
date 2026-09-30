"""Model selection with MLflow tracking, from the cleaned table to the served model.

    python -m cardiorisk.train

Each candidate family is tuned by grid search around the full feature pipeline and logged as one
MLflow run, with a nested run per grid point. The champion is the family with the best mean CV
ROC-AUC (the hold-out split never influences selection); it is registered under the 'champion'
alias, scored on hospitals it never saw (leave-one-site-out) and saved for serving.
"""

import hashlib
import json
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import skops.io
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    check_scoring,
    classification_report,
    make_scorer,
    precision_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from cardiorisk import config
from cardiorisk.data import load_processed
from cardiorisk.eda import BLUE, GRAY, INK, SAVE, SEQUENTIAL, SITE_NAMES
from cardiorisk.features import INPUT_FEATURES, build_pipeline
from cardiorisk.predict import LABELS, THRESHOLD

SEED = config.RANDOM_STATE
METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]
# zero_division=0: a model that never predicts disease has no precision to measure, not an error.
SCORING = {m: m for m in METRICS} | {"precision": make_scorer(precision_score, zero_division=0)}
TITLES = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

# family -> (estimator, grid over its hyper-parameters, run description). Grids stay coarse: 734
# training rows cannot rank finer settings reliably, and the whole run has to fit in a CI job.
CANDIDATES = {
    "baseline_majority": (
        DummyClassifier(strategy="most_frequent"),
        {},
        "Majority-class baseline: always predicts the most common training class, so its ROC-AUC "
        "is 0.5 by construction. Every real model has to beat it.",
    ),
    "logistic_regression": (
        LogisticRegression(solver="saga", max_iter=5000, random_state=SEED),
        {
            "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "l1_ratio": [0.0, 0.5, 1.0],  # L2, elastic net, L1
            "class_weight": [None, "balanced"],
        },
        "Regularised logistic regression (saga solver) on scaled, one-hot and imputed inputs: "
        "linear, interpretable and hard to beat on small tabular data.",
    ),
    "random_forest": (
        RandomForestClassifier(n_jobs=1, random_state=SEED),  # the search already runs in parallel
        {
            "n_estimators": [200, 500],
            "max_depth": [4, 8, None],
            "min_samples_leaf": [1, 3, 10],
            "max_features": ["sqrt", 0.5],
        },
        "Random forest: averaged decorrelated trees capture interactions (age with heart rate, "
        "sex with chest pain) without scaling; depth and leaf size bound the variance.",
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
        {
            "learning_rate": [0.03, 0.1],
            "max_leaf_nodes": [4, 8, 16],
            "l2_regularization": [0.0, 1.0],
        },
        "Histogram gradient boosting: shallow trees fitted in sequence; learning rate, tree size "
        "and the L2 penalty trade bias against variance.",
    ),
}


def _sha256(table: pd.DataFrame) -> str:
    """Content hash of a table as CSV: `shasum -a 256` of the cleaned file when read from it."""
    return hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()


def _pretty(family: str) -> str:
    return family.replace("_", " ").capitalize()


def _unprefixed(params: dict) -> dict:
    return {name.removeprefix("model__"): value for name, value in params.items()}


def _cv_scores(row: pd.Series) -> dict:
    """<metric>_mean and <metric>_std of one GridSearchCV candidate."""
    return {f"{m}_{s}": float(row[f"{s}_test_{m}"]) for m in METRICS for s in ("mean", "std")}


def _confusion_figure(y_true, y_pred, family: str) -> Figure:
    fig = Figure(figsize=(5.2, 4.6), layout="constrained")
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No disease", "Disease"],
        cmap=SEQUENTIAL,
        colorbar=False,
        ax=ax,
        text_kw={"fontsize": 14},
    )
    ax.grid(False)
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"{_pretty(family)}: hold-out confusion matrix\n(threshold {THRESHOLD})")
    return fig


def _roc_figure(y_true, proba, family: str) -> Figure:
    fig = Figure(figsize=(5.4, 5.2), layout="constrained")
    ax = fig.subplots()
    RocCurveDisplay.from_predictions(
        y_true,
        proba,
        name=_pretty(family),
        ax=ax,
        curve_kwargs={"color": BLUE, "lw": 2},
        plot_chance_level=True,
        chance_level_kw={"color": GRAY, "lw": 1, "ls": "-"},
    )
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)")
    ax.set_title("ROC curve on the hold-out set")
    ax.legend(loc="lower right")
    return fig


def _pr_figure(y_true, proba, family: str) -> Figure:
    fig = Figure(figsize=(5.4, 5.2), layout="constrained")
    ax = fig.subplots()
    PrecisionRecallDisplay.from_predictions(
        y_true,
        proba,
        name=_pretty(family),
        ax=ax,
        curve_kwargs={"color": BLUE, "lw": 2},
        plot_chance_level=True,
        chance_level_kw={"color": GRAY, "lw": 1, "ls": "-"},
    )
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision-recall curve on the hold-out set")
    ax.legend(loc="lower left")
    return fig


def _importance_figure(result, family: str) -> Figure:
    order = np.argsort(result.importances_mean)
    fig = Figure(figsize=(7, 5.2), layout="constrained")
    ax = fig.subplots()
    ax.barh(
        np.array(INPUT_FEATURES)[order],
        result.importances_mean[order],
        xerr=result.importances_std[order],
        height=0.6,
        color=BLUE,
        error_kw={"ecolor": INK, "elinewidth": 1},
    )
    ax.axvline(0, color=INK, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Drop in hold-out ROC-AUC when the input is shuffled (mean ± sd, 20 shuffles)")
    ax.set_title(f"{_pretty(family)}: permutation importance of the raw inputs")
    return fig


def _comparison_figure(results: dict, champion: str) -> Figure:
    """CV mean ± sd (dot and whisker) and hold-out score (diamond); the champion is highlighted."""
    families = list(results)[::-1]  # first candidate on top
    fig = Figure(figsize=(15, 1.6 + 0.6 * len(families)), layout="constrained")
    axes = fig.subplots(1, len(METRICS), sharey=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        for y, family in enumerate(families):
            cv, test = results[family]["cv"], results[family]["test"]
            color = BLUE if family == champion else GRAY
            ax.errorbar(
                cv[f"{metric}_mean"],
                y,
                xerr=cv[f"{metric}_std"],
                fmt="o",
                ms=8,
                color=color,
                elinewidth=2,
            )
            # Hollow, so a CV dot that coincides with the test score stays visible inside it.
            ax.scatter(
                test[metric],
                y,
                marker="D",
                s=60,
                facecolors="none",
                edgecolors=color,
                linewidths=1.8,
                zorder=3,
            )
        ax.set(title=TITLES[metric], xlim=(0.4, 1.02))
    labels = [f"{_pretty(f)}\n(champion)" if f == champion else _pretty(f) for f in families]
    axes[0].set_yticks(range(len(families)), labels)
    axes[0].set_ylim(-0.6, len(families) - 0.4)
    handles = [
        Line2D([], [], color=GRAY, marker="o", ms=8, lw=2, label="5-fold CV mean ± sd"),
        Line2D(
            [],
            [],
            color=GRAY,
            marker="D",
            ms=7,
            mfc="none",
            mew=1.8,
            ls="",
            label="Hold-out test (20%)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2)
    fig.suptitle("Model comparison (champion chosen on CV ROC-AUC only)")
    return fig


def _loso_figure(loso: dict, holdout: dict) -> Figure:
    """Per-site scores of the champion trained on the other three sites."""
    labels = [
        f"{SITE_NAMES.get(site, site)}\nn = {s['n']}, {s['prevalence']:.0%} disease"
        for site, s in loso.items()
    ]
    fig = Figure(figsize=(13, 1.8 + 0.8 * len(loso)), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, metric in zip(axes, ["roc_auc", "accuracy", "recall"], strict=True):
        bars = ax.barh(labels, [s[metric] for s in loso.values()], height=0.6, color=BLUE)
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="white")  # clear of the line
        ax.axvline(holdout[metric], color=INK, lw=1)
        ax.set(title=TITLES[metric], xlim=(0, 1.12), xticks=np.linspace(0, 1, 5))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    line = Line2D([], [], color=INK, lw=1, label="Random 80/20 hold-out, all sites mixed")
    fig.legend(handles=[line], loc="outside lower center")
    fig.suptitle("Leave-one-site-out: trained on three hospitals, scored on the unseen fourth")
    return fig


def _table(head: list[str], rows: list[list[str]]) -> str:
    lines = [head, ["---"] * len(head), *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def _markdown(results: dict, champion: str, loso: dict, n_train: int, n_test: int) -> str:
    head = ["Family", "CV ROC-AUC", *(f"CV {TITLES[m]}" for m in METRICS[1:])]
    head += [f"Test {TITLES[m]}" for m in METRICS]
    rows = []
    for family, r in results.items():
        cv, test = r["cv"], r["test"]
        name = f"**{family}** (champion)" if family == champion else family
        rows.append(
            [
                name,
                f"{cv['roc_auc_mean']:.3f} ± {cv['roc_auc_std']:.3f}",
                *(f"{cv[f'{m}_mean']:.3f}" for m in METRICS[1:]),
                *(f"{test[m]:.3f}" for m in METRICS),
            ]
        )
    loso_rows = [
        [
            SITE_NAMES.get(site, site),
            str(s["n"]),
            f"{s['prevalence']:.0%}",
            *(f"{s[m]:.3f}" for m in ("roc_auc", "accuracy", "recall")),
        ]
        for site, s in loso.items()
    ]
    intro = (
        f"{n_train} training and {n_test} hold-out patients (80/20 split stratified on site x "
        "class). CV = 5 stratified folds of the training split; test = the hold-out, never used "
        f"for selection. The champion has the best mean CV ROC-AUC. Threshold {THRESHOLD}."
    )
    loso_head = ["Held-out site", "Patients", "Disease", "ROC-AUC", "Accuracy", "Recall"]
    sections = [
        "# Model comparison",
        intro,
        _table(head, rows),
        f"## Leave-one-site-out ({champion})",
        "Champion hyper-parameters, trained on the other three sites, scored on the fourth.",
        _table(loso_head, loso_rows),
    ]
    return "\n\n".join(sections) + "\n"


def _run_family(family, estimator, grid, note, split, folds, session, dataset) -> dict:
    """Grid-search one family and log it as an MLflow run (one nested run per grid point)."""
    X_train, y_train, X_test, y_test = split
    search = GridSearchCV(
        build_pipeline(estimator),
        {f"model__{name}": values for name, values in grid.items()},
        scoring=SCORING,
        refit="roc_auc",
        cv=folds,
        n_jobs=-1,
        error_score="raise",
    )
    tags = {"training_session": session, "family": family}
    with mlflow.start_run(run_name=family, description=note, tags=tags) as run:
        started = time.perf_counter()
        search.fit(X_train, y_train)
        seconds = time.perf_counter() - started
        cv = pd.DataFrame(search.cv_results_)
        for i, row in cv.iterrows():
            with mlflow.start_run(run_name=f"{family}-{i + 1:02d}", nested=True, tags=tags):
                mlflow.log_params(_unprefixed(row["params"]))
                scores = {f"cv_{k}": v for k, v in _cv_scores(row).items()}
                mlflow.log_metrics(scores | {"fit_seconds": row["mean_fit_time"]})
                per_fold = {
                    m: [row[f"split{k}_test_{m}"] for k in range(len(folds))] for m in METRICS
                }
                mlflow.log_dict(per_fold, "cv_folds.json")

        model = search.best_estimator_
        cv_scores = _cv_scores(cv.loc[search.best_index_])
        scorer = check_scoring(model, SCORING)
        test_scores = {k: float(v) for k, v in scorer(model, X_test, y_test).items()}
        proba, pred = model.predict_proba(X_test)[:, 1], model.predict(X_test)
        importance = permutation_importance(
            model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1
        )
        figures = {
            "confusion_matrix": _confusion_figure(y_test, pred, family),
            "roc": _roc_figure(y_test, proba, family),
            "precision_recall": _pr_figure(y_test, proba, family),
            "permutation_importance": _importance_figure(importance, family),
        }

        mlflow.log_params(model[-1].get_params())  # tuned values plus the fixed settings
        mlflow.log_metrics(
            {f"cv_{k}": v for k, v in cv_scores.items()}
            | {f"test_{k}": v for k, v in test_scores.items()}
            | {"search_seconds": seconds}
        )
        with warnings.catch_warnings():
            # MLflow's integer-column hint is about model inputs; this schema's int is the label.
            warnings.filterwarnings("ignore", "Hint: Inferred schema contains integer")
            mlflow.log_input(dataset, context="training")
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"plots/{name}.png", save_kwargs=SAVE)
        mlflow.log_text(cv.to_csv(index=False), "cv_results.csv")
        report = classification_report(y_test, pred, target_names=LABELS, digits=3, zero_division=0)
        mlflow.log_text(report, "classification_report.txt")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
            pip_requirements=str(config.ROOT / "requirements-serve.txt"),
            code_paths=[str(Path(__file__).parent)],  # loads without cardiorisk installed
            # skops refuses unknown types on load. This object was built in-process, so the types
            # it contains (our feature functions, tree internals) are exactly the ones to allow.
            skops_trusted_types=skops.io.get_untrusted_types(data=skops.io.dumps(model)),
        )
    return {
        "run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "model": model,
        "baseline": isinstance(estimator, DummyClassifier),
        "params": _unprefixed(search.best_params_),
        "cv": cv_scores,
        "test": test_scores,
        "search_seconds": round(seconds, 1),
        "figures": figures,
    }


def _leave_one_site_out(model, X, y, sites) -> dict:
    """Refit on three hospitals, score the fourth: does the model transfer to an unseen site?"""
    scores = cross_validate(
        model,
        X,
        y,
        groups=sites,
        cv=LeaveOneGroupOut(),
        scoring=SCORING,
        n_jobs=-1,
        return_indices=True,
    )
    loso = {}
    for fold, test in enumerate(scores["indices"]["test"]):
        site = sites.iloc[test[0]]
        loso[site] = {"n": len(test), "prevalence": float(y.iloc[test].mean())}
        loso[site] |= {m: float(scores[f"test_{m}"][fold]) for m in METRICS}
    return loso


def train(
    df: pd.DataFrame,
    out_dir: Path = config.ROOT,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    candidates: dict = CANDIDATES,
) -> dict:
    """Tune, track, select and save; returns the model card.

    Writes models/ and reports/ (and MLflow artifacts in mlruns/ for a new experiment) under
    out_dir, so tests can run a reduced candidate set into tmp_path with a throwaway sqlite URI.
    """
    out = Path(out_dir)
    trained_at = datetime.now(UTC)
    session = trained_at.strftime("%Y%m%d-%H%M%S")
    X, y, sites = df[INPUT_FEATURES].astype(float), df[config.TARGET], df["site"]
    strata = sites + "/" + y.astype(str)  # keeps both the site mix and the prevalence
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=strata, random_state=SEED
    )
    split = (X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(split[0], strata.loc[train_idx]))  # site x class in every fold, too

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        # MLflow's default artifact root is ./mlruns under whatever the working directory is.
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT, artifact_location=(out / "mlruns").as_uri()
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    # Our own digest: comparable with the card's data_sha256, and it skips MLflow's digest code,
    # which trips a pandas 4 deprecation.
    dataset = mlflow.data.from_pandas(
        df.loc[train_idx],
        targets=config.TARGET,
        name="heart_disease_clean (training split)",
        digest=_sha256(df.loc[train_idx])[:16],
    )
    results = {
        family: _run_family(family, *spec, split, folds, session, dataset)
        for family, spec in candidates.items()
    }

    # Selection sees CV scores only; the hold-out stays an unbiased estimate of the champion.
    champion = max(
        (f for f, r in results.items() if not r["baseline"]),
        key=lambda f: results[f]["cv"]["roc_auc_mean"],
    )
    best = results[champion]
    version = mlflow.register_model(best["model_uri"], config.REGISTERED_MODEL).version
    client = MlflowClient()
    client.set_registered_model_alias(config.REGISTERED_MODEL, "champion", version)
    client.update_model_version(
        config.REGISTERED_MODEL,
        version,
        description=f"{champion} from training session {session}: CV ROC-AUC "
        f"{best['cv']['roc_auc_mean']:.3f}, hold-out ROC-AUC {best['test']['roc_auc']:.3f}.",
    )
    loso = _leave_one_site_out(best["model"], X, y, sites)
    loso_figure = _loso_figure(loso, best["test"])
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("champion", "true")
        mlflow.log_metrics(
            {
                f"loso_{site}_{m}": s[m]
                for site, s in loso.items()
                for m in ("roc_auc", "accuracy", "recall")
            }
        )
        mlflow.log_figure(loso_figure, "plots/loso_by_site.png", save_kwargs=SAVE)

    def at(path: Path) -> Path:
        return out / path.relative_to(config.ROOT)

    for directory in (at(config.MODEL_PATH).parent, at(config.FIGURES_DIR)):
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], at(config.MODEL_PATH))  # exactly the model scored above
    card = {
        "version": str(version),
        "family": champion,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
        "metrics": {"cv": best["cv"], "test": best["test"], "loso": loso},
        "features": INPUT_FEATURES,
        "params": best["params"],
        "mlflow_run_id": best["run_id"],
        "mlflow_model_uri": f"models:/{config.REGISTERED_MODEL}/{version}",
        "training_session": session,
        "sklearn_version": sklearn.__version__,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "class_balance": {
            part: {LABELS[c]: int(n) for c, n in y.loc[idx].value_counts().sort_index().items()}
            for part, idx in (("train", train_idx), ("test", test_idx))
        },
        "data_sha256": _sha256(df),
    }
    at(config.MODEL_CARD_PATH).write_text(json.dumps(card, indent=2) + "\n")

    summary = {
        "champion": champion,
        "training_session": session,
        "families": {
            f: {k: r[k] for k in ("cv", "test", "params", "run_id", "search_seconds")}
            for f, r in results.items()
        },
        "loso": loso,
    }
    reports = at(config.REPORTS_DIR)
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = _markdown(results, champion, loso, len(train_idx), len(test_idx))
    (reports / "metrics.md").write_text(markdown)

    figures = at(config.FIGURES_DIR)
    _comparison_figure(results, champion).savefig(figures / "model_comparison.png", **SAVE)
    loso_figure.savefig(figures / "model_loso_by_site.png", **SAVE)
    for name in ("roc", "confusion_matrix", "permutation_importance"):
        best["figures"][name].savefig(figures / f"model_champion_{name}.png", **SAVE)
    return card


def main() -> None:
    started = time.perf_counter()
    card = train(load_processed())
    print((config.REPORTS_DIR / "metrics.md").read_text())
    print(
        f"Champion {card['family']} -> {config.MODEL_PATH} and {card['mlflow_model_uri']} "
        f"(alias 'champion'); tracking {config.MLFLOW_TRACKING_URI}; "
        f"{time.perf_counter() - started:.0f} s"
    )


if __name__ == "__main__":
    main()
