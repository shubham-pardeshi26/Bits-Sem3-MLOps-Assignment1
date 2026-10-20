# Screenshots

Evidence for [reports/REPORT.md](../reports/REPORT.md), which embeds these files by name
(`../screenshots/<file>`). Keep the file names exactly as listed. Commands are Makefile targets
from the [README](../README.md); run them from the repository root. The local host port is 8081
(the Makefile's `PORT`) and the MLflow UI is on 5001; if 8081 is taken on your machine, add the
same `PORT=<port>` to `docker-run` and `smoke`, and use that port in the URLs. Every kubectl
command below names `--context minikube`; keep it visible in the terminal captures.

The "In report" column says where REPORT.md shows each file: next to the section it proves
("body", the 5 that prove CI/CD and deployment) or in the two-column "Appendix C. Additional
screenshots" ("appendix", 12 files). After adding or replacing screenshots, rebuild both report
files: `make report` (editable `reports/report.docx`) and `make report-pdf` (final
`reports/report.pdf`, needs Google Chrome).

| File | In report | What it must show | How to get there |
| --- | --- | --- | --- |
| `01_mlflow_experiment_runs.png` | appendix | MLflow experiment `cardiorisk-heart-disease`: the four family runs (`baseline_majority`, `logistic_regression`, `random_forest`, `hist_gradient_boosting`) with key metric columns | `make mlflow-ui`, open http://127.0.0.1:5001, Experiments > `cardiorisk-heart-disease`. Filter on one session, e.g. `tags.training_session = '<latest>'`, keep nested runs collapsed, add the columns `cv_roc_auc_mean`, `test_roc_auc`, `test_accuracy`, `test_recall`, `test_f1` |
| `02_mlflow_champion_run-main-page.png` | appendix | The champion run's Overview: description, metrics (`cv_*`, `test_*`) and parameters | Same experiment, open the latest `logistic_regression` run (tag `champion: true`), Overview tab; zoom out so metrics and parameters both fit |
| `02_mlflow_champion_run-artifact-plots.png` | appendix | The champion run's Artifacts: `plots/` (e.g. `loso_by_site.png` previewed), `cv_results.csv`, `classification_report.txt` | Same run, Artifacts tab, click `plots/loso_by_site.png` |
| `03_mlflow_compare_runs.png` | appendix | Compare view (parallel coordinates) of the nested grid runs of one family | Expand the `logistic_regression` run of one session, select its 36 nested runs (`logistic_regression-01` ... `-36`), Compare > Parallel coordinates with params `C`, `l1_ratio`, `class_weight` and metric `cv_roc_auc_mean` |
| `04_mlflow_model_registry.png` | appendix | Registered model `cardiorisk-classifier` with the `@champion` alias | MLflow UI > Models > `cardiorisk-classifier`: the version list with the `champion` alias on the latest version (the one in `models/model_card.json`) |
| `05_github_actions_pipeline.png` | body | A workflow run graph with all six jobs green | GitHub > Actions > CI/CD > a run triggered by a push to `main` (`publish` is skipped on pull requests). The graph: Lint (ruff), Unit tests (pytest), Train and track (MLflow), Docker image + smoke test, Deploy to kind + monitoring check, Publish image |
| `06_github_actions_summary.png` | body | The run summary: test summary and model comparison table | Same run, Summary page: "Unit tests passed", the model comparison and leave-one-site-out tables; include the Prometheus targets and "Published image" sections if they fit |
| `07_github_actions_artifacts.png` | appendix | The run's artifacts list | Bottom of the same Summary page: `test-results`, `model`, `training-report`, `mlflow-tracking`, `image`. Capture on the day of the run: `image` is kept for 1 day only, then shows as expired |
| `08_ghcr_package.png` | appendix | The GHCR package page with the `sha-<7>` and `latest` tags | Repository page > Packages (right sidebar) > the package named after the repository (`ghcr.io/OWNER/REPO`, lower case) |
| `09_docker_build.png` | appendix | Terminal: `make docker-build` (cached layers) and `make smoke` ending in `SMOKE TEST PASSED` | Terminal 2: `docker info`, `make docker-build`, then `make smoke` while the container runs |
| `09_docker_build_run.png` | appendix | Terminal: `make docker-run`: uvicorn started and the container's JSON `model_loaded`, `request` and `prediction` lines | Terminal 1: `make docker-run`, captured after the smoke test ran |
| `10_fastapi_docs.png` | appendix | Swagger UI `/docs` with a `/predict` request and response | With `make docker-run` still running, open http://127.0.0.1:8081/docs, POST `/predict` > Try it out (prefilled with the sample patient) > Execute. Show the request body and the 200 response |
| `11_k8s_resources.png` | body | Terminal: pods, services, ingress and HPA in namespace `cardiorisk` | After `make k8s-deploy` and some traffic: `kubectl --context minikube -n cardiorisk get pods,svc,ingress,hpa`. Two `cardiorisk-api` pods Running, the LoadBalancer Service on port 8081 with an external IP (shown only while `minikube tunnel` runs), the Ingress, the HPA with a CPU target (`<unknown>` for the first minute, until metrics-server has samples) |
| `12_k8s_smoke_test.png` | body | Terminal: `minikube tunnel` running and `make smoke` passing against the LoadBalancer and the Ingress | Terminal 1: `minikube tunnel`. Terminal 2: `make smoke` (LoadBalancer, http://127.0.0.1:8081) and `make smoke URL=http://127.0.0.1` (Ingress), both ending in `SMOKE TEST PASSED` |
| `13_prometheus_targets.png` | appendix | Prometheus `/targets` with the `cardiorisk-api` pods UP | `make k8s-forward`, open http://127.0.0.1:9090/targets: job `cardiorisk-api` with one target per pod (instance = pod name), all UP |
| `14_grafana_dashboard.png` | body | Grafana "CardioRisk API" dashboard with traffic from `make load` | `make load URL=http://127.0.0.1`, then (with `make k8s-forward` running) open http://127.0.0.1:3000; the dashboard is the home page. Time range "Last 15 minutes" so the load is visible in every panel |
| `15_api_json_logs.png` | appendix | `kubectl logs` output with JSON `request` and `prediction` lines | `kubectl --context minikube -n cardiorisk logs deployment/cardiorisk-api --tail=20` right after `make load`. `/health` and `/metrics` are logged at DEBUG, so the lines are `/predict` traffic, not probes |

## Capture order (one pass through the pipeline)

1. **MLflow (01-04):** `make train`, then `make mlflow-ui`. Capture 01, 02, 03, 04. Stop the UI.
2. **Docker (09-10):** `make docker-build`, `make docker-run`, `make smoke` in a second terminal:
   capture 09. Open `/docs` on the running container: capture 10. Stop it with Ctrl-C.
3. **minikube (11-15):** `make minikube-start`, `make k8s-image`, `make k8s-deploy`. Start
   `minikube tunnel` in its own terminal, run both smoke tests: capture 12. Run
   `make load URL=http://127.0.0.1`, then capture 11 and 15. Start `make k8s-forward`: capture 13
   and 14 (run `make load URL=http://127.0.0.1` again if the panels are flat). Finish with
   `make k8s-delete` and `minikube stop`.
4. **GitHub (05-08):** push to `main`, wait for the six jobs, then capture 05, 06, 07 and 08.

## Capture tips

- Crop to the relevant window or panel; no desktop, dock or unrelated tabs.
- Readable text: terminal font 14 pt or larger, browser zoom 100-125%. A reader of the .docx
  report should read command output without zooming.
- Show the command together with its result (prompt line included), so each image stands alone.
- Keep secrets and unrelated infrastructure out of frame: no tokens, no `kubectl config` output
  listing other clusters.
- On macOS, Cmd-Shift-4 then Space captures a single window; save as PNG under the exact name.
