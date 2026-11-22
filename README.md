# Telecom Revenue Anomaly Detection

This project compares ten methods for detecting anomalies in weekly subscriber revenue. It uses an expanding training window and evaluates the same subscriber-week records across methods at weekly audit budgets of 0.5%, 1%, 2%, and 5%.

| Test result, weeks 32–35 | Precision at top 1% |
| --- | ---: |
| LightGBM absolute residual | 40.60% |
| Global robust score | 39.48% |

The paired difference was 1.12 percentage points (95% subscriber-cluster bootstrap interval: 0.84 to 1.44). The primary test uses injected anomalies from the supplied dataset. Additional synthetic evaluations cover selected configurations. The full synthetic grid and training contamination evaluation remain pending.

## Project materials

- [Project directory and reproduction guide](Telecom_Anomaly_Experiment/)
- [Main results table](Telecom_Anomaly_Experiment/outputs/stage12_delivery/article_main_table.csv)
- [Article figures](Telecom_Anomaly_Experiment/outputs/stage12_delivery/figures/)
- [Evaluation code and configuration](Telecom_Anomaly_Experiment/outputs/stage12_delivery/)
- [Weekly metrics](Telecom_Anomaly_Experiment/outputs/stage8_metrics/weekly_metrics.csv)
- [Confidence intervals](Telecom_Anomaly_Experiment/outputs/stage9_stability/confidence_intervals.csv)
- [Analytical report in Kazakh](Telecom_Anomaly_Experiment/outputs/stage12_delivery/analysis_kk.md)

This public repository contains the code, protocols, aggregate tables, and six article figures. The original archive, subscriber-level scores and histories, and injection logs are stored in the local project.
