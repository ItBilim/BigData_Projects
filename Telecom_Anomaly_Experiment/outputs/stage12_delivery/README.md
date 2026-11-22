# Reproduce the Telecom Anomaly Experiment

The code was run on Windows with Python 3.12. Work from this project directory. Full recomputation requires authorized access to the original archive at D:\BigData\BigData_Projects\cdr_datasets.tar.gz, the location expected by the registered stage scripts.

```powershell
python -m pip install -r outputs/stage12_delivery/requirements.txt
python -B outputs/stage12_delivery/run.py --rerun --from-stage 2
```

Use a separate working copy for recomputation. The runner rebuilds intermediate files and writes a run log. With no flags it packages existing outputs and requires the intermediate scores and archive. The public snapshot contains the code and aggregate outputs for inspection. The original private project contains the intermediate inputs and subscriber-level files.

## Layout

| Location | Contents |
| --- | --- |
| outputs/stage1_audit | Data audit and data dictionary |
| outputs/stage2_protocol | Evaluation protocol and feature eligibility |
| outputs/stage3_splits | Chronological validation and test folds |
| outputs/stage4_features | Historical feature engineering and checks |
| outputs/stage5_methods | Methods, validation, and selected parameters |
| outputs/stage6_scenarios | Synthetic injection code and scenario summaries |
| outputs/stage7_leakage | Available-history and hindsight comparison |
| outputs/stage8_metrics | Weekly, pooled, and segment results |
| outputs/stage9_stability | Subscriber bootstrap and seed sensitivity |
| outputs/stage10_11_audit | Error analysis, timings, and regression checks |
| outputs/stage12_delivery | Runner, pinned dependencies, report, and figures |
