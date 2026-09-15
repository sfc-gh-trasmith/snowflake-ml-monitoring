# Snowflake ML Monitoring Examples

End-to-end examples for monitoring ML models deployed on Snowflake. Covers three approaches -- built-in Model Monitor, custom metrics (Task + Alert + ML Job), and Gateway A/B testing -- all using the same synthetic churn prediction dataset.

## Repository Structure

```
snowflake-ml-monitoring/
├── notebooks/
│   ├── 00_setup.ipynb                 # Shared setup: data, models, infrastructure
│   ├── 01_model_monitor.ipynb         # Built-in Model Monitor (drift, performance, segments)
│   ├── 02_model_monitor_custom.ipynb  # Custom metrics (Task + StoredProc + Alert, ML Job)
│   └── 03_gateway_monitor.ipynb       # Gateway A/B testing with traffic splits
├── utils/
│   ├── custom_metrics.py              # Reusable metric functions (chi-squared, KS, etc.)
│   ├── custom_metrics_job.py          # ML Job script for custom metrics
│   └── gatway_monitor_simulate.py     # Long-running gateway inference simulator
├── scripts/
│   └── setup.sh                       # Infrastructure setup via Snowflake CLI
├── pyproject.toml
└── README.md
```

## Quickstart

### Prerequisites

- Python 3.9+
- A Snowflake account with ACCOUNTADMIN access
- A compute pool (only needed for notebooks 02 ML Job and 03 gateway)

### Option A: Local Development

```bash
# Clone and set up
git clone <repo-url>
cd snowflake-ml-monitoring
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the setup notebook first, then any demo notebook
# Open in your IDE (VS Code, Cortex Code, JupyterLab, etc.)
```

Update the `LOCAL CONFIG` section in each notebook's connection cell with your account, user, and key path.

### Option B: Snowsight Workspaces

Upload the `notebooks/` and `utils/` folders to a Snowflake Workspace. The notebooks auto-detect Snowsight and use `get_active_session()` -- no connection configuration needed. Run `00_setup.ipynb` first, then open any demo notebook.

### Running the Notebooks

**Always run `00_setup.ipynb` first.** It creates the database, schema, warehouse, synthetic data, and registers the models that all other notebooks depend on.

| Notebook | What it demonstrates | Time to run | Requires SPCS |
|----------|---------------------|-------------|---------------|
| `00_setup` | Data generation, model training, registry | ~3 min | No |
| `01_model_monitor` | Built-in drift/performance monitoring | ~5 min | No |
| `02_model_monitor_custom` | Custom metrics with Tasks, Alerts, ML Jobs | ~5 min | ML Job section only |
| `03_gateway_monitor` | A/B testing with traffic-split gateway | ~20 min | Yes |

## Notebook Summaries

### 00_setup -- Shared Setup

Creates all shared resources: `ML_DEMO.ML_CHURN` database/schema, synthetic churn dataset (5,000 rows), train/test splits stored as Snowflake tables, and two registered models (LogisticRegression V1, XGBoost V2).

### 01_model_monitor -- Built-In Model Monitoring

Walks through Snowflake's managed Model Monitor service step by step:
1. Create a prediction source table from batch inference
2. Create a monitor with statistical metrics (COUNT, MEAN, STDDEV)
3. Set a baseline to unlock drift metrics (PSI, Jensen-Shannon, Wasserstein)
4. Add ground truth to unlock performance metrics (Accuracy, F1, Precision, Recall)
5. Segmented monitoring by contract type
6. Monitor management (suspend, resume, alter)

### 02_model_monitor_custom -- Custom Metrics

For metrics outside the built-in service (chi-squared, KS test, prediction confidence shift), demonstrates two production-grade approaches:
- **Approach A:** Python stored procedure scheduled by a Snowflake Task, with an Alert for threshold violations
- **Approach B:** ML Job on SPCS compute pool, scheduled by a Task

Both write to a shared `CUSTOM_METRICS` table with timestamped results.

### 03_gateway_monitor -- Gateway A/B Testing

Full A/B testing workflow using Snowflake Gateway:
1. Deploy two model versions as inference services with auto-capture
2. Create a 50/50 traffic-split gateway
3. Send live inference traffic via REST API (JWT auth)
4. Create a gateway model monitor tracking drift and performance across both services
5. Query comparative metrics to decide the winner
6. Shift traffic to the winning model

## Snowflake Objects Created

All objects are created in `ML_DEMO.ML_CHURN`:

| Object | Type | Created by |
|--------|------|-----------|
| `ML_DEMO` | Database | 00_setup |
| `ML_CHURN` | Schema | 00_setup |
| `ML_CHURN_WH` | Warehouse | 00_setup |
| `CHURN_TRAIN` / `CHURN_TEST` | Tables | 00_setup |
| `CHURN_MODEL` (V1, V2) | Model | 00_setup |
| `GROUND_TRUTH` | Table | 00_setup / 03_gateway |
| `CHURN_PREDICTIONS` / `CHURN_BASELINE` | Tables | 01_model_monitor |
| `CHURN_MONITOR` | Model Monitor | 01_model_monitor |
| `CUSTOM_METRICS` / `CUSTOM_ALERT_HISTORY` | Tables | 02_custom |
| `COMPUTE_CUSTOM_METRICS` | Stored Procedure | 02_custom |
| `CUSTOM_METRICS_TASK` | Task | 02_custom |
| `CUSTOM_DRIFT_ALERT` | Alert | 02_custom |
| `CHURN_V1_SVC` / `CHURN_V2_SVC` | Services | 03_gateway |
| `CHURN_GATEWAY` | Gateway | 03_gateway |
| `CHURN_AB_MONITOR` | Model Monitor | 03_gateway |

Each notebook includes a cleanup section that drops only the objects it created, leaving shared resources intact.

## Cleanup

To remove everything:

```sql
DROP DATABASE IF EXISTS ML_DEMO;
DROP WAREHOUSE IF EXISTS ML_CHURN_WH;
```