#!/bin/bash
# =============================================================================
# Gateway Monitoring & A/B Testing Quickstart - Infrastructure Setup
# =============================================================================
# Prerequisites:
#   - Snowflake CLI (snow) installed and configured with a connection
#   - ACCOUNTADMIN role access
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e

SNOW_CMD="snow sql -q"

echo ">>> Setting role to ACCOUNTADMIN..."
$SNOW_CMD "USE ROLE ACCOUNTADMIN;"

echo ">>> Creating database ML_DEMO..."
$SNOW_CMD "CREATE DATABASE IF NOT EXISTS ML_DEMO;"

echo ">>> Creating schema ML_DEMO.ML_CHURN..."
$SNOW_CMD "CREATE SCHEMA IF NOT EXISTS ML_DEMO.ML_CHURN;"

echo ">>> Creating warehouse ML_CHURN_WH..."
$SNOW_CMD "
CREATE WAREHOUSE IF NOT EXISTS ML_CHURN_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;
"

echo ">>> Creating ground truth table..."
$SNOW_CMD "
CREATE OR REPLACE TABLE ML_DEMO.ML_CHURN.GROUND_TRUTH (
    \"request_id\" VARCHAR,
    \"churned\" NUMBER
);
"

echo ">>> Granting required privileges..."
$SNOW_CMD "GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE ACCOUNTADMIN;"
$SNOW_CMD "GRANT CREATE GATEWAY ON SCHEMA ML_DEMO.ML_CHURN TO ROLE ACCOUNTADMIN;"
$SNOW_CMD "GRANT CREATE MODEL MONITOR ON SCHEMA ML_DEMO.ML_CHURN TO ROLE ACCOUNTADMIN;"
$SNOW_CMD "GRANT EXECUTE TASK ON ACCOUNT TO ROLE ACCOUNTADMIN;"
$SNOW_CMD "GRANT EXECUTE ALERT ON ACCOUNT TO ROLE ACCOUNTADMIN;"
$SNOW_CMD "GRANT EXECUTE MANAGED ALERT ON ACCOUNT TO ROLE ACCOUNTADMIN;"

echo ""
echo "=== Setup complete ==="
echo "Database:  ML_DEMO"
echo "Schema:    ML_DEMO.ML_CHURN"
echo "Warehouse: ML_CHURN_WH"
echo "Table:     ML_DEMO.ML_CHURN.GROUND_TRUTH"
echo ""
echo "Next: Open notebooks/00_setup.ipynb to generate data and train models."
