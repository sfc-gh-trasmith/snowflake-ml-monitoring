#!/usr/bin/env python3
"""
simulate.py — Full-day A/B gateway inference simulator

Sends inference traffic through CHURN_GATEWAY for 24 hours (configurable),
writes ground truth to Snowflake periodically, and checkpoints progress so
the run can be resumed after any interruption.

Auth: keypair JWT generated from rsa_key.p8 — refreshed every 55 minutes.
No PAT needed.

Usage:
    nohup .venv/bin/python simulate.py &

Resume after interruption (auto-detected via checkpoint file):
    nohup .venv/bin/python simulate.py &

Override duration:
    DURATION_SECONDS=3600 .venv/bin/python simulate.py
"""

import base64
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jwt
import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import serialization
from scipy.special import expit
from sklearn.model_selection import train_test_split
from snowflake.snowpark import Session

# ── Config ───────────────────────────────────────────────────────────────────
ACCOUNT   = "<your-account-identifier>"
USER      = "<your-username>"
ROLE      = "ACCOUNTADMIN"
DATABASE  = "ML_DEMO"
SCHEMA    = "ML_CHURN"
WAREHOUSE = "ML_CHURN_WH"
KEY_PATH  = Path.home() / ".snowflake" / "keys" / "rsa_key.p8"

GATEWAY_FQN        = "ML_DEMO.ML_CHURN.CHURN_GATEWAY"
SERVICE_V1         = "ML_DEMO.ML_CHURN.CHURN_V1_SVC"
SERVICE_V2         = "ML_DEMO.ML_CHURN.CHURN_V2_SVC"
GROUND_TRUTH_TABLE = "GROUND_TRUTH"

DURATION_SECONDS     = int(os.environ.get("DURATION_SECONDS", 86400))  # default 24h
BATCH_SIZE           = 5
SLEEP_BETWEEN        = 0.5    # seconds between batches
GT_WRITE_EVERY       = 100    # write ground truth every N successful batches
CHECKPOINT_EVERY     = 300    # checkpoint every N seconds
LOG_EVERY            = 60     # progress log every N successful batches
JWT_LIFETIME         = 3600   # JWT token lifetime in seconds (1 hour)
JWT_REFRESH_BUFFER   = 300    # refresh JWT this many seconds before expiry
SERVICE_WAIT_TIMEOUT = 900    # seconds to wait for services to be READY at startup

REPO_DIR        = Path(__file__).parent
CHECKPOINT_FILE = REPO_DIR / "simulate_checkpoint.json"
LOG_FILE        = REPO_DIR / "simulate.log"
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "TENURE_MONTHS", "MONTHLY_CHARGES", "TOTAL_CHARGES",
    "CONTRACT_TYPE", "NUM_SUPPORT_TICKETS", "INTERNET_SERVICE",
]


def setup_logging():
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    handlers = [logging.FileHandler(LOG_FILE, mode="a")]
    if sys.stdout.isatty() or os.environ.get("SIMULATE_STDOUT"):
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_private_key():
    with open(KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _private_key_bytes(pk) -> bytes:
    return pk.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def generate_jwt(private_key) -> tuple[str, float]:
    """Return (jwt_token, expiry_timestamp). Token valid for JWT_LIFETIME seconds."""
    pub_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(pub_bytes).digest()).decode()
    qualified = f"{ACCOUNT.upper()}.{USER.upper()}"
    now = int(time.time())
    exp = now + JWT_LIFETIME
    payload = {
        "iss": f"{qualified}.{fingerprint}",
        "sub": qualified,
        "iat": now,
        "exp": exp,
    }
    token = jwt.encode(payload, private_key, algorithm="RS256")
    return token, float(exp)


def get_auth_headers(token: str) -> dict:
    return {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
    }


def create_session(pk) -> Session:
    return Session.builder.configs({
        "account":     ACCOUNT,
        "user":        USER,
        "private_key": _private_key_bytes(pk),
        "role":        ROLE,
        "database":    DATABASE,
        "schema":      SCHEMA,
        "warehouse":   WAREHOUSE,
    }).create()


def get_gateway_url(session: Session) -> str:
    rows = session.sql(f"DESC GATEWAY {GATEWAY_FQN}").collect()
    for row in rows:
        d = row.as_dict()
        url = d.get("ingress_url", "")
        if url and "snowflakecomputing" in url:
            return f"https://{url}/predict"
    raise RuntimeError(f"No ingress_url found for {GATEWAY_FQN}.")


def wait_for_services(session: Session, timeout: int = SERVICE_WAIT_TIMEOUT):
    """Block until both inference services report RUNNING status."""
    services = [SERVICE_V1, SERVICE_V2]
    start = time.time()
    while time.time() - start < timeout:
        all_ready = True
        for svc in services:
            raw = session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{svc}')").collect()[0][0]
            statuses = json.loads(raw)
            # Check all containers in the service
            pending = [s for s in statuses if s.get("status") != "READY"]
            if pending:
                all_ready = False
                msgs = {s["containerName"]: s["status"] for s in statuses}
                elapsed = int(time.time() - start)
                logging.info("  %s not ready (%ds): %s", svc.split(".")[-1], elapsed, msgs)
        if all_ready:
            logging.info("Both services are READY.")
            return
        time.sleep(30)
    raise TimeoutError(f"Services not READY within {timeout}s. Check SYSTEM$GET_SERVICE_STATUS.")


def generate_test_data():
    """Reproduce the same synthetic dataset as the notebook (same seed)."""
    np.random.seed(42)
    n = 5000
    data = pd.DataFrame({
        "TENURE_MONTHS":       np.random.randint(1, 72, n),
        "MONTHLY_CHARGES":     np.round(np.random.uniform(20, 120, n), 2),
        "TOTAL_CHARGES":       np.round(np.random.uniform(100, 8000, n), 2),
        "CONTRACT_TYPE":       np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2]),
        "NUM_SUPPORT_TICKETS": np.random.poisson(2, n),
        "INTERNET_SERVICE":    np.random.choice([0, 1, 2], n, p=[0.2, 0.4, 0.4]),
    })
    churn_prob = (
        -0.02 * data["TENURE_MONTHS"]
        + 0.01 * data["MONTHLY_CHARGES"]
        + 0.15 * data["NUM_SUPPORT_TICKETS"]
        - 0.50 * data["CONTRACT_TYPE"]
        + np.random.normal(0, 0.5, n)
    )
    data["CHURNED"] = (expit(churn_prob) > 0.5).astype(int)
    _, X_test, _, y_test = train_test_split(
        data[FEATURE_COLS], data["CHURNED"],
        test_size=0.2, random_state=42, stratify=data["CHURNED"],
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def load_checkpoint() -> dict | None:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None


def save_checkpoint(data: dict):
    data["last_saved"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def write_ground_truth(session: Session, request_ids: list, labels: list, overwrite: bool = False):
    df = pd.DataFrame({"request_id": request_ids, "churned": labels})
    if overwrite:
        session.sql(f'TRUNCATE TABLE IF EXISTS {DATABASE}.{SCHEMA}.{GROUND_TRUTH_TABLE}').collect()
    session.write_pandas(
        df,
        table_name=GROUND_TRUTH_TABLE,
        database=DATABASE,
        schema=SCHEMA,
        overwrite=False,
        quote_identifiers=True,
    )
    logging.info("Wrote %d ground truth rows (overwrite=%s)", len(df), overwrite)


def main():
    setup_logging()

    # ── Resume or fresh start ────────────────────────────────────────────────
    cp = load_checkpoint()
    if cp and not cp.get("completed"):
        prior_elapsed = cp.get("elapsed_seconds", 0)
        batch_count   = cp.get("batch_count", 0)
        error_count   = cp.get("error_count", 0)
        first_write   = False
        remaining     = DURATION_SECONDS - prior_elapsed
        logging.info(
            "Resuming: %.0fs elapsed, %d batches, %.0fs remaining",
            prior_elapsed, batch_count, remaining,
        )
        if remaining <= 0:
            logging.info("Already complete. Delete %s to restart.", CHECKPOINT_FILE)
            return
    else:
        if cp and cp.get("completed"):
            logging.info("Prior run completed. Delete %s to restart.", CHECKPOINT_FILE)
            return
        prior_elapsed = 0
        batch_count   = 0
        error_count   = 0
        first_write   = True
        remaining     = DURATION_SECONDS
        logging.info(
            "Fresh run: %ds (%.1fh), batch_size=%d, sleep=%.1fs",
            DURATION_SECONDS, DURATION_SECONDS / 3600, BATCH_SIZE, SLEEP_BETWEEN,
        )

    private_key = load_private_key()

    logging.info("Connecting to Snowflake...")
    session = create_session(private_key)
    logging.info("Connected: %s / %s.%s", session.get_current_role(),
                 session.get_current_database(), session.get_current_schema())

    # Resume services if suspended (auto_resume handles it on traffic, but we check explicitly)
    for svc in [SERVICE_V1, SERVICE_V2]:
        try:
            session.sql(f"ALTER SERVICE {svc} RESUME").collect()
            logging.info("Resumed %s", svc.split('.')[-1])
        except Exception:
            pass  # already running

    logging.info("Waiting for services to become READY...")
    wait_for_services(session)

    gateway_url = get_gateway_url(session)
    logging.info("Gateway URL: %s", gateway_url)

    X_test, y_test = generate_test_data()
    logging.info("Test data ready: %d rows", len(X_test))

    # ── Inference loop ───────────────────────────────────────────────────────
    pending_ids:    list[str] = []
    pending_labels: list[int] = []

    run_start          = time.time()
    last_checkpoint_at = time.time()
    start_ts           = cp.get("start_timestamp") if cp else datetime.now(timezone.utc).isoformat()

    # Generate initial JWT
    jwt_token, jwt_exp = generate_jwt(private_key)
    headers = get_auth_headers(jwt_token)
    logging.info("Inference loop started. Target remaining: %.0fs", remaining)

    while time.time() - run_start < remaining:

        # Refresh JWT before it expires
        if time.time() >= jwt_exp - JWT_REFRESH_BUFFER:
            jwt_token, jwt_exp = generate_jwt(private_key)
            headers = get_auth_headers(jwt_token)
            logging.info("JWT refreshed (next expiry in %ds).", JWT_LIFETIME)

        idx          = np.random.choice(len(X_test), size=BATCH_SIZE, replace=False)
        sample       = X_test.iloc[idx].copy()
        true_labels  = y_test.iloc[idx].tolist()
        request_ids  = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]
        sample["request_id"] = request_ids

        payload = {
            "dataframe_records": json.loads(sample.to_json(orient="records")),
            "extra_columns": ["request_id"],
        }

        try:
            resp = requests.post(gateway_url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                pending_ids.extend(request_ids)
                pending_labels.extend(true_labels)
                batch_count += 1
            else:
                error_count += 1
                if error_count <= 10:
                    logging.warning("HTTP %d: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            error_count += 1
            if error_count <= 5:
                logging.warning("Request error: %s", exc)

        # Periodic ground truth flush
        if len(pending_ids) >= GT_WRITE_EVERY * BATCH_SIZE:
            try:
                write_ground_truth(session, pending_ids, pending_labels, overwrite=first_write)
                first_write = False
                pending_ids.clear()
                pending_labels.clear()
            except Exception as exc:
                logging.error("Ground truth write failed: %s", exc)

        # Progress log
        if batch_count > 0 and batch_count % LOG_EVERY == 0:
            elapsed_total = prior_elapsed + (time.time() - run_start)
            pct = elapsed_total / DURATION_SECONDS * 100
            logging.info(
                "%.1f%% | %.0f/%ds | %d batches (%d preds) | %d errors",
                pct, elapsed_total, DURATION_SECONDS,
                batch_count, batch_count * BATCH_SIZE, error_count,
            )

        # Periodic checkpoint
        if time.time() - last_checkpoint_at >= CHECKPOINT_EVERY:
            elapsed_total = prior_elapsed + (time.time() - run_start)
            save_checkpoint({
                "total_duration":  DURATION_SECONDS,
                "elapsed_seconds": elapsed_total,
                "batch_count":     batch_count,
                "error_count":     error_count,
                "start_timestamp": start_ts,
            })
            last_checkpoint_at = time.time()

        time.sleep(SLEEP_BETWEEN)

    # ── Final flush ──────────────────────────────────────────────────────────
    if pending_ids:
        try:
            write_ground_truth(session, pending_ids, pending_labels, overwrite=first_write)
        except Exception as exc:
            logging.error("Final ground truth write failed: %s", exc)

    elapsed_total = prior_elapsed + (time.time() - run_start)
    save_checkpoint({
        "total_duration":  DURATION_SECONDS,
        "elapsed_seconds": elapsed_total,
        "batch_count":     batch_count,
        "error_count":     error_count,
        "start_timestamp": start_ts,
        "completed":       True,
    })

    logging.info(
        "Done! %d batches (%d predictions) | %d errors | %.0fs total",
        batch_count, batch_count * BATCH_SIZE, error_count, elapsed_total,
    )
    session.close()


if __name__ == "__main__":
    main()
