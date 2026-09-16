# Banking Analytics Dashboard (Streamlit)

A Streamlit port of the `banking_database_project` notebook — Descriptive, Diagnostic,
and Predictive analysis over 10 banking tables (customers, branches, employees, accounts,
cards, loans, loan payments, transactions, card transactions, support tickets).

The original notebook pulled data from **Snowflake → Databricks**. Streamlit Cloud can't
reach either of those directly, so this app instead reads **CSV exports** of the same 10
tables. A synthetic sample dataset is included so the app works immediately; swap in your
real exports whenever you're ready (see below).

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app (UI + all analysis/plots/models) |
| `auth.py` | Username/password login gate |
| `utils.py` | Data loading & feature-engineering helpers (cached) |
| `generate_sample_data.py` | Creates synthetic demo CSVs in `/data` |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme + server settings |
| `.streamlit/secrets.toml` | Your real login credentials (never commit this) |
| `.streamlit/secrets.toml.example` | Template showing the expected format |
| `data/*.csv` | The 10 source tables (sample data by default) |
| `.gitignore` | Standard Python/Streamlit ignores (also ignores `secrets.toml`) |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# one-time: create sample data so the app has something to show
python generate_sample_data.py

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Log in with the
credentials in `.streamlit/secrets.toml` (default: `admin` / `changeme123` —
**change this before sharing the app with anyone**).

## Login / access control

The app is gated by a simple username/password screen (`auth.py`). Credentials
are stored in `.streamlit/secrets.toml`, which is **git-ignored** so real
passwords never get committed. Add as many users as you like:

```toml
[credentials]
admin = "some-strong-password"
analyst1 = "another-password"
```

- **Locally**: edit `.streamlit/secrets.toml` directly (copy it from
  `.streamlit/secrets.toml.example` if it's missing).
- **On Streamlit Community Cloud**: don't upload `secrets.toml` at all — instead
  go to your app → **Settings → Secrets** and paste the same `[credentials]`
  block there. Streamlit injects it as `st.secrets` at runtime.

This is a lightweight gate meant for keeping a dashboard private among a small
trusted team (passwords are plain text in secrets, compared in the app). If you
need SSO, per-user roles, audit logs, or password resets, put this behind a
proper identity provider (e.g. an auth proxy, or Streamlit Cloud's built-in
viewer-restriction / SSO features on paid tiers) instead.

## Using your real data

Export each of these 10 Databricks/Snowflake tables to CSV with the **same column names**
used in the notebook (upper-case, e.g. `CUSTOMER_ID`, `ANNUAL_INCOME`, `IS_FRAUD`, …):

```
customers, branches, employees, accounts, cards, loan,
loan_payment, transaction_list, card_transaction, support_ticket
```

Then either:
- **Replace the files** in `/data` (`customers.csv`, `branches.csv`, … matching names above), or
- Run the app and use the **"Upload my own CSVs"** option in the sidebar — no file changes needed,
  good for one-off exploration or for teammates without repo access.

If a table's real export uses slightly different capitalization, `utils.py` upper-cases all
column headers on load, so mixed case is fine — just keep the underlying names the same
(e.g. `CARD_TYPE`, not `card type`).

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **public or private GitHub repo** (include `/data` if you want the
   sample data to ship with the app, or upload real CSVs there if you're comfortable
   committing them — otherwise use the in-app uploader instead and leave `/data` empty besides
   `.gitkeep`).
2. Go to **https://share.streamlit.io** → **New app**.
3. Pick your repo, branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.
5. Any time you push to the branch, the app redeploys automatically.

## Deploy elsewhere (Render / Railway / Docker / your own server)

Any host that can run `pip install -r requirements.txt && streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
will work. A minimal `Dockerfile` if you need containerized deploy:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Notes

- The **Predictive** tab trains Logistic Regression + Random Forest models live, in-memory,
  each time you click "Train model" (mirrors the notebook's 3 models: fraud detection,
  credit-default, next-late-payment). This is fine at sample-data scale; on large real
  exports (hundreds of thousands+ rows) training may take longer — consider pre-training
  and pickling models offline and loading them instead if that becomes slow (the original
  notebook already saves `fraud_model.pkl`, `credit_default_model.pkl`, `late_payment_model.pkl`
  for this purpose — a `models/` folder is included if you want to wire that in).
- Snowflake credentials were hard-coded in the original notebook cell — **do not commit
  credentials to this repo**. This app never connects to Snowflake/Databricks; it only reads
  local/uploaded CSVs.
