"""
Banking Analytics Dashboard — Streamlit
Reproduces the Descriptive / Diagnostic / Predictive analysis from the
banking_database_project notebook, reading CSV exports instead of
Snowflake/Databricks (which Streamlit Cloud can't reach).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from auth import check_login, logout_button
from utils import load_tables, build_derived, TABLE_FILES

# Must run before any other Streamlit call — shows a login form and st.stop()s
# the script until the user authenticates.
check_login()

st.set_page_config(page_title="Banking Analytics Dashboard", layout="wide", page_icon="🏦")
sns.set_style("whitegrid")
logout_button()

# --------------------------------------------------------------------------
# Sidebar — data source
# --------------------------------------------------------------------------
st.sidebar.title("🏦 Data source")
mode = st.sidebar.radio(
    "Choose data source",
    ["Bundled sample data", "Upload my own CSVs"],
    help="Bundled sample data is synthetic, generated to match the notebook's "
         "schema — replace it with real table exports for real numbers.",
)

uploaded = None
if mode == "Upload my own CSVs":
    st.sidebar.caption("Export each Databricks/Snowflake table to CSV and upload below.")
    uploaded = {}
    for name in TABLE_FILES:
        uploaded[name] = st.sidebar.file_uploader(f"{name}.csv", type="csv", key=name)
    if any(v is None for v in uploaded.values()):
        st.warning("Upload all 10 tables in the sidebar to continue, or switch to sample data.")
        st.stop()

try:
    tables = load_tables(uploaded)
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python generate_sample_data.py` once to create demo CSVs in /data, "
             "or switch to 'Upload my own CSVs'.")
    st.stop()

row_hash = tuple(len(df) for df in tables.values())
d = build_derived(row_hash, tables)

st.sidebar.markdown("---")
st.sidebar.caption(f"Loaded {sum(row_hash):,} total rows across 10 tables.")

st.title("🏦 Banking Analytics Dashboard")
st.caption("Descriptive · Diagnostic · Predictive analysis over customers, accounts, "
           "cards, loans, transactions and support tickets.")

tab_desc, tab_diag, tab_pred = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🤖 Predictive"])

# ==========================================================================
# DESCRIPTIVE — "what happened?"
# ==========================================================================
with tab_desc:
    section = st.selectbox(
        "Choose analysis",
        [
            "1. Portfolio composition (accounts / cards / loans)",
            "2. Fraud incidence by category, card type, month",
            "3. Default & late-payment rate by loan type / branch / tenure",
            "4. Support-ticket volume & resolution time",
            "5. Card spend vs fraud by merchant category",
            "6. Customer segmentation (income × credit score)",
            "7. Card utilization by card type",
            "8. Monthly transaction volume & value trend",
        ],
        key="desc_section",
    )

    accounts, cards, loan = tables["accounts"], tables["cards"], tables["loan"]
    card_txn, loan_full = d["card_txn"], d["loan_full"]

    if section.startswith("1."):
        account_mix = accounts.groupby(["ACCOUNT_TYPE", "STATUS"]).agg(
            accounts=("ACCOUNT_ID", "count"), total_balance=("BALANCE", "sum"), avg_balance=("BALANCE", "mean")
        ).reset_index().sort_values("accounts", ascending=False)
        card_mix = cards.groupby("CARD_TYPE").agg(
            cards=("CARD_ID", "count"), total_credit_balance=("CREDIT_LIMIT", "sum"),
            avg_credit_balance=("CREDIT_LIMIT", "mean")
        ).reset_index().sort_values("total_credit_balance", ascending=False)
        loan_mix = loan.groupby("LOAN_TYPE").agg(
            loans=("LOAN_ID", "count"), total_disbursed=("LOAN_AMOUNT", "sum"),
            avg_interest_rate=("INTEREST_RATE", "mean")
        ).reset_index().sort_values("total_disbursed", ascending=False)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Accounts by type")
            fig, ax = plt.subplots(figsize=(4, 4))
            account_mix.groupby("ACCOUNT_TYPE")["accounts"].sum().plot.pie(autopct="%1.0f%%", ylabel="", ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("Cards by type")
            fig, ax = plt.subplots(figsize=(4, 4))
            card_mix.set_index("CARD_TYPE")["cards"].plot.bar(color="orange", ax=ax)
            st.pyplot(fig)
        with c3:
            st.subheader("Loan volume by type")
            fig, ax = plt.subplots(figsize=(4, 4))
            loan_mix.set_index("LOAN_TYPE")["total_disbursed"].plot.bar(color="green", ax=ax)
            st.pyplot(fig)

        st.dataframe(account_mix, use_container_width=True)
        st.dataframe(card_mix, use_container_width=True)
        st.dataframe(loan_mix, use_container_width=True)

        top_account, top_loan = account_mix.iloc[0], loan_mix.iloc[0]
        st.info(
            f"**Insight:** '{top_account['ACCOUNT_TYPE']}/{top_account['STATUS']}' is the most common account "
            f"type (₹{top_account['total_balance']:,.0f} total balance). '{top_loan['LOAN_TYPE']}' is the "
            f"largest loan book (₹{top_loan['total_disbursed']:,.0f} disbursed, "
            f"{top_loan['avg_interest_rate']:.2f}% avg rate)."
        )

    elif section.startswith("2."):
        fraud_by_category = card_txn.groupby("MERCHANT_CATEGORY").agg(
            total_txn=("CARD_TXN_ID", "count"), fraud_txn=("IS_FRAUD", "sum")
        ).reset_index()
        fraud_by_category["fraud_rate_pct"] = (fraud_by_category["fraud_txn"] / fraud_by_category["total_txn"] * 100).round(2)
        fraud_by_category = fraud_by_category.sort_values("fraud_rate_pct", ascending=False)

        fraud_by_card_type = card_txn.groupby("CARD_TYPE").agg(
            total_txn=("CARD_TXN_ID", "count"), fraud_txn=("IS_FRAUD", "sum")
        ).reset_index()
        fraud_by_card_type["fraud_rate_pct"] = (fraud_by_card_type["fraud_txn"] / fraud_by_card_type["total_txn"] * 100).round(2)
        fraud_by_card_type = fraud_by_card_type.sort_values("fraud_rate_pct", ascending=False)

        fraud_by_month = card_txn.groupby("txn_month").agg(
            total_txn=("CARD_TXN_ID", "count"), fraud_txn=("IS_FRAUD", "sum")
        ).reset_index()
        fraud_by_month["fraud_rate_pct"] = (fraud_by_month["fraud_txn"] / fraud_by_month["total_txn"] * 100).round(2)
        fraud_by_month = fraud_by_month.sort_values("txn_month")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Fraud rate % by merchant category")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(fraud_by_category["MERCHANT_CATEGORY"], fraud_by_category["fraud_rate_pct"], color="crimson")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
        with c2:
            st.subheader("Fraud rate % by card type")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(fraud_by_card_type["CARD_TYPE"], fraud_by_card_type["fraud_rate_pct"], color="darkorange")
            st.pyplot(fig)

        st.subheader("Fraud rate % trend by month")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(fraud_by_month["txn_month"], fraud_by_month["fraud_rate_pct"], marker="o", color="firebrick")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

        top_cat, top_card = fraud_by_category.iloc[0], fraud_by_card_type.iloc[0]
        st.info(
            f"**Insight:** '{top_cat['MERCHANT_CATEGORY']}' is the riskiest merchant category "
            f"({top_cat['fraud_rate_pct']}% fraud rate). '{top_card['CARD_TYPE']}' cards show the highest "
            f"fraud rate among card types ({top_card['fraud_rate_pct']}%)."
        )

    elif section.startswith("3."):
        by_loan_type = loan_full.groupby("LOAN_TYPE").agg(
            loans=("LOAN_ID", "count"), default_rate_pct=("is_default", "mean"), late_rate_pct=("ever_late", "mean")
        ).reset_index()
        by_loan_type[["default_rate_pct", "late_rate_pct"]] = (by_loan_type[["default_rate_pct", "late_rate_pct"]] * 100).round(2)

        by_branch = loan_full.groupby("BRANCH_ID").agg(
            loans=("LOAN_ID", "count"), default_rate_pct=("is_default", "mean"), late_rate_pct=("ever_late", "mean")
        ).reset_index()
        by_branch[["default_rate_pct", "late_rate_pct"]] = (by_branch[["default_rate_pct", "late_rate_pct"]] * 100).round(2)

        by_tenure = loan_full.groupby("tenure_band", observed=True).agg(
            loans=("LOAN_ID", "count"), default_rate_pct=("is_default", "mean"), late_rate_pct=("ever_late", "mean")
        ).reset_index()
        by_tenure[["default_rate_pct", "late_rate_pct"]] = (by_tenure[["default_rate_pct", "late_rate_pct"]] * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Default rate % by loan type")
            fig, ax = plt.subplots(figsize=(6, 4))
            by_loan_type.set_index("LOAN_TYPE")["default_rate_pct"].plot.bar(color="steelblue", ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("Default rate % by tenure band")
            fig, ax = plt.subplots(figsize=(6, 4))
            by_tenure.set_index("tenure_band")["default_rate_pct"].plot.bar(color="darkred", ax=ax)
            st.pyplot(fig)

        st.subheader("Worst 10 branches by default rate")
        st.dataframe(by_branch.sort_values("default_rate_pct", ascending=False).head(10), use_container_width=True)

        worst = by_loan_type.sort_values("default_rate_pct", ascending=False).iloc[0]
        st.info(
            f"**Insight:** '{worst['LOAN_TYPE']}' loans have the worst default rate "
            f"({worst['default_rate_pct']}%) and a {worst['late_rate_pct']}% late-payment rate."
        )

    elif section.startswith("4."):
        support_ticket = d["support_ticket"]
        ticket_summary = support_ticket.groupby("ISSUE_TYPE").agg(
            tickets=("TICKET_ID", "count"), avg_resolution_days=("resolution_days", "mean")
        ).reset_index().sort_values("tickets", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Ticket volume by issue type")
            fig, ax = plt.subplots(figsize=(6, 4))
            ticket_summary.set_index("ISSUE_TYPE")["tickets"].plot.bar(color="teal", ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
        with c2:
            st.subheader("Avg resolution days by issue type")
            fig, ax = plt.subplots(figsize=(6, 4))
            ticket_summary.set_index("ISSUE_TYPE")["avg_resolution_days"].plot.bar(color="slateblue", ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)

        top_ticket = ticket_summary.iloc[0]
        slowest_ticket = ticket_summary.sort_values("avg_resolution_days", ascending=False).iloc[0]
        st.info(
            f"**Insight:** '{top_ticket['ISSUE_TYPE']}' generates the most tickets ({top_ticket['tickets']}). "
            f"'{slowest_ticket['ISSUE_TYPE']}' takes the longest to resolve "
            f"({slowest_ticket['avg_resolution_days']:.1f} days)."
        )

    elif section.startswith("5."):
        spend_fraud = card_txn.groupby("MERCHANT_CATEGORY").agg(
            total_spend=("AMOUNT", "sum"), total_txn=("CARD_TXN_ID", "count"),
            fraud_rate_pct=("IS_FRAUD", lambda x: round(x.mean() * 100, 2)),
        ).reset_index().sort_values("total_spend", ascending=False)

        st.subheader("Spend vs fraud rate by merchant category")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(spend_fraud["MERCHANT_CATEGORY"], spend_fraud["total_spend"], color="steelblue")
        ax1.set_ylabel("Total spend")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(spend_fraud["MERCHANT_CATEGORY"], spend_fraud["fraud_rate_pct"], color="red", marker="o")
        ax2.set_ylabel("Fraud rate %")
        st.pyplot(fig)
        st.dataframe(spend_fraud, use_container_width=True)

        top_spend = spend_fraud.iloc[0]
        top_fraud = spend_fraud.sort_values("fraud_rate_pct", ascending=False).iloc[0]
        st.info(
            f"**Insight:** '{top_spend['MERCHANT_CATEGORY']}' drives the most spend "
            f"(₹{top_spend['total_spend']:,.0f}). '{top_fraud['MERCHANT_CATEGORY']}' has the highest fraud "
            f"rate ({top_fraud['fraud_rate_pct']}%)."
        )

    elif section.startswith("6."):
        customers_with_balance = d["customers_with_balance"]
        segmentation = customers_with_balance.groupby(["income_band", "credit_band"], observed=True).agg(
            customers=("CUSTOMER_ID", "count"), avg_balance=("BALANCE", "mean")
        ).reset_index().sort_values(["income_band", "credit_band"])

        st.subheader("Avg account balance by income & credit-score band")
        pivot = segmentation.pivot(index="income_band", columns="credit_band", values="avg_balance")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_ylabel("Income Band"); ax.set_xlabel("Credit Score Band")
        st.pyplot(fig)
        st.dataframe(segmentation, use_container_width=True)

        top_seg = segmentation.sort_values("avg_balance", ascending=False).iloc[0]
        st.info(
            f"**Insight:** Highest avg balance ({top_seg['avg_balance']:.0f}) comes from income band "
            f"'{top_seg['income_band']}' and credit band '{top_seg['credit_band']}'."
        )

    elif section.startswith("7."):
        card_util = d["card_util"]
        util_by_type = card_util.groupby("CARD_TYPE")["utilization_pct"].mean().round(2).reset_index().sort_values(
            "utilization_pct", ascending=False
        )

        st.subheader("Avg utilization % by card type")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(util_by_type["CARD_TYPE"], util_by_type["utilization_pct"], color="darkcyan")
        ax.set_ylabel("Utilization %")
        st.pyplot(fig)
        st.dataframe(util_by_type, use_container_width=True)

        top_util = util_by_type.iloc[0]
        st.info(
            f"**Insight:** '{top_util['CARD_TYPE']}' cards run the highest average utilization "
            f"({top_util['utilization_pct']}%) — candidates for credit-limit review or targeted offers."
        )

    elif section.startswith("8."):
        transaction_list = d["transaction_list"]
        monthly_trend = transaction_list.groupby("txn_month").agg(
            txn_count=("TRANSACTION_ID", "count"), total_amount=("AMOUNT", "sum")
        ).reset_index().sort_values("txn_month")

        st.subheader("Monthly transaction value trend")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly_trend["txn_month"], monthly_trend["total_amount"], marker="o")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
        st.dataframe(monthly_trend, use_container_width=True)

# ==========================================================================
# DIAGNOSTIC — "why did it happen?"
# ==========================================================================
with tab_diag:
    section = st.selectbox(
        "Choose analysis",
        [
            "1. Credit-score band vs default & late-payment rate",
            "2. Merchant category disproportionate fraud rate",
            "3. Prior fraud-report ticket vs subsequent confirmed fraud",
            "4. Branch actual vs expected default rate (loan-mix adjusted)",
            "5. Escalated tickets vs account dormancy/closure",
            "6. Card utilization band vs fraud rate",
            "7. Account tenure vs dormancy/closure",
        ],
        key="diag_section",
    )

    loan_full, card_txn = d["loan_full"], d["card_txn"]
    customers, card_util = d["customers"], d["card_util"]
    accounts, support_ticket = d["accounts"], d["support_ticket"]
    cards = tables["cards"]

    if section.startswith("1."):
        loan_cust = loan_full.merge(customers[["CUSTOMER_ID", "credit_band"]], on="CUSTOMER_ID", how="left")
        loan1 = loan_cust.groupby("credit_band", observed=True).agg(
            loans=("LOAN_ID", "count"),
            defaults_rate_pct=("is_default", lambda x: round(x.mean() * 100, 2)),
            late_rate_pct=("ever_late", lambda x: round(x.mean() * 100, 2)),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="credit_band", y="defaults_rate_pct", data=loan1, ax=ax)
            ax.set_title("Default Rate by Credit Score Band")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="credit_band", y="late_rate_pct", data=loan1, ax=ax)
            ax.set_title("Late Payment Rate by Credit Score Band")
            st.pyplot(fig)
        st.dataframe(loan1, use_container_width=True)

        top_def = loan1.sort_values("defaults_rate_pct", ascending=False).iloc[0]
        top_late = loan1.sort_values("late_rate_pct", ascending=False).iloc[0]
        st.info(
            f"**Insight:** Highest default rate ({top_def['defaults_rate_pct']}%) comes from credit band "
            f"'{top_def['credit_band']}'. Highest late-payment rate ({top_late['late_rate_pct']}%) comes "
            f"from credit band '{top_late['credit_band']}'."
        )

    elif section.startswith("2."):
        fraud_by_category = card_txn.groupby("MERCHANT_CATEGORY").agg(
            total_txn=("CARD_TXN_ID", "count"), fraud_txn=("IS_FRAUD", "sum")
        ).reset_index()
        fraud_by_category["fraud_rate_pct"] = (fraud_by_category["fraud_txn"] / fraud_by_category["total_txn"] * 100).round(2)
        overall_fraud_rate = card_txn["IS_FRAUD"].mean() * 100
        cat_fraud = fraud_by_category.copy()
        cat_fraud["disproportionate"] = cat_fraud["fraud_rate_pct"] - overall_fraud_rate
        cat_fraud = cat_fraud.sort_values("disproportionate", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="MERCHANT_CATEGORY", y="disproportionate", data=cat_fraud, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Disproportionate Fraud Rate vs Overall Baseline")
        st.pyplot(fig)
        st.dataframe(cat_fraud, use_container_width=True)
        st.caption(f"Overall baseline fraud rate: {overall_fraud_rate:.2f}%")

    elif section.startswith("3."):
        fraud_reporters = support_ticket[support_ticket["ISSUE_TYPE"] == "Fraud Report"][
            ["CUSTOMER_ID", "DATE_OPENED"]
        ].rename(columns={"DATE_OPENED": "fraud_report_date"})
        card_cust = cards[["CARD_ID", "CUSTOMER_ID"]]
        card_txn_cust = card_txn.merge(card_cust, on="CARD_ID", how="left")
        txn_with_reports = card_txn_cust.merge(fraud_reporters, on="CUSTOMER_ID", how="left")
        txn_with_reports["had_prior_fraud_report"] = (
            ~txn_with_reports["fraud_report_date"].isna()
        ) & (txn_with_reports["TXN_DATE"] >= txn_with_reports["fraud_report_date"])

        fraud_correlation = txn_with_reports.groupby("had_prior_fraud_report").agg(
            transactions=("CARD_TXN_ID", "count"), fraud_count=("IS_FRAUD", "sum"),
            fraud_rate_pct=("IS_FRAUD", lambda x: round(x.mean() * 100, 2)),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["No Prior Report", "Had Prior Report"], fraud_correlation["fraud_rate_pct"],
               color=["steelblue", "coral"])
        ax.set_title("Fraud Rate: With vs Without Prior Fraud Report")
        st.pyplot(fig)
        st.dataframe(fraud_correlation, use_container_width=True)

        if len(fraud_correlation) == 2:
            rate_with = fraud_correlation.loc[fraud_correlation["had_prior_fraud_report"] == True, "fraud_rate_pct"].values[0]
            rate_without = fraud_correlation.loc[fraud_correlation["had_prior_fraud_report"] == False, "fraud_rate_pct"].values[0]
            lift = rate_with - rate_without
            st.info(
                f"**Insight:** Prior reporters show a {rate_with}% fraud rate vs {rate_without}% for others "
                f"(lift: {lift:.2f} pts). "
                + ("Prior fraud reporting is a strong predictor of subsequent confirmed fraud."
                   if lift > 0 else "No meaningful link found.")
            )

    elif section.startswith("4."):
        branches = tables["branches"]
        network_avg = loan_full.groupby("LOAN_TYPE")["is_default"].mean().reset_index().rename(
            columns={"is_default": "network_default_rate"}
        )
        loan_with_expected = loan_full.merge(network_avg, on="LOAN_TYPE", how="left")
        branch_analysis = loan_with_expected.groupby("BRANCH_ID").agg(
            loans=("LOAN_ID", "count"), actual_default_rate=("is_default", "mean"),
            expected_default_rate=("network_default_rate", "mean"),
        ).reset_index().merge(branches[["BRANCH_ID", "BRANCH_NAME"]], on="BRANCH_ID")
        branch_analysis["excess_default_pct"] = (
            (branch_analysis["actual_default_rate"] - branch_analysis["expected_default_rate"]) * 100
        ).round(2)
        branch_analysis["actual_default_pct"] = (branch_analysis["actual_default_rate"] * 100).round(2)
        branch_analysis["expected_default_pct"] = (branch_analysis["expected_default_rate"] * 100).round(2)

        top_n = branch_analysis.reindex(
            branch_analysis["excess_default_pct"].abs().sort_values(ascending=False).index
        ).head(15).sort_values("excess_default_pct")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_n["BRANCH_NAME"], top_n["excess_default_pct"],
                color=["red" if x > 0 else "blue" for x in top_n["excess_default_pct"]])
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title("Top 15 Branches by |Excess Default Rate| (red=worse, blue=better)")
        st.pyplot(fig)
        st.dataframe(
            branch_analysis[["BRANCH_ID", "BRANCH_NAME", "loans", "actual_default_pct",
                              "expected_default_pct", "excess_default_pct"]].sort_values(
                "excess_default_pct", ascending=False
            ),
            use_container_width=True,
        )

    elif section.startswith("5."):
        customers_tbl = tables["customers"]
        escalated_status = [s for s in support_ticket["STATUS"].unique() if s not in ["closed", "resolved"]]
        escalated_customers = set(support_ticket[support_ticket["STATUS"].isin(escalated_status)]["CUSTOMER_ID"].unique())

        account_cust = accounts.merge(customers_tbl[["CUSTOMER_ID"]], on="CUSTOMER_ID")
        account_cust["had_escalated_ticket"] = account_cust["CUSTOMER_ID"].isin(escalated_customers)
        account_mix = pd.crosstab(account_cust["STATUS"], account_cust["had_escalated_ticket"], normalize="index") * 100

        fig, ax = plt.subplots(figsize=(9, 5))
        account_mix.plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax)
        ax.set_title("Account Status Mix: Escalated-Ticket Customers vs Others")
        st.pyplot(fig)
        st.dataframe(account_mix.round(2), use_container_width=True)

    elif section.startswith("6."):
        card_util_fraud = card_util.groupby("util_band", observed=True)["fraud_on_card"].mean().mul(100).round(2)
        fig, ax = plt.subplots(figsize=(9, 5))
        card_util_fraud.plot(kind="bar", color="orange", ax=ax)
        ax.set_title("Card Fraud Rate by Utilization Band")
        st.pyplot(fig)
        st.dataframe(card_util_fraud.reset_index(name="fraud_rate_pct"), use_container_width=True)
        top_band = card_util_fraud.idxmax()
        st.info(f"**Insight:** Utilization band **{top_band}** has the highest fraud rate ({card_util_fraud[top_band]}%).")

    elif section.startswith("7."):
        acc_age = pd.crosstab(accounts["account_age_band"], accounts["STATUS"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        acc_age.plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax)
        ax.set_title("Account Status Mix by Age Band")
        st.pyplot(fig)
        st.dataframe(acc_age.round(2), use_container_width=True)

# ==========================================================================
# PREDICTIVE — "what's likely to happen?"
# ==========================================================================
with tab_pred:
    st.write(
        "Train the three notebook models on the currently loaded data. Models train in-memory "
        "each run (fast on sample data; may take longer on large real exports)."
    )
    model_choice = st.selectbox(
        "Choose model",
        [
            "1. Fraud detection (card transactions)",
            "2. Credit-default prediction (loans)",
            "3. Next-late-payment prediction (active loans)",
        ],
    )
    train_btn = st.button("Train model", type="primary")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        roc_auc_score, confusion_matrix, classification_report,
        precision_score, recall_score, roc_curve,
    )

    def plot_confusion(cm, title, labels=("Legit/On-time", "Fraud/Late")):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
        st.pyplot(fig)

    def metrics_table(names, roc_aucs, y_tests, y_preds):
        rows = []
        for n, auc, yt, yp in zip(names, roc_aucs, y_tests, y_preds):
            rows.append({
                "Model": n, "ROC-AUC": round(auc, 4),
                "Recall (positive class)": round(recall_score(yt, yp, pos_label=1), 4),
                "Precision (positive class)": round(precision_score(yt, yp, pos_label=1, zero_division=0), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if train_btn:
        if model_choice.startswith("1."):
            card_txn = d["card_txn"]
            feature_cols = ["AMOUNT", "CARD_TYPE", "MERCHANT_CATEGORY", "days_since_txn"]
            model_df = card_txn[feature_cols + ["IS_FRAUD"]].dropna().copy()
            le1, le2 = LabelEncoder(), LabelEncoder()
            model_df["CARD_TYPE"] = le1.fit_transform(model_df["CARD_TYPE"])
            model_df["MERCHANT_CATEGORY"] = le2.fit_transform(model_df["MERCHANT_CATEGORY"])
            X, y = model_df.drop(columns="IS_FRAUD"), model_df["IS_FRAUD"]
            if y.nunique() < 2:
                st.error("Not enough class variety in IS_FRAUD to train — check your data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

                lr = LogisticRegression().fit(X_train_s, y_train)
                y_pred_lr, y_prob_lr = lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1]
                roc_lr = roc_auc_score(y_test, y_prob_lr)

                rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf, y_prob_rf = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
                roc_rf = roc_auc_score(y_test, y_prob_rf)

                st.subheader("Model performance")
                metrics_table(["Logistic Regression", "Random Forest"], [roc_lr, roc_rf],
                               [y_test, y_test], [y_pred_lr, y_pred_rf])

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
                    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
                    ax.plot(fpr_lr, tpr_lr, label=f"LR (AUC {roc_lr:.3f})")
                    ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC {roc_rf:.3f})")
                    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
                    st.pyplot(fig)
                with c2:
                    plot_confusion(confusion_matrix(y_test, y_pred_rf), "Random Forest — Confusion Matrix",
                                    labels=("Legit", "Fraud"))

        elif model_choice.startswith("2."):
            loan_full, customers = d["loan_full"], tables["customers"]
            default_df = loan_full.merge(customers[["CUSTOMER_ID", "CREDIT_SCORE", "ANNUAL_INCOME"]], on="CUSTOMER_ID")
            default_df["tenure_days"] = default_df["tenure_days"].fillna(0)
            feature_cols = ["ANNUAL_INCOME", "CREDIT_SCORE", "tenure_days", "LOAN_AMOUNT", "INTEREST_RATE", "ever_late"]
            model_df = default_df[feature_cols + ["is_default"]].dropna()
            X, y = model_df.drop(columns="is_default"), model_df["is_default"]
            if y.nunique() < 2:
                st.error("Not enough class variety in is_default to train — check your data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

                lr = LogisticRegression().fit(X_train_s, y_train)
                y_pred_lr, y_prob_lr = lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1]
                roc_lr = roc_auc_score(y_test, y_prob_lr)

                rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf, y_prob_rf = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
                roc_rf = roc_auc_score(y_test, y_prob_rf)

                st.subheader("Model performance")
                metrics_table(["Logistic Regression", "Random Forest"], [roc_lr, roc_rf],
                               [y_test, y_test], [y_pred_lr, y_pred_rf])
                plot_confusion(confusion_matrix(y_test, y_pred_rf), "Random Forest — Confusion Matrix",
                                labels=("No Default", "Default"))

                importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.subheader("Feature importance (Random Forest)")
                st.bar_chart(importance)
                st.info(f"**Insight:** '{importance.index[0]}' is the strongest driver of default risk, "
                        f"contributing {importance.iloc[0]*100:.1f}% of the model's decision weight.")

        else:  # model 3
            loan_full, customers = d["loan_full"], tables["customers"]
            loan_payment = tables["loan_payment"]
            pay_stats = loan_payment.groupby("LOAN_ID").agg(
                payments_made=("PAYMENT_ID", "count"), late_payments=("LATE_PAYMENT_FLAG", "sum")
            ).reset_index()
            pay_stats["late_ratio"] = pay_stats["late_payments"] / pay_stats["payments_made"]

            p3_df = loan_full.merge(pay_stats, on="LOAN_ID", how="left")
            for c in ["payments_made", "late_payments", "late_ratio"]:
                p3_df[c] = p3_df[c].fillna(0)
            p3_df = p3_df.merge(customers[["CUSTOMER_ID", "CREDIT_SCORE"]], on="CUSTOMER_ID")

            feat_cols = ["CREDIT_SCORE", "LOAN_AMOUNT", "INTEREST_RATE", "payments_made", "late_ratio"]
            model_df = p3_df[feat_cols + ["ever_late"]].dropna()
            X, y = model_df.drop(columns="ever_late"), model_df["ever_late"]
            if y.nunique() < 2:
                st.error("Not enough class variety in ever_late to train — check your data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

                lr = LogisticRegression().fit(X_train_s, y_train)
                y_pred_lr, y_prob_lr = lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1]
                roc_lr = roc_auc_score(y_test, y_prob_lr)

                rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf, y_prob_rf = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
                roc_rf = roc_auc_score(y_test, y_prob_rf)

                st.subheader("Model performance")
                metrics_table(["Logistic Regression", "Random Forest"], [roc_lr, roc_rf],
                               [y_test, y_test], [y_pred_lr, y_pred_rf])
                plot_confusion(confusion_matrix(y_test, y_pred_rf), "Random Forest — Confusion Matrix",
                                labels=("On Time", "Late"))

                active_loans = p3_df[p3_df["STATUS"] == "Active"].copy()
                if len(active_loans):
                    active_loans["late_risk_score"] = rf.predict_proba(active_loans[feat_cols].fillna(0))[:, 1]
                    st.subheader("Top 10 highest-risk active loans")
                    st.dataframe(
                        active_loans[["LOAN_ID", "CUSTOMER_ID", "late_risk_score"]]
                        .sort_values("late_risk_score", ascending=False).head(10),
                        use_container_width=True,
                    )
    else:
        st.info("Pick a model above and click **Train model** to run it on the loaded data.")
