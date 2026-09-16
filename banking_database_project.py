# Converted from banking_database_project (1).ipynb
# Auto-converted: markdown cells become comment blocks, code cells kept as-is.
# NOTE: this notebook was written for a Databricks environment (uses the implicit
# `spark` session and Databricks %pip magics) plus a live Snowflake connection.
# It will NOT run as-is outside Databricks — see comments below for what to change.


# ---- Cell 1 ------------------------------------------------------------
# Step 1: Python-Snowflake connector install/import
try:
    import snowflake.connector
except ImportError:
# [Databricks magic, not valid in plain Python]:     %pip install snowflake-connector-python
    import snowflake.connector

import pandas as pd

# Step 2: Connect directly to Snowflake
# SECURITY NOTE: the original notebook had a real username/password hard-coded
# here. That has been removed — do not commit real credentials to source
# control. Use environment variables (or Databricks/Streamlit secrets) instead.
import os
ctx = snowflake.connector.connect(
    user=os.environ.get("SNOWFLAKE_USER"),
    password=os.environ.get("SNOWFLAKE_PASSWORD"),
    account=os.environ.get("SNOWFLAKE_ACCOUNT"),  # e.g. 'xxxxx-xxxxxxx' from the Snowflake URL
    warehouse='bank_warehouse',
    database='banking_dataset',
    schema='core'
)

cursor = ctx.cursor()  ##creates a cursor object from the Snowflake connection (ctx) to execute queries.
print("⚡ from snowflake created a python connection\n")

# Step 3: setup local databricks database
spark.sql("CREATE DATABASE IF NOT EXISTS banking_project_db")

# list of 10 tables
snowflake_tables = [
    "CUSTOMERS",
    "BRANCHES",
    "EMPLOYEES",
    "ACCOUNTS",
    "CARDS",
    "LOAN",
    "LOAN_PAYMENT",
    "TRANSACTION_LIST",
    "CARD_TRANSACTION",
    "SUPPORT_TICKET"
]

# Step 4: Fetching data through a loop using Pandas and inserting it into Databricks catalog

for table_name in snowflake_tables:
    try:
        print(f"🔄 Fetching data for table: {table_name}...")

        # 1. Loading data into a Pandas DataFrame using a standard SQL query.
        sql_query = f"SELECT * FROM {table_name}"
        data = pd.read_sql(sql_query, ctx)

        # 2. Converting this standard Pandas DataFrame into a Spark DataFrame so it can be saved into the database.
        spark_df = spark.createDataFrame(data)

        # 3. Saving it into the Databricks catalog.
        short_name = table_name.lower()
        full_table_name = f"banking_project_catalog.banking_project_db.{short_name}"
        spark_df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)

        print(f"✅ Success! Saved as: {full_table_name} (Rows: {len(data)})\n")

    except Exception as e:
        print(f"❌ Error loading {table_name}: {str(e)}\n")

# close connection
cursor.close()
ctx.close()
print("🎉 Finally! All the tables have been safely saved in the Databricks Catalog database!")

# ---- Cell 2 ------------------------------------------------------------
# Verify row counts for all tables (Databricks side only, no Snowflake connection needed)

tables = ["customers", "branches", "employees", "accounts", "cards",
          "loan", "loan_payment", "transaction_list", "card_transaction", "support_ticket"]

for table in tables:
    count = spark.table(f"banking_project_catalog.banking_project_db.{table}").count()
    print(f"{table}: {count} rows")

# ============================================================================
# ### Import all required **_libraries_**
# ============================================================================

# ---- Cell 3 ------------------------------------------------------------
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# ---- Cell 4 ------------------------------------------------------------
CATALOG = "banking_project_catalog.banking_project_db"
 
customers          = spark.read.table(f"{CATALOG}.customers").toPandas()
branches           = spark.read.table(f"{CATALOG}.branches").toPandas()
employees          = spark.read.table(f"{CATALOG}.employees").toPandas()
accounts           = spark.read.table(f"{CATALOG}.accounts").toPandas()
cards              = spark.read.table(f"{CATALOG}.cards").toPandas()
loan               = spark.read.table(f"{CATALOG}.loan").toPandas()
loan_payment       = spark.read.table(f"{CATALOG}.loan_payment").toPandas()
transaction_list   = spark.read.table(f"{CATALOG}.transaction_list").toPandas()
card_transaction   = spark.read.table(f"{CATALOG}.card_transaction").toPandas()
support_ticket     = spark.read.table(f"{CATALOG}.support_ticket").toPandas()

# ---- Cell 5 ------------------------------------------------------------
tables={
    'customer':customers,
    'branches':branches,
    'employees':employees,
    'accounts':accounts,
    'cards':cards,
    'loan':loan,
    'loan_payment':loan_payment,
    'transaction_list':transaction_list,
    'card_transaction':card_transaction,
    'support_ticket':support_ticket
}
for name,df in tables.items():
    print(f"Table: {name}")
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())


# ---- Cell 6 ------------------------------------------------------------
# ensures date column are actual datetimes
for df,cols in [
    (accounts,['OPEN_DATE']),
    (branches,['OPENED_DATE']),
    (cards,['ISSUE_DATE','EXPIRY_DATE']),
    (customers,['JOIN_DATE','DATE_OF_BIRTH']),
    (employees,['HIRE_DATE']),
    (loan,['START_DATE']),
    (loan_payment,['PAYMENT_DATE']),
    (transaction_list,['TXN_DATE']) if "txn_date" in transaction_list.columns else (transaction_list, []),
    (card_transaction,['TXN_DATE']) if "txn_date" in card_transaction.columns else (card_transaction, []),
    (support_ticket,['DATE_OPENED','DATE_RESOLVED'])
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
today=pd.Timestamp(datetime.now().date())

# ---- Cell 7 ------------------------------------------------------------
print(accounts["OPEN_DATE"].dtype)
print(customers["JOIN_DATE"].dtype)
print(today)

# ============================================================================
# ### **Descriptive analysis  (what happened?)**
# ============================================================================

# ============================================================================
# 1. Portfolio composition-account, card & loan mix; balances and credit limits by segment.
#
# Question:-How is the customer based distributed across account types, card types, loan types, and what balances/limits sit behind each?
#
# ============================================================================

# ---- Cell 8 ------------------------------------------------------------
# 1a. Account mix — count, total & avg balance by type+status
account_mix = accounts.groupby(["ACCOUNT_TYPE","STATUS"]).agg(
    accounts=("ACCOUNT_ID","count"), total_balance=("BALANCE","sum"), avg_balance=("BALANCE","mean")
).reset_index().sort_values('accounts', ascending=False)
print(account_mix)

#Plot — accounts by type (pie chart)
plt.subplots(figsize=(6,3))
account_mix.groupby("ACCOUNT_TYPE")['accounts'].sum().plot.pie(autopct='%1.0f%%', title='Accounts by type')
plt.ylabel("")
plt.show()

# ---- Cell 9 ------------------------------------------------------------
# 1b. Card mix — count, total & avg credit limit by type
card_mix = cards.groupby("CARD_TYPE").agg(
    cards=("CARD_ID","count"),
    total_credit_balance=("CREDIT_LIMIT","sum"), 
    avg_credit_balance=("CREDIT_LIMIT","mean")
).reset_index().sort_values('total_credit_balance', ascending=False)
print(card_mix)

#Plot — cards by type (bar chart)
plt.subplots(figsize=(6,3))
card_mix.set_index("CARD_TYPE")["cards"].plot.bar(title="Cards by type", color="orange")
plt.show()

# ---- Cell 10 ------------------------------------------------------------
# 1c. Loan mix — count, total disbursed & avg interest rate by type
loan_mix = loan.groupby("LOAN_TYPE").agg(
    loans=("LOAN_ID","count"), 
    total_disbursed=("LOAN_AMOUNT","sum"), 
    avg_interest_rate=("INTEREST_RATE","mean")
).reset_index().sort_values('total_disbursed', ascending=False)
print(loan_mix)

#Plot — loan volume by type (bar chart)
plt.subplots(figsize=(6,3))
loan_mix.set_index("LOAN_TYPE")["total_disbursed"].plot.bar(title="Loan volume by type", color="green")
plt.show()

# ---- Cell 11 ------------------------------------------------------------
# 1e. Insight
top_account = account_mix.iloc[0]
top_loan = loan_mix.iloc[0]
print(f"\n Insight: '{top_account['ACCOUNT_TYPE']}/{top_account['STATUS']}' is the most common account type "
      f"\n (₹{top_account['total_balance']:,.0f} total balance). '{top_loan['LOAN_TYPE']}' is the largest loan "
      f"\n book (₹{top_loan['total_disbursed']:,.0f} disbursed, {top_loan['avg_interest_rate']:.2f}% avg rate).")
 

# ============================================================================
# 2. Fraud incidence and value by merchant category, card type and month.
#
# Question: where does fraud concentrates- which merchant category , card type, and which calender month carries the most fraud volume and values
# ============================================================================

# ---- Cell 12 ------------------------------------------------------------
#2a. join card transaction to cards to pull in card_type and extract transaction month 
card_txn=card_transaction.merge(cards[['CARD_ID','CARD_TYPE']],on='CARD_ID',how='left')
# ensure TXN_DATE is datetime before extracting period
card_txn['TXN_DATE'] = pd.to_datetime(card_txn['TXN_DATE'], errors='coerce')
card_txn['txn_month']=card_txn['TXN_DATE'].dt.to_period('M').astype(str) 

print(card_txn)

# ---- Cell 13 ------------------------------------------------------------
#2b. fraud rate & value by merchant category
fraud_by_category=card_txn.groupby('MERCHANT_CATEGORY').agg(
    total_txn=('CARD_TXN_ID','count'),
    fraud_txn=('IS_FRAUD','sum'),
).reset_index()
fraud_by_category['fraud_rate_pct']=(fraud_by_category['fraud_txn']/fraud_by_category['total_txn']*100).round(2)
fraud_by_category=fraud_by_category.sort_values('fraud_rate_pct',ascending=False)
print('fraud by merchant category:\n',fraud_by_category)

#Plot — fraud rate by merchant category
plt.figure(figsize=(8,4))
plt.bar(fraud_by_category["MERCHANT_CATEGORY"], fraud_by_category["fraud_rate_pct"], color="crimson")
plt.title("Fraud rate % by merchant category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Cell 14 ------------------------------------------------------------
#2c. fraud rate and value by card type
fraud_by_card_type=card_txn.groupby('CARD_TYPE').agg(
    total_txn=('CARD_TXN_ID','count'),
    fraud_txn=('IS_FRAUD','sum'),
).reset_index()
fraud_by_card_type['fraud_rate_pct']=(fraud_by_card_type['fraud_txn']/fraud_by_card_type['total_txn']*100).round(2)
fraud_by_card_type=fraud_by_card_type.sort_values('fraud_rate_pct',ascending=False)
print('fraud by card type:\n',fraud_by_card_type)

#Plot — fraud rate by card type
plt.figure(figsize=(8,4))
plt.bar(fraud_by_card_type["CARD_TYPE"], fraud_by_card_type["fraud_rate_pct"], color="darkorange")
plt.title("Fraud rate % by card type")
plt.tight_layout()
plt.show()

# ---- Cell 15 ------------------------------------------------------------
#2d. fraud rate and value by month 
fraud_by_month=card_txn.groupby('txn_month').agg(
        total_txn=('CARD_TXN_ID','count'),
        fraud_txn=('IS_FRAUD','sum'),
    ).reset_index()
fraud_by_month['fraud_rate_pct']=(fraud_by_month['fraud_txn']/fraud_by_month['total_txn']*100).round(2)
print(fraud_by_month.sort_values("txn_month"))

#Plot — fraud rate trend by month
plt.figure(figsize=(8,4))
plt.plot(fraud_by_month.sort_values("txn_month")["txn_month"], fraud_by_month.sort_values("txn_month")["fraud_rate_pct"], marker="o", color="firebrick")
plt.title("Fraud rate % by month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Cell 16 ------------------------------------------------------------
#2e. insights 
top_fraud_cardtype=fraud_by_card_type.iloc[0]
top_fraud_category=fraud_by_category.iloc[0]
print(f"Insight: '{top_fraud_category['MERCHANT_CATEGORY']}' is the riskiest merchant category "
      f"({top_fraud_category['fraud_rate_pct']}% fraud rate). '{top_fraud_cardtype['CARD_TYPE']}' cards show the "
      f"highest fraud rate among card types ({top_fraud_cardtype['fraud_rate_pct']}%).")

# ============================================================================
# 3.Default/write-off & late-payment rate by loan type, branch and tenure band.
#
# Question:-Which loan types, branches, and loan-age bands carry the worst repayment performance?
# ============================================================================

# ---- Cell 17 ------------------------------------------------------------
# 3a. Flag each loan "ever late"
ever_late = loan_payment.groupby("LOAN_ID")["LATE_PAYMENT_FLAG"].max().rename("ever_late").reset_index()
 

# ---- Cell 18 ------------------------------------------------------------
# 3b. Build loan_full — is_default flag + tenure band
loan_full = loan.merge(ever_late, on="LOAN_ID", how="left")
loan_full["ever_late"] = loan_full["ever_late"].fillna(0)
loan_full["is_default"] = loan_full["STATUS"].isin(["Defaulted","Written Off"]).astype(int)
loan_full["tenure_days"] = (pd.Timestamp.today() - loan_full["START_DATE"]).dt.days
loan_full["tenure_band"] = pd.cut(loan_full["tenure_days"], bins=[-1,180,365,730,1825,10000],labels=["<6m","6-12m","1-2y","2-5y","5y+"])
print(loan_full)

# ---- Cell 19 ------------------------------------------------------------
# 3c. Default & late rate by loan type
by_loan_type = loan_full.groupby("LOAN_TYPE").agg(
    loans=("LOAN_ID","count"), default_rate_pct=("is_default","mean"), late_rate_pct=("ever_late","mean")
).reset_index()
by_loan_type[["default_rate_pct","late_rate_pct"]] = (by_loan_type[["default_rate_pct","late_rate_pct"]] * 100).round(2)
print(by_loan_type.sort_values("default_rate_pct", ascending=False))

by_loan_type.set_index("LOAN_TYPE")["default_rate_pct"].plot.bar(title="Default rate % by loan type", color="steelblue")
plt.show()

# ---- Cell 20 ------------------------------------------------------------
# 3d. Default & late rate by branch
by_branch = loan_full.groupby("BRANCH_ID").agg(
    loans=("LOAN_ID","count"), default_rate_pct=("is_default","mean"), late_rate_pct=("ever_late","mean")
).reset_index()
by_branch[["default_rate_pct","late_rate_pct"]] = (by_branch[["default_rate_pct","late_rate_pct"]] * 100).round(2)
print(by_branch.sort_values("default_rate_pct", ascending=False).head(10))

# ---- Cell 21 ------------------------------------------------------------
# 3e. Default & late rate by tenure band
by_tenure = loan_full.groupby("tenure_band").agg(
    loans=("LOAN_ID","count"), default_rate_pct=("is_default","mean"), late_rate_pct=("ever_late","mean")
).reset_index()
by_tenure[["default_rate_pct","late_rate_pct"]] = (by_tenure[["default_rate_pct","late_rate_pct"]] * 100).round(2)
print(by_tenure.sort_values("default_rate_pct", ascending=False))

by_tenure.set_index("tenure_band")["default_rate_pct"].plot.bar(title="Default rate % by tenure band", color="darkred")
plt.show()

# ---- Cell 22 ------------------------------------------------------------
# 3f. Insight
worst_loan_type = by_loan_type.sort_values("default_rate_pct", ascending=False).iloc[0]
print(f"Insight: '{worst_loan_type['LOAN_TYPE']}' loans have the worst default rate "
      f"({worst_loan_type['default_rate_pct']}%) and a {worst_loan_type['late_rate_pct']}% late-payment rate.")

# ============================================================================
# 4.Support-ticket volume, resolution time by issue type
#
# Question: Which issue types generate the most tickets, and how long do they take to resolve?
# ============================================================================

# ---- Cell 23 ------------------------------------------------------------
print(support_ticket["STATUS"].unique())

# ---- Cell 24 ------------------------------------------------------------
# 4a. Compute resolution days + group by issue type
support_ticket["resolution_days"] = (support_ticket["DATE_RESOLVED"] - support_ticket["DATE_OPENED"]).dt.days
ticket_summary = support_ticket.groupby("ISSUE_TYPE").agg(
    tickets=("TICKET_ID","count"), avg_resolution_days=("resolution_days","mean")
).reset_index().sort_values("tickets", ascending=False)
print(ticket_summary)

# ---- Cell 25 ------------------------------------------------------------
# 4b. Plot — ticket volume by issue type
ticket_summary.set_index("ISSUE_TYPE")["tickets"].plot.bar(title="Ticket volume by issue type", color="teal")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Cell 26 ------------------------------------------------------------
# 4c. Plot — avg resolution days by issue type
ticket_summary.set_index("ISSUE_TYPE")["avg_resolution_days"].plot.bar(title="Avg resolution days by issue type", color="slateblue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Cell 27 ------------------------------------------------------------
# 4d. Insight
top_ticket = ticket_summary.iloc[0]
slowest_ticket = ticket_summary.sort_values("avg_resolution_days", ascending=False).iloc[0]
print(f"Insight: '{top_ticket['ISSUE_TYPE']}' generates the most tickets ({top_ticket['tickets']}). "
      f"'{slowest_ticket['ISSUE_TYPE']}' takes the longest to resolve ({slowest_ticket['avg_resolution_days']:.1f} days).")

# ============================================================================
# 5. Card spend & fraud incidence by merchant category
#
# Question: Where is spend concentrated, and does high-spend category = high-fraud category?
# ============================================================================

# ---- Cell 28 ------------------------------------------------------------
# group by merchant category:total spend, transaction count, fraud rate
spend_fraud=card_txn.groupby("MERCHANT_CATEGORY").agg(
        total_spend=('AMOUNT', 'sum'),
        total_txn=('CARD_TXN_ID', 'count'),
        fraud_rate_pct=('IS_FRAUD', lambda x: round((x == 1).mean() * 100, 2))
    ).reset_index()
spend_fraud["fraud_rate_pct"] = (spend_fraud["fraud_rate_pct"] * 100).round(2)
spend_fraud = spend_fraud.sort_values("total_spend", ascending=False)
print(spend_fraud)
# visualization
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(spend_fraud["MERCHANT_CATEGORY"], spend_fraud["total_spend"], color="steelblue")
ax1.set_ylabel("Total spend"); plt.xticks(rotation=45)
ax2 = ax1.twinx() ## combine two charts in single plot
ax2.plot(spend_fraud["MERCHANT_CATEGORY"], spend_fraud["fraud_rate_pct"], color="red", marker="o")
ax2.set_ylabel("Fraud rate %")
plt.title("Spend vs Fraud rate by merchant category"); 
plt.tight_layout(); 
plt.show()

# 5c. Insight
top_spend = spend_fraud.iloc[0]
top_fraud = spend_fraud.sort_values("fraud_rate_pct", ascending=False).iloc[0]
print(f"Insight: '{top_spend['MERCHANT_CATEGORY']}' drives the most spend (₹{top_spend['total_spend']:,.0f}). "
      f"'{top_fraud['MERCHANT_CATEGORY']}' has the highest fraud rate ({top_fraud['fraud_rate_pct']}%).")
  

# ============================================================================
# 6.Customer segmentation — income & credit-score bands
#
# Question: How does the customer base split by income and credit-score band, and what's the average account balance in each?
# ============================================================================

# ---- Cell 29 ------------------------------------------------------------
# bucket customers into income_band (quartiles) and credit_band(bureau_range)
customers['income_band'] = pd.qcut(customers['ANNUAL_INCOME'], 4, labels=['Q1','Q2','Q3','Q4'])
customers['credit_band'] = pd.cut(customers['CREDIT_SCORE'], bins=[0,600,700,800,900], labels=['<600','600-700','700-800','800+'])

# merge account balances to customer
cust_acc = accounts.groupby('CUSTOMER_ID')['BALANCE'].mean().reset_index()
customers_with_balance = customers.merge(cust_acc, on='CUSTOMER_ID', how='left')

# group by income_band & credit_band: count, avg balance
segmentation = customers_with_balance.groupby(['income_band','credit_band']).agg(
    customers=('CUSTOMER_ID','count'),
    avg_balance=('BALANCE','mean')
).reset_index().sort_values(['income_band','credit_band'])

print(segmentation)

# visualization
plt.figure(figsize=(8,5))
pivot = segmentation.pivot(index="income_band",columns="credit_band",values="avg_balance")
sns.heatmap(pivot,annot=True,fmt=".0f",cmap="Blues")
plt.title("Avg Account Balance by Income & Credit Score Band")
plt.ylabel("Income Band"); plt.xlabel("Credit Score Band")
plt.tight_layout()
plt.show()

# insights
top_seg = segmentation.sort_values("avg_balance", ascending=False).iloc[0]
print(f"\n - Highest avg balance ({top_seg['avg_balance']:.0f}) comes from income band '{top_seg['income_band']}' and credit band '{top_seg['credit_band']}'.")

# ============================================================================
# 7.Card type mix — utilization by card type
#
# Question: Which card type carries the highest utilization (spend ÷ credit limit)?
#
# ============================================================================

# ---- Cell 30 ------------------------------------------------------------
# total spend per card, joined to credit_limit, to compute utilization %
card_spend = card_txn.groupby("CARD_ID")["AMOUNT"].sum().rename("total_spend").reset_index()
card_util = cards.merge(card_spend, on="CARD_ID", how="left")
card_util["total_spend"] = card_util["total_spend"].fillna(0)
card_util["utilization_pct"] = np.where(card_util["CREDIT_LIMIT"] > 0,
         round(card_util["total_spend"] / card_util["CREDIT_LIMIT"] * 100, 2), np.nan)
 
# -> average utilization % by card_type
util_by_type = card_util.groupby("CARD_TYPE")["utilization_pct"].mean().round(2).reset_index().sort_values("utilization_pct", ascending=False)
print(util_by_type)
 
plt.figure(figsize=(7, 4))
plt.bar(util_by_type["CARD_TYPE"], util_by_type["utilization_pct"], color="darkcyan")
plt.title("Avg utilization % by card type"); plt.ylabel("Utilization %"); plt.tight_layout(); plt.show()
 
# Insight
top_util = util_by_type.iloc[0]
print(f"Insight: '{top_util['CARD_TYPE']}' cards run the highest average utilization "
      f"({top_util['utilization_pct']}%) — candidates for credit-limit review or targeted offers.")

# ============================================================================
# 8.Monthly transaction volume & value trend
#
# Question: How is overall transaction volume/value trending month over month?
# ============================================================================

# ---- Cell 31 ------------------------------------------------------------
# -> truncate txn_date to month, aggregate count and total amount
transaction_list['TXN_DATE'] = pd.to_datetime(transaction_list['TXN_DATE'], errors='coerce')
transaction_list["txn_month"] = transaction_list["TXN_DATE"].dt.to_period("M").astype(str)
monthly_trend = transaction_list.groupby("txn_month").agg(
        txn_count=("TRANSACTION_ID", "count"), total_amount=("AMOUNT", "sum")
    ).reset_index()
print(monthly_trend)
 
plt.figure(figsize=(20, 8))
plt.plot(monthly_trend["txn_month"], monthly_trend["total_amount"], marker="o")
plt.xticks(rotation=45); 
plt.title("Monthly transaction value trend"); 
plt.tight_layout(); 
plt.show()
    

# ============================================================================
# ### DIAGNOSTIC ANALYSIS — "Why did it happen?"
# ============================================================================

# ============================================================================
# 1.Do lower credit-score bands show materially higher default & late-payment rates?
#
#
# ============================================================================

# ---- Cell 32 ------------------------------------------------------------
# join loan full to customers credit band then compare default/late payment rates?
loan_cust=loan_full.merge(customers[['CUSTOMER_ID','credit_band']],on='CUSTOMER_ID',how='left')
loan1=loan_cust.groupby('credit_band',observed=True).agg(
    loans=('LOAN_ID','count'),
    defaults_rate_pct=('is_default',lambda x: round(x.mean()*100,2)),
    late_rate_pct=('ever_late',lambda x:round(x.mean()*100,2))
).reset_index()
print(loan1)

plt.figure(figsize=(8,5))
sns.barplot(x='credit_band',y='defaults_rate_pct',data=loan1)
plt.title("Default Rate by Credit Score Band")
plt.ylabel("Default Rate (%)"); plt.xlabel("Credit Score Band")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='credit_band',y='late_rate_pct',data=loan1)
plt.title("Late Payment Rate by Credit Score Band")
plt.ylabel("Late Rate (%)"); plt.xlabel("Credit Score Band")
plt.tight_layout()
plt.show()

# insights
top_def = loan1.sort_values("defaults_rate_pct", ascending=False).iloc[0]
print(f"\n - Highest default rate ({top_def['defaults_rate_pct']}%) comes from credit band '{top_def['credit_band']}'.")
top_late = loan1.sort_values("late_rate_pct", ascending=False).iloc[0]
print(f" - Highest late payment rate ({top_late['late_rate_pct']}%) comes from credit band '{top_late['credit_band']}'.")

# ============================================================================
# 2. Do certain merchant categories or channels show disproportionately higher fraud rates?
# ============================================================================

# ---- Cell 33 ------------------------------------------------------------
# Analyze fraud rates by merchant category and transaction channel
# Group by merchant category and channel to identify disproportionate fraud patterns

overall_fraud_rate=card_txn['IS_FRAUD'].mean()*100 ##baseline fraud rate across all transactions, as a percentage.
if "MERCHANT_CATEGORY" in card_txn.columns:
    cat_fraud=fraud_by_category.copy()
    cat_fraud['disproportionate']=cat_fraud['fraud_rate_pct']-overall_fraud_rate  ##calculates the actual gap in percentage points
    cat_fraud=cat_fraud.sort_values('disproportionate',ascending=False)
    print(cat_fraud)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x='MERCHANT_CATEGORY',y='disproportionate',data=cat_fraud)
    plt.title("Disproportionate Fraud Rates by Merchant Category")
    plt.ylabel("Disproportionate Fraud Rate (%)"); plt.xlabel("Merchant Category")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.show()
    
    # Note: Card transaction data does not include CHANNEL information
    # Channel analysis would require transaction_list data instead




# ============================================================================
# 3. Does prior 'Fraud Report' ticket history correlate with subsequent confirmed fraud?
# ============================================================================

# ---- Cell 34 ------------------------------------------------------------
# Join support tickets (fraud reports) to card transactions to see if customers who reported fraud later had confirmed fraud
# Step 1: Identify customers who filed 'Fraud Report' tickets
fraud_reporters = support_ticket[support_ticket['ISSUE_TYPE'] == 'Fraud Report'][['CUSTOMER_ID', 'DATE_OPENED']].copy()
fraud_reporters = fraud_reporters.rename(columns={'DATE_OPENED': 'fraud_report_date'})

# Step 2: Join to card transactions via cards table to get customer_id
card_cust = cards[['CARD_ID', 'CUSTOMER_ID']]
card_txn_cust = card_txn.merge(card_cust, on='CARD_ID', how='left')

# Step 3: Join fraud reporters to transactions
txn_with_reports = card_txn_cust.merge(fraud_reporters, on='CUSTOMER_ID', how='left')

# Step 4: Flag whether customer had prior fraud report before the transaction
txn_with_reports['had_prior_fraud_report'] = (~txn_with_reports['fraud_report_date'].isna()) & \
(txn_with_reports['TXN_DATE'] >= txn_with_reports['fraud_report_date'])

# Step 5: Compare fraud rates for customers with vs without prior fraud reports
fraud_correlation = txn_with_reports.groupby('had_prior_fraud_report').agg(
    transactions=('CARD_TXN_ID', 'count'),
    fraud_count=('IS_FRAUD', 'sum'),
    fraud_rate_pct=('IS_FRAUD', lambda x: round(x.mean() * 100, 2))
).reset_index()

print("\nFraud rate by prior fraud report history:")
print(fraud_correlation)

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(['No Prior Report', 'Had Prior Report'], 
        fraud_correlation['fraud_rate_pct'], 
        color=['steelblue', 'coral'])
plt.title('Fraud Rate: Customers With vs Without Prior Fraud Report')
plt.ylabel('Fraud Rate (%)')
plt.xlabel('Customer History')
plt.tight_layout()
plt.show()

# Insight
if len(fraud_correlation) == 2:
    rate_with = fraud_correlation[fraud_correlation['had_prior_fraud_report'] == True]['fraud_rate_pct'].values[0]
    rate_without = fraud_correlation[fraud_correlation['had_prior_fraud_report'] == False]['fraud_rate_pct'].values[0]
    lift = rate_with - rate_without
    print(f"\n - Customers with prior fraud reports show a {rate_with}% fraud rate vs {rate_without}% for those without.")
    print(f" - Lift: {lift:.2f} percentage points {'higher' if lift > 0 else 'lower'} fraud rate for prior reporters.")
    if lift > 0:
        print(f" - This suggests prior fraud reporting is a strong predictor of subsequent confirmed fraud — prioritize monitoring for these customers.")
else:
    print("\n - Insufficient data to compare fraud rates between groups.")

# ============================================================================
# 4.Do specific branches show default rates above network average after controlling for loan mix?
#
# --"which branch has the highest default rate"
# ============================================================================

# ---- Cell 35 ------------------------------------------------------------
# Step 1: Compute network-wide default rate per loan_type (the "expected" rate)
network_avg = loan_full.groupby('LOAN_TYPE')['is_default'].mean().reset_index()
network_avg = network_avg.rename(columns={'is_default': 'network_default_rate'})
##Groups every loan in the entire dataset by its LOAN_TYPE, and averages is_default (which is 0/1) within each group — giving you the overall default rate for each product across the whole bank, respectively This becomes your baseline/benchmark(the reference point you're comparing everyone against — the "normal" or "expected" level of performance that you measure each branch relative to.).

# Step 2: Join network average to loan_full(attach benchmark to each individual loan)
loan_with_expected = loan_full.merge(network_avg, on='LOAN_TYPE', how='left')

# Step 3: Group by branch - compute actual default rate and expected default rate (weighted by loan mix) (branch level actual vs expected)
branch_analysis = loan_with_expected.groupby('BRANCH_ID').agg(
    loans=('LOAN_ID', 'count'),
    actual_default_rate=('is_default', 'mean'),  ## how manny loans it has, and its default rate
    expected_default_rate=('network_default_rate', 'mean')  # Weighted by the branch's loan mix
).reset_index()

# ATTACH BRANCH NAMES (so that output is easier to read)
branch_analysis = branch_analysis.merge(branches[['BRANCH_ID', 'BRANCH_NAME']], on='BRANCH_ID')

# Step 4: Compute the difference (excess default rate)
branch_analysis['excess_default_rate'] = branch_analysis['actual_default_rate'] - branch_analysis['expected_default_rate']
branch_analysis['excess_default_pct'] = round(branch_analysis['excess_default_rate'] * 100, 2)
branch_analysis['actual_default_pct'] = round(branch_analysis['actual_default_rate'] * 100, 2)
branch_analysis['expected_default_pct'] = round(branch_analysis['expected_default_rate'] * 100, 2)

# Sort by excess default rate
branch_analysis = branch_analysis.sort_values('excess_default_rate', ascending=False)

print("\nBranch Default Rates (Actual vs Expected after controlling for loan mix):")
print(branch_analysis[['BRANCH_ID',"BRANCH_NAME", 'loans', 'actual_default_pct', 'expected_default_pct', 'excess_default_pct']])

# Show only the top 15 branches by excess rate (both worst offenders and biggest underperformers)
top_n=branch_analysis.reindex(
    branch_analysis['excess_default_pct'].abs().sort_values(ascending=False).index).head(15).sort_values('excess_default_rate')

# Visualization
plt.figure(figsize=(12, 6))
plt.barh(top_n['BRANCH_NAME'], top_n['excess_default_pct'], color=['red' if x > 0 else 'blue' for x in top_n['excess_default_pct']])
plt.xlabel('Excess Default Rate (%)')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.title('Top 15 Branches by |Excess Default Rate|\n(Red = worse than expected, Blue = better than expected)')
plt.ylabel('BRANCH NAME')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Insights
worst_branches = branch_analysis[branch_analysis['excess_default_rate'] > 0]
if len(worst_branches) > 0:
    print(f"\n - {len(worst_branches)} branches show default rates ABOVE network average after controlling for loan mix.")
    top_worst = worst_branches.iloc[0]
    print(f" - Worst performer: Branch {top_worst['BRANCH_NAME']} with {top_worst['actual_default_pct']}% actual default rate ")
    print(f"   vs {top_worst['expected_default_pct']}% expected (excess: +{top_worst['excess_default_pct']}%)")
    print(f" - These branches may have operational, underwriting, or servicing issues beyond their loan-type mix.")


# ============================================================================
# 5. Do escalated/negative-outcome tickets correlate with account dormancy or closure?
# ============================================================================

# ---- Cell 36 ------------------------------------------------------------
# flag(label) customers with any unresolved/escalated tickets
escalated_status=[x for x in support_ticket['STATUS'].unique() if x not in ['closed', 'resolved']]
escalated_customers=set(support_ticket[support_ticket['STATUS'].isin(escalated_status)]['CUSTOMER_ID'].unique())

# compare account status mix:escalated - ticket customers  vs everyone else
account_cust=accounts.merge(customers[['CUSTOMER_ID']], on='CUSTOMER_ID')
account_cust['had_escalated_ticket']=account_cust['CUSTOMER_ID'].isin(escalated_customers)

account_mix=pd.crosstab(account_cust['STATUS'], account_cust['had_escalated_ticket'],normalize='index')*100
print(account_mix.round(2))

#visualization
plt.figure(figsize=(10,5))
account_mix.plot(kind='bar',stacked=True,colormap='coolwarm')
plt.title('Account Status Mix: Customers with Escalated Tickets vs Everyone Else')
plt.xlabel('Account Status')
plt.ylabel('Number of Accounts')
plt.legend(title='Customer Group', loc='upper right')
plt.tight_layout()
plt.show()

# insights 
# Need a column-normalized version: "of all True/False customers, what % are Dormant/Closed?"
account_mix_col = pd.crosstab(account_cust['STATUS'], account_cust['had_escalated_ticket'], normalize='columns') * 100

# identify risk status(es) (dormant/closed) and compare their prevalence among customers with and without escalated tickets
print('\nInsights:')
print(f" - {account_mix_col.loc['Dormant', True]:.1f}% of customers with escalated tickets are Dormant, "
      f"vs {account_mix_col.loc['Dormant', False]:.1f}% of other customers.")
print(f" - {account_mix_col.loc['Closed', True]:.1f}% of customers with escalated tickets are Closed, "
      f"vs {account_mix_col.loc['Closed', False]:.1f}% of other customers.")
print(f" - {account_mix_col.loc['Active', True]:.1f}% of customers with escalated tickets are Active, "
      f"vs {account_mix_col.loc['Active', False]:.1f}% of other customers.")

# .strip() removes hidden whitespace, .lower() handles casing — defensive against uncleaned raw data
risk_status = [x for x in account_mix_col.index if str(x).strip().lower() in ('dormant', 'closed')]

if risk_status and True in account_mix_col.columns:
    with_ticket_risk_pct = account_mix_col.loc[risk_status, True].sum()
    without_ticket_risk_pct = account_mix_col.loc[risk_status, False].sum()
    print(
        f"Insight: {with_ticket_risk_pct:.1f}% of accounts held by escalated-ticket customers are "
        f"{'/'.join(risk_status)}, vs {without_ticket_risk_pct:.1f}% for other customers — "
        + ("escalated tickets ARE linked to higher dormancy/closure."
           if with_ticket_risk_pct > without_ticket_risk_pct
           else "no meaningful link between escalated tickets and dormancy/closure.")
    )
else:
    print(f"Insight: could not compute — matched risk_status: {risk_status}, "
          f"columns available: {list(account_mix_col.columns)}")

# ============================================================================
# 6. Does high card utilization correlate with fraud on that card?
# ============================================================================

# ---- Cell 37 ------------------------------------------------------------
# flag(label) cards with any fraud detection,bucket utilization into bands,compare fraud rate 
card_util['fraud_on_card']=card_util['CARD_ID'].isin(card_txn.loc[card_txn['IS_FRAUD']==1,'CARD_ID']).astype(int)
card_util['util_band']=pd.cut(card_util['utilization_pct'],bins=[0,25,50,75,300],labels=['0-25%','25-50%','50-75%','75%+'])
card_util_fraud=card_util.groupby('util_band',observed=True)['fraud_on_card'].mean().mul(100).round(2)
print(card_util_fraud)

#visualization
plt.figure(figsize=(10,5))
card_util_fraud.plot(kind='bar',color='orange')
plt.title('Card Fraud Rate by Utilization Band')
plt.xlabel('Utilization Band')
plt.ylabel('Fraud Rate')
plt.tight_layout()
plt.show()

# insights 
top_util_band=card_util_fraud.idxmax()
print(f'The card utilization band with the highest fraud rate is {top_util_band} with a fraud rate of {card_util_fraud[top_util_band]}%')

# ============================================================================
# 7. Does account tenure correlate with dormancy/closure?
# ============================================================================

# ---- Cell 38 ------------------------------------------------------------
# acccount age  (from open date) into bands,compare status mix per band
accounts['account_age_days']=(today-pd.to_datetime(accounts['OPEN_DATE'])).dt.days
accounts['account_age_band']=pd.cut(accounts['account_age_days'],bins=[0,365,730,3650],labels=['0-1yr','1-2yr','2yr+'])

acc_age=pd.crosstab(accounts['account_age_band'],accounts['STATUS'],normalize='index')*100
print(acc_age.round(2))

# visualization
plt.figure(figsize=(10,5))
acc_age.plot(kind='bar',stacked=True,colormap='coolwarm')
plt.title('Account Status Mix by Age Band')
plt.xlabel('Account Age Band')
plt.ylabel('Status Mix')
plt.tight_layout()
plt.show()

# insights 
top_age_band=acc_age.idxmax(axis=1)
print(f'insight: The account age band with the highest percentage of {top_age_band} accounts is {top_age_band} with a percentage of {acc_age[top_age_band]}%')


# ============================================================================
# > PREDICTIVE ANALYSIS — "**What is likely to happen?"**
# ============================================================================

# ---- Cell 39 ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score,roc_curve


# ============================================================================
# 1. Fraud-detection model
#
#  Business Statement: Fraudulent card transactions cause direct financial loss and erode customer trust. The bank needs to flag suspicious transactions in near real-time, before settlement, rather than discovering fraud after the fact.
#
#  Goal- Classify each card transaction as fraud or not fraud from amount, merchant category, card type and and recency
# ============================================================================

# ---- Cell 40 ------------------------------------------------------------
# Prepare features
# Create days_since_txn feature (recency)
card_txn['days_since_txn'] = (today - card_txn['TXN_DATE']).dt.days

feature_cols=['AMOUNT','CARD_TYPE','MERCHANT_CATEGORY','days_since_txn']
model_df=card_txn[feature_cols+['IS_FRAUD']].dropna()

# Encode text columns to numbers
le=LabelEncoder()
model_df['CARD_TYPE']=le.fit_transform(model_df['CARD_TYPE'])
model_df['MERCHANT_CATEGORY']=le.fit_transform(model_df['MERCHANT_CATEGORY'])

# Split into train and test sets
X=model_df.drop(columns='IS_FRAUD')
y=model_df['IS_FRAUD']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Scale features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print('\n logistic regression classifier and evaluation')
# Train logistic regression model
lr=LogisticRegression()
lr.fit(X_train_scaled,y_train)

# Evaluate model performance
y_pred_lr=lr.predict(X_test_scaled)
y_prob_lr=lr.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test,y_pred_lr))
print(confusion_matrix(y_test,y_pred_lr))
roc_auc_lr = roc_auc_score(y_test,y_pred_lr)
print(f'roc_auc_lr: {roc_auc_lr:.4f}')

print('\n random forest classifier and evaluation')

# Train random forest model
rf = RandomForestClassifier(
    n_estimators=100, max_depth=5, class_weight='balanced', random_state=42
)
rf.fit(X_train, y_train)
 
# Evaluate model performance
y_pred_rf = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test, y_prob)
print(f'roc_auc_rf: {roc_auc_rf:.4f}')

# ==================== PERFORMANCE METRICS SUMMARY TABLE ====================
print("MODEL PERFORMANCE SUMMARY")


metrics_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'ROC-AUC': [roc_auc_lr, roc_auc_rf],
    'Fraud Recall': [
        recall_score(y_test, y_pred_lr, pos_label=1),
        recall_score(y_test, y_pred_rf, pos_label=1),
    ],
    'Fraud Precision': [
        precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0),
        precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0),
    ],
})
##zero_division=0 replaces the manual if (...) > 0 else 0

metrics_summary = metrics_summary.round(4)


metrics_summary['Fraud Recall'] = metrics_summary['Fraud Recall'].round(4)
metrics_summary['Fraud Precision'] = metrics_summary['Fraud Precision'].round(4)
metrics_summary['ROC-AUC'] = metrics_summary['ROC-AUC'].round(4)

print(metrics_summary.to_string(index=False))

# ==================== GRAPH 1: ROC Curve + Metrics Comparison ====================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)      
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob)

#fpr-false positive rate
#tpr-true positive rate
#lr=logistic regression
#rf=random forest

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: ROC curve
axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_lr:.4f})')
axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_rf:.4f})')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison'); axes[0].legend()

# Right: Metrics bar chart (ROC-AUC, Recall, Precision side by side)
metrics_summary.set_index('Model')[['ROC-AUC', 'Fraud Recall', 'Fraud Precision']].plot(
    kind='bar', ax=axes[1], color=['steelblue', 'orange', 'green']
)
axes[1].set_title('Model Performance Summary'); axes[1].set_ylabel('Score'); axes[1].set_ylim(0, 1)
axes[1].legend(loc='lower right'); axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout(); plt.show()

# Insight
scores = {"Logistic Regression": roc_auc_lr, "Random Forest": roc_auc_rf}
best_model = max(scores, key=scores.get)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top_feature = feature_importance.index[0]
print(f"\n Insight: {best_model} performs best on fraud detection (ROC-AUC {scores[best_model]}) "
      f"\n among the three models compared. '{top_feature}' is the strongest predictor of fraud in the "
      f"\n Random Forest, contributing {feature_importance.iloc[0]*100:.1f}% of its decision weight.")
 

# ============================================================================
# 2. Credit-default model
#
# Business Statement: Loans that end in Default/Write-off directly hit the bank's P&L through provisioning and bad-debt losses. The bank needs to identify high-risk borrowers at or before loan approval/renewal.
#
# Goal: predict loan Default/Write-off using credit_score, income ,loan_amount, interest_rate, tenure and early late-payment behaviour.
# ============================================================================

# ---- Cell 41 ------------------------------------------------------------
# build feature set from loan + customer info 
default_df=loan_full.merge(customers[['CUSTOMER_ID','CREDIT_SCORE','ANNUAL_INCOME',]],on ='CUSTOMER_ID')
default_df['tenure_days']=default_df['tenure_days'].fillna(0)

feature_columns=['ANNUAL_INCOME','CREDIT_SCORE','tenure_days','LOAN_AMOUNT','INTEREST_RATE','ever_late']
model_df2=default_df[feature_columns+['is_default']].dropna()

# split into train and test sets
X2=model_df2.drop(columns='is_default')
y2=model_df2['is_default']
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=42,stratify=y2)

# scale features
scaler=StandardScaler()
X_train_scaled2=scaler.fit_transform(X2_train)
X_test_scaled2=scaler.transform(X2_test)

print('\n logistic regression classifier and evaluation')
# Train logistic regression model
lr=LogisticRegression()
lr.fit(X_train_scaled2,y2_train)

# Evaluate model performance
y_pred_lr2=lr.predict(X_test_scaled2)
y_prob_lr2 = lr.predict_proba(X_test_scaled2)[:, 1]
print(classification_report(y2_test,y_pred_lr2))
print(confusion_matrix(y2_test,y_pred_lr2))
roc_auc_lr2 = roc_auc_score(y2_test,y_pred_lr2)
print(f'roc_auc_lr2: {roc_auc_lr2:.4f}')

print('\n random forest classifier and evaluation')

# Train random forest model
rf=RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X2_train,y2_train)

# Evaluate model performance
y_pred_rf2=rf.predict(X2_test)
y_prob2=rf.predict_proba(X2_test)[:,1]
print(classification_report(y2_test,y_pred_rf2))
print(confusion_matrix(y2_test,y_pred_rf2))
roc_auc_rf2 = roc_auc_score(y2_test,y_prob2)
print(f'roc_auc_rf2: {roc_auc_rf2:.4f}')

# Insight
p2_scores = {"Logistic Regression": roc_auc_lr2, "Random Forest": roc_auc_rf2}
p2_best_model = max(p2_scores, key=p2_scores.get)
p2_importance = pd.Series(rf.feature_importances_, index=X2.columns).sort_values(ascending=False)
print(f"\n Insight: {p2_best_model} performs best here (ROC-AUC {p2_scores[p2_best_model]}) among the three "
      f"\n models compared. '{p2_importance.index[0]}' is the single strongest driver of default risk "
      f"\n according to the Random Forest, contributing {p2_importance.iloc[0]*100:.1f}% of its decision "
      f"\n weight — worth prioritizing in underwriting criteria.")

# ============================================================================
# 3. Next-late-payment prediction
#
# Business Statement: Loans that go into a late-payment cycle are early warning signs of eventual default. Collections needs to know which currently active loans are likely to miss their NEXT installment.
#
# Goal: flag loans likely to miss their next installment based on payment history trends.
# ============================================================================

# ---- Cell 42 ------------------------------------------------------------
# Create payment statistics from loan_payment
pay_stats = loan_payment.groupby("LOAN_ID").agg(
    payments_made=("PAYMENT_ID", "count"),
    late_payments=("LATE_PAYMENT_FLAG", "sum")
).reset_index()
pay_stats["late_ratio"] = pay_stats["late_payments"] / pay_stats["payments_made"]

# Build feature set from loan + payment history + customer
p3_df = loan_full.merge(pay_stats, on="LOAN_ID", how="left")
p3_df["payments_made"] = p3_df["payments_made"].fillna(0)
p3_df["late_payments"] = p3_df["late_payments"].fillna(0)
p3_df["late_ratio"] = p3_df["late_ratio"].fillna(0)
p3_df = p3_df.merge(customers[["CUSTOMER_ID", "CREDIT_SCORE"]], on="CUSTOMER_ID")
 
feat_cols_p3 = ["CREDIT_SCORE", "LOAN_AMOUNT", "INTEREST_RATE", "payments_made", "late_ratio"]
p3_model_df = p3_df[feat_cols_p3 + ["ever_late"]].dropna()

#split into train and test sets
X3=p3_model_df.drop(columns='ever_late')
y3=p3_model_df['ever_late']
X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=0.2,random_state=42,stratify=y3)

# scale features
scaler=StandardScaler()
X_train_scaled3=scaler.fit_transform(X3_train)
X_test_scaled3=scaler.transform(X3_test)

print('\n logistic regression classifier and evaluation')
# Train logistic regression model
lr=LogisticRegression()
lr.fit(X_train_scaled3,y3_train)

# Evaluate model performance
y_pred_lr3=lr.predict(X_test_scaled3)
y_prob_lr3 = lr.predict_proba(X_test_scaled3)[:, 1]
print(classification_report(y3_test,y_pred_lr3))
print(confusion_matrix(y3_test,y_pred_lr3))
roc_auc_lr3 = roc_auc_score(y3_test,y_pred_lr3)
print(f'roc_auc_lr3: {roc_auc_lr3:.4f}')

print('\n random forest classifier and evaluation')

# Train random forest model
rf=RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X3_train,y3_train)

# Evaluate model performance
y_pred_rf3=rf.predict(X3_test)
y_prob3=rf.predict_proba(X3_test)[:,1]
print(classification_report(y3_test,y_pred_rf3))
print(confusion_matrix(y3_test,y_pred_rf3))
roc_auc_rf3 = roc_auc_score(y3_test,y_prob3)
print(f'roc_auc_rf3: {roc_auc_rf3:.4f}')

# Confusion matrix (Random Forest)
p3_cm = confusion_matrix(y3_test, rf.predict(X3_test))
plt.figure(figsize=(5, 4))
sns.heatmap(p3_cm, annot=True, fmt="d", cmap="Greens", xticklabels=["On Time", "Late"], yticklabels=["On Time", "Late"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("P3. Next-Late-Payment — Confusion Matrix (Random Forest)"); plt.tight_layout(); plt.show()

# Score active loans with predicted late-payment risk (using best-performing model: Random Forest)
active_loans = p3_df[p3_df["STATUS"] == "Active"]
active_loans["late_risk_score"] = rf.predict_proba(active_loans[feat_cols_p3].fillna(0))[:, 1]
print("\nTop 10 highest-risk active loans:")
top_risk_loans = active_loans[["LOAN_ID", "CUSTOMER_ID", "late_risk_score"]].sort_values("late_risk_score", ascending=False).head(10)
print(top_risk_loans)

# Insight
p3_scores = {"Logistic Regression": roc_auc_lr3, "Random Forest": roc_auc_rf3}
p3_best_model = max(p3_scores, key=p3_scores.get)
n_high_risk = (active_loans["late_risk_score"] > 0.5).sum()
pct_high_risk = round(n_high_risk / len(active_loans) * 100, 1)
print(f"\n Insight: {p3_best_model} performs best (ROC-AUC {p3_scores[p3_best_model]}) among the three models "
      f"\n compared. {n_high_risk} active loans ({pct_high_risk}% of all active loans) are scored above 50% "
      f"\n risk of an upcoming late payment — these are the loans collections should prioritize for proactive outreach.")


# ============================================================================
#
# # PICKLE FILES — save trained models for reuse without retraining
#
#
# ============================================================================

# ---- Cell 43 ------------------------------------------------------------
import pickle

# ---- Cell 44 ------------------------------------------------------------
import pickle# Pickle a. Save fraud-detection model (P1)

with open("/tmp/fraud_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("Saved: fraud_model.pkl")
 
 
# Pickle b. Save credit-default model (P2)
with open("/tmp/credit_default_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("Saved: credit_default_model.pkl")
 
 
# Pickle c. Save late-payment model (P3)
with open("/tmp/late_payment_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("Saved: late_payment_model.pkl")
 
