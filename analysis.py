import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data/goldman_sachs.csv")

# ==============================
# TASK 1: DATA CLEANING
# ==============================
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d-%m-%Y')
df['AccountType'] = df['AccountType'].str.strip().str.title()
df['TransactionType'] = df['TransactionType'].str.strip().str.title()

cols = ['TransactionAmount', 'AccountBalance', 'RiskScore', 'CreditRating']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

df['YearMonth'] = df['TransactionDate'].dt.to_period('M')
df['Year'] = df['TransactionDate'].dt.year

print("Data Cleaning Completed ✅")

# ==============================
# TASK 2: TRANSACTION ANALYSIS
# ==============================

# Monthly Summary
credit = df[df['TransactionType'] == 'Deposit'].groupby('YearMonth')['TransactionAmount'].sum()
debit = df[df['TransactionType'] == 'Withdrawal'].groupby('YearMonth')['TransactionAmount'].sum()

summary = pd.DataFrame({'Credit': credit, 'Debit': debit}).fillna(0)
summary['Net'] = summary['Credit'] - summary['Debit']

print("\nMonthly Summary:")
print(summary.head())

# Yearly Summary
yearly_credit = df[df['TransactionType'] == 'Deposit'].groupby('Year')['TransactionAmount'].sum()
yearly_debit = df[df['TransactionType'] == 'Withdrawal'].groupby('Year')['TransactionAmount'].sum()

yearly_summary = pd.DataFrame({'Credit': yearly_credit, 'Debit': yearly_debit}).fillna(0)
yearly_summary['Net'] = yearly_summary['Credit'] - yearly_summary['Debit']

print("\nYearly Summary:")
print(yearly_summary)

# Graph 1
summary[['Credit','Debit']].plot(figsize=(10,5))
plt.title("Credit vs Debit Trend")
plt.savefig("outputs/charts/credit_debit.png")
plt.close()

# Top & Bottom Accounts
account_flow = df.groupby('AccountID')['TransactionAmount'].sum()

print("\nTop Accounts:")
print(account_flow.nlargest(5))

print("\nBottom Accounts:")
print(account_flow.nsmallest(5))

# Dormant Accounts
df = df.sort_values(['AccountID','TransactionDate'])
df['PrevDate'] = df.groupby('AccountID')['TransactionDate'].shift(1)
df['Gap'] = (df['TransactionDate'] - df['PrevDate']).dt.days

dormant = df[df['Gap'] > 60]['AccountID'].unique()
print("\nDormant Accounts:", dormant)

# ==============================
# TASK 3: CUSTOMER PROFILE
# ==============================

txn_count = df.groupby('CustomerID').size()

def segment(x):
    if x > 50:
        return "High"
    elif x > 20:
        return "Medium"
    else:
        return "Low"

segments = txn_count.apply(segment)

print("\nCustomer Segments:")
print(segments.value_counts())

# Graph 2
segments.value_counts().plot(kind='bar')
plt.title("Customer Segmentation")
plt.savefig("outputs/charts/customer_segments.png")
plt.close()

# Profiles
account_net = df.groupby('AccountID')['TransactionAmount'].sum()

print("\nHigh Net Inflow Accounts:")
print(account_net.nlargest(5))

low_balance_accounts = df[df['AccountBalance'] < 1000]

print("\nHigh Frequency Low Balance Accounts:")
print(low_balance_accounts['AccountID'].value_counts().head(5))

print("\nNegative / Near Zero Balance Accounts:")
print(df[df['AccountBalance'] <= 0][['AccountID','AccountBalance']].head())

# ==============================
# TASK 4: RISK ANALYSIS
# ==============================

# Volatility
volatility = df.groupby('AccountID')['AccountBalance'].std()
print("\nTop Volatile Accounts:")
print(volatility.nlargest(5))

# Large Withdrawals
large_withdrawals = df[(df['TransactionType'] == 'Withdrawal') &
                       (df['TransactionAmount'] > df['TransactionAmount'].mean())]

print("\nFrequent Large Withdrawals:")
print(large_withdrawals[['AccountID','TransactionAmount']].head())

# IQR Anomaly Detection
Q1 = df['TransactionAmount'].quantile(0.25)
Q3 = df['TransactionAmount'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

iqr_anomalies = df[(df['TransactionAmount'] < lower) | 
                   (df['TransactionAmount'] > upper)]

print("\nIQR Anomalies Detected:", len(iqr_anomalies))

# ==============================
# TASK 6: HYPOTHESIS TESTING
# ==============================

high = df[df['CustomerID'].isin(txn_count[txn_count > 50].index)]
low = df[df['CustomerID'].isin(txn_count[txn_count <= 20].index)]

print("\nHypothesis Testing:")
print("H0: High activity customers do NOT have higher balance")
print("H1: High activity customers HAVE higher balance")

if high['AccountBalance'].mean() > low['AccountBalance'].mean():
    print("Result: Reject H0 ✅")
else:
    print("Result: Fail to Reject H0 ❌")

# ==============================
# END
# ==============================

print("\nProject Completed Successfully ✅")

