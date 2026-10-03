# ============================================================
# LOAN DEFAULT & CREDIT RISK ANALYSIS
# Complete Beginner-Friendly Python Project
# ============================================================

# If needed, install libraries once:
# pip install pandas numpy matplotlib seaborn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. SETTINGS
# -----------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
sns.set_theme(style="whitegrid")

DATA_FILE = "Loan_Credit_Risk_Raw.csv"
CLEAN_FILE = "Loan_Credit_Risk_Cleaned.csv"
CHART_FOLDER = "Loan_Credit_Risk_Charts"
os.makedirs(CHART_FOLDER, exist_ok=True)

# -----------------------------
# 2. DATA LOADING
# -----------------------------
df = pd.read_csv(DATA_FILE)

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== LAST 5 RECORDS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# -----------------------------
# 3. DATA EXPLORATION & CLEANING
# -----------------------------
print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").T)

print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates().copy()

# Fill numeric missing values with median
numeric_fill_cols = ["Annual_Income", "Credit_Score", "Employment_Years"]
for col in numeric_fill_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_fill_cols = ["Property_Ownership"]
for col in categorical_fill_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Clean text columns
text_cols = [
    "Gender", "Education", "Employment_Status", "Marital_Status",
    "Loan_Type", "Credit_History", "Property_Ownership",
    "Region", "Loan_Status", "Default_Status"
]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

print("\n========== UNIQUE VALUES ==========")
for col in text_cols:
    print(f"\n{col}:")
    print(df[col].unique())

# Check numerical outliers using IQR (report only; do not automatically delete)
outlier_cols = [
    "Age", "Annual_Income", "Credit_Score", "Loan_Amount",
    "Interest_Rate", "Monthly_Installment",
    "Debt_to_Income_Ratio", "Employment_Years"
]

print("\n========== IQR OUTLIER CHECK ==========")
for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {count} possible outliers")

# -----------------------------
# 4. FEATURE ENGINEERING
# -----------------------------
df["Default_Flag"] = np.where(df["Default_Status"] == "Defaulted", 1, 0)

df["Credit_Score_Group"] = pd.cut(
    df["Credit_Score"],
    bins=[0, 579, 669, 739, 799, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    include_lowest=True
)

df["Income_Group"] = pd.cut(
    df["Annual_Income"],
    bins=[0, 400000, 700000, 1000000, np.inf],
    labels=["Low Income", "Lower-Middle Income", "Upper-Middle Income", "High Income"],
    include_lowest=True
)

df["DTI_Group"] = pd.cut(
    df["Debt_to_Income_Ratio"],
    bins=[0, 20, 35, 50, np.inf],
    labels=["Low DTI", "Moderate DTI", "High DTI", "Very High DTI"],
    include_lowest=True
)

# -----------------------------
# 5. VERIFY & SAVE CLEANED DATA
# -----------------------------
print("\n========== CLEANED DATA CHECK ==========")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
print(df.head())

df.to_csv(CLEAN_FILE, index=False)
print(f"\nCleaned dataset saved as: {CLEAN_FILE}")

# -----------------------------
# 6. DATA ANALYSIS
# -----------------------------
total_customers = df["Customer_ID"].nunique()
total_loans = df["Loan_ID"].nunique()
defaulted_loans = (df["Default_Status"] == "Defaulted").sum()
non_defaulted_loans = (df["Default_Status"] == "Non-Defaulted").sum()
default_rate = defaulted_loans / total_loans * 100

print("\n========== KEY METRICS ==========")
print("Total Customers:", total_customers)
print("Total Loans:", total_loans)
print("Defaulted Loans:", defaulted_loans)
print("Non-Defaulted Loans:", non_defaulted_loans)
print(f"Overall Default Rate: {default_rate:.2f}%")
print(f"Average Annual Income: {df['Annual_Income'].mean():,.2f}")
print(f"Average Credit Score: {df['Credit_Score'].mean():.2f}")
print(f"Average Loan Amount: {df['Loan_Amount'].mean():,.2f}")
print(f"Average Interest Rate: {df['Interest_Rate'].mean():.2f}%")
print(f"Average Debt-to-Income Ratio: {df['Debt_to_Income_Ratio'].mean():.2f}%")

print("\n========== LOANS BY LOAN TYPE ==========")
print(df["Loan_Type"].value_counts())

print("\n========== LOANS BY REGION ==========")
print(df["Region"].value_counts())

def default_rate_by(column):
    result = (
        df.groupby(column, observed=False)["Default_Flag"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count":"Total_Loans", "sum":"Defaulted_Loans", "mean":"Default_Rate"})
        .reset_index()
    )
    result["Default_Rate"] = result["Default_Rate"] * 100
    return result.sort_values("Default_Rate", ascending=False)

print("\n========== DEFAULT RATE BY LOAN TYPE ==========")
print(default_rate_by("Loan_Type").to_string(index=False))

print("\n========== DEFAULT RATE BY EMPLOYMENT STATUS ==========")
print(default_rate_by("Employment_Status").to_string(index=False))

print("\n========== DEFAULT RATE BY EDUCATION ==========")
print(default_rate_by("Education").to_string(index=False))

print("\n========== DEFAULT RATE BY PROPERTY OWNERSHIP ==========")
print(default_rate_by("Property_Ownership").to_string(index=False))

print("\n========== DEFAULT RATE BY CREDIT SCORE GROUP ==========")
print(default_rate_by("Credit_Score_Group").to_string(index=False))

print("\n========== DEFAULT RATE BY INCOME GROUP ==========")
print(default_rate_by("Income_Group").to_string(index=False))

avg_credit = df["Credit_Score"].mean()
avg_loan = df["Loan_Amount"].mean()

below_avg_credit = df[df["Credit_Score"] < avg_credit]
above_avg_loan = df[df["Loan_Amount"] > avg_loan]
high_dti = df[df["Debt_to_Income_Ratio"] > 50]
top_10_loan = df.nlargest(10, "Loan_Amount")[
    ["Customer_ID", "Loan_ID", "Loan_Type", "Loan_Amount", "Credit_Score", "Default_Status"]
]

print("\nCustomers with below-average credit scores:", len(below_avg_credit))
print("Customers with above-average loan amounts:", len(above_avg_loan))
print("Customers with high DTI (>50%):", len(high_dti))

print("\n========== TOP 10 CUSTOMERS BY LOAN AMOUNT ==========")
print(top_10_loan.to_string(index=False))

loan_type_rates = default_rate_by("Loan_Type")
region_rates = default_rate_by("Region")

print("\nLoan type with highest default rate:")
print(loan_type_rates.iloc[0].to_string())

print("\nRegion with highest default rate:")
print(region_rates.iloc[0].to_string())

print("\n========== CREDIT SCORE VS DEFAULT ==========")
print(df.groupby("Default_Status")["Credit_Score"].agg(["count","mean","median"]).round(2))

print("\n========== INCOME VS DEFAULT ==========")
print(df.groupby("Default_Status")["Annual_Income"].agg(["count","mean","median"]).round(2))

print("\n========== LOAN AMOUNT VS DEFAULT ==========")
print(df.groupby("Default_Status")["Loan_Amount"].agg(["count","mean","median"]).round(2))

print("\n========== PREVIOUS DEFAULTS VS CURRENT DEFAULT ==========")
print(default_rate_by("Previous_Defaults").to_string(index=False))

# Customer segment analysis
segment_analysis = (
    df.groupby(["Credit_Score_Group", "Income_Group"], observed=False)["Default_Flag"]
      .agg(["count","sum","mean"])
      .reset_index()
)
segment_analysis = segment_analysis[segment_analysis["count"] >= 10].copy()
segment_analysis["Default_Rate"] = segment_analysis["mean"] * 100
segment_analysis = segment_analysis.sort_values("Default_Rate", ascending=False)

print("\n========== HIGH-RISK CUSTOMER SEGMENTS ==========")
print(segment_analysis[["Credit_Score_Group","Income_Group","count","sum","Default_Rate"]].head(10).to_string(index=False))

# -----------------------------
# 7. VISUALIZATIONS
# -----------------------------
def save_show(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, filename), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# 1. Loan Default Distribution — Pie Chart
plt.figure(figsize=(7,6))
counts = df["Default_Status"].value_counts()
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Loan Default Distribution")
save_show("01_Loan_Default_Distribution.png")

# 2. Loan Count by Loan Type — Bar Chart
plt.figure(figsize=(9,6))
sns.countplot(data=df, x="Loan_Type", order=df["Loan_Type"].value_counts().index)
plt.title("Loan Count by Loan Type")
plt.xlabel("Loan Type")
plt.ylabel("Number of Loans")
plt.xticks(rotation=20)
save_show("02_Loan_Count_by_Loan_Type.png")

# 3. Default Rate by Loan Type — Bar Chart
temp = default_rate_by("Loan_Type")
plt.figure(figsize=(9,6))
sns.barplot(data=temp, x="Loan_Type", y="Default_Rate")
plt.title("Default Rate by Loan Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=20)
save_show("03_Default_Rate_by_Loan_Type.png")

# 4. Default Rate by Employment Status — Bar Chart
temp = default_rate_by("Employment_Status")
plt.figure(figsize=(9,6))
sns.barplot(data=temp, x="Employment_Status", y="Default_Rate")
plt.title("Default Rate by Employment Status")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=20)
save_show("04_Default_Rate_by_Employment_Status.png")

# 5. Default Rate by Education — Bar Chart
temp = default_rate_by("Education")
plt.figure(figsize=(9,6))
sns.barplot(data=temp, x="Education", y="Default_Rate")
plt.title("Default Rate by Education")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=20)
save_show("05_Default_Rate_by_Education.png")

# 6. Credit Score Distribution — Histogram
plt.figure(figsize=(9,6))
sns.histplot(df["Credit_Score"], bins=25, kde=True)
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
save_show("06_Credit_Score_Distribution.png")

# 7. Loan Amount Distribution — Histogram
plt.figure(figsize=(9,6))
sns.histplot(df["Loan_Amount"], bins=30, kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
save_show("07_Loan_Amount_Distribution.png")

# 8. Annual Income Distribution — Histogram
plt.figure(figsize=(9,6))
sns.histplot(df["Annual_Income"], bins=30, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
save_show("08_Annual_Income_Distribution.png")

# 9. Credit Score vs Loan Amount — Scatter Plot
plt.figure(figsize=(9,6))
sns.scatterplot(data=df, x="Credit_Score", y="Loan_Amount", hue="Default_Status", alpha=0.65)
plt.title("Credit Score vs Loan Amount")
save_show("09_Credit_Score_vs_Loan_Amount.png")

# 10. Income vs Loan Amount — Scatter Plot
plt.figure(figsize=(9,6))
sns.scatterplot(data=df, x="Annual_Income", y="Loan_Amount", hue="Default_Status", alpha=0.65)
plt.title("Income vs Loan Amount")
save_show("10_Income_vs_Loan_Amount.png")

# 11. Loan Amount by Default Status — Box Plot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Default_Status", y="Loan_Amount")
plt.title("Loan Amount by Default Status")
save_show("11_Loan_Amount_by_Default_Status.png")

# 12. Credit Score by Default Status — Box Plot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Default_Status", y="Credit_Score")
plt.title("Credit Score by Default Status")
save_show("12_Credit_Score_by_Default_Status.png")

# 13. Debt-to-Income Ratio by Default Status — Box Plot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Default_Status", y="Debt_to_Income_Ratio")
plt.title("Debt-to-Income Ratio by Default Status")
save_show("13_DTI_by_Default_Status.png")

# 14. Default Rate by Income Group — Bar Chart
temp = default_rate_by("Income_Group")
plt.figure(figsize=(10,6))
sns.barplot(data=temp, x="Income_Group", y="Default_Rate")
plt.title("Default Rate by Income Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=20)
save_show("14_Default_Rate_by_Income_Group.png")

# 15. Correlation Heatmap
corr_cols = [
    "Age", "Dependents", "Annual_Income", "Credit_Score",
    "Existing_Loans", "Loan_Amount", "Loan_Term_Months",
    "Interest_Rate", "Monthly_Installment",
    "Debt_to_Income_Ratio", "Employment_Years",
    "Previous_Defaults", "Default_Flag"
]
plt.figure(figsize=(13,9))
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
save_show("15_Correlation_Heatmap.png")

# -----------------------------
# 8. AUTOMATIC BUSINESS INSIGHTS
# -----------------------------
default_summary = df.groupby("Default_Status").agg(
    Avg_Credit_Score=("Credit_Score","mean"),
    Avg_Income=("Annual_Income","mean"),
    Avg_Loan_Amount=("Loan_Amount","mean"),
    Avg_DTI=("Debt_to_Income_Ratio","mean"),
    Avg_Previous_Defaults=("Previous_Defaults","mean")
).round(2)

print("\n========== BUSINESS INSIGHTS ==========")
print(f"1. Overall loan default rate is {default_rate:.2f}%.")
print(f"2. Highest-default loan type: {loan_type_rates.iloc[0]['Loan_Type']} ({loan_type_rates.iloc[0]['Default_Rate']:.2f}%).")
print(f"3. Highest-default region: {region_rates.iloc[0]['Region']} ({region_rates.iloc[0]['Default_Rate']:.2f}%).")
print("4. Compare average credit score between defaulted and non-defaulted borrowers:")
print(default_summary[["Avg_Credit_Score"]])
print("5. Compare average income between defaulted and non-defaulted borrowers:")
print(default_summary[["Avg_Income"]])
print("6. Compare average DTI between defaulted and non-defaulted borrowers:")
print(default_summary[["Avg_DTI"]])
print("7. Compare previous defaults between defaulted and non-defaulted borrowers:")
print(default_summary[["Avg_Previous_Defaults"]])
if not segment_analysis.empty:
    s = segment_analysis.iloc[0]
    print(f"8. Highest-risk sufficiently sized segment: {s['Credit_Score_Group']} credit / {s['Income_Group']} with {s['Default_Rate']:.2f}% default rate.")
print("9. Loan amount comparison by default status:")
print(default_summary[["Avg_Loan_Amount"]])
print("10. Credit-risk monitoring should focus on credit score, DTI, previous defaults, income, loan type, and repayment/default status.")

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")
print(f"Cleaned CSV: {CLEAN_FILE}")
print(f"Charts folder: {CHART_FOLDER}")
