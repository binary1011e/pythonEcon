import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader.fred import FredReader
from pathlib import Path
from sklearn.metrics import r2_score

# ── CONFIG ─────────────────────────────────────────────────────────
folder = Path("./GoogleTrendsData")   # adjust to your monthly CSV directory
kw_files = {
    "debt"         : "debtMonthly.csv",
    "deficit"      : "deficitMonthly.csv",
    "repayment"    : "repaymentMonthly.csv",
    "gov_spending" : "GovernmentSpendingMonthly.csv",
    "debt_ceiling" : "debtCeilingMonthly.csv",
    "debt_to_gdp"  : "debtToGdpMonthly.csv",
    "public_debt"  : "publicDebtMonthly.csv",
}

def read_monthly(label, fn):
    df = pd.read_csv(
        folder/ fn,
        skiprows=1,            # drop the "Category:" line
        header=0,
        parse_dates=["Month"],
        index_col="Month"
    )
    df.columns = [label]
    df[label] = df[label].replace("<1", "0.5")
    df[label] = pd.to_numeric(df[label], errors="coerce")
    return df

# ── 1) Load & align monthly series ──────────────────────────────────
monthly = pd.concat(
    [read_monthly(k, fn) for k,fn in kw_files.items()],
    axis=1
).ffill()

exp = FredReader("EXPINF1YR",
                 start=monthly.index.min(),
                 end  =monthly.index.max()
                ).read()
exp.index.name = "Month"
exp.rename(columns={"EXPINF1YR":"Expect"}, inplace=True)

data = exp.join(monthly, how="inner").dropna()

# ── 2) Standardize all trend series ─────────────────────────────────
trends   = data.drop(columns="Expect")
z_trends = (trends - trends.mean()) / trends.std()

# ── 3) First-stage OLS: build Debt-Worry index ───────────────────────
Y = data["Expect"]
X = sm.add_constant(z_trends)
first = sm.OLS(Y, X).fit()

weights = first.params.drop("const")
print(weights)
intercept = first.params["const"]

DW = (z_trends * weights).sum(axis=1).rename("DW")

# ── 4) Train/test split ─────────────────────────────────────────────
end_date = pd.to_datetime("2025-12-31")
data = pd.concat([data["Expect"], DW], axis=1).loc[:end_date].dropna()

split = "2020-01-01"
train = data.loc[:split]
test  = data.loc[ split:]

# ── 5) No-lag model: Expect ~ DW ───────────────────────────────────
X_tr = sm.add_constant(train[["DW"]])
y_tr = train["Expect"]
model = sm.OLS(y_tr, X_tr).fit()
print(model.summary())

# ── 6) Out-of-sample R² ─────────────────────────────────────────────
X_te = sm.add_constant(test[["DW"]])
y_te = test["Expect"]
y_pred_te = model.predict(X_te)
print("Test R² =", r2_score(y_te, y_pred_te))

# ── 7) Coefficient ±95% CI ─────────────────────────────────────────
coef = model.params.drop("const")
ci   = model.conf_int().drop("const")
err_lo = coef - ci[0]
err_hi = ci[1] - coef

plt.figure(figsize=(8,4))
plt.bar(coef.index, coef.values, yerr=[err_lo, err_hi], capsize=5)
plt.axhline(0, color="gray", lw=0.8)
plt.title("Model Coefficient ±95% CI")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.show()

# ── 8) –log10(p-value) significance ─────────────────────────────────
pvals   = model.pvalues.drop("const")
neglogp = -np.log10(pvals)

plt.figure(figsize=(8,4))
plt.bar(neglogp.index, neglogp.values)
plt.axhline(-np.log10(0.05), color="red", ls="--", label="p=0.05")
plt.title("Significance (−log10 p-values)")
plt.ylabel("−log10(p-value)")
plt.legend()
plt.tight_layout()
plt.show()

# ── 9) Actual vs Predicted ──────────────────────────────────────────
pred_all = model.predict(sm.add_constant(data[["DW"]]))

plt.figure(figsize=(10,4))
plt.plot(data.index, data["Expect"], label="Actual", color="C1")
plt.plot(data.index, pred_all,           label="Predicted", color="C0")
plt.axvline(pd.to_datetime(split), ls="--", color="gray", label="Train/Test")
plt.title("Actual vs Predicted Expectations")
plt.ylabel("Expect (%)")
plt.legend()
plt.tight_layout()
plt.show()

DW = (z_trends * weights).sum(axis=1).rename("DW")
data = exp.join(monthly, how="inner").dropna()  # or whatever your final `data` is

# ── 12) Equally‐weighted Debt‐Worry index ─────────────────────────────
DW_eq = z_trends.mean(axis=1).rename("DW_eq")

# ── 14) Plot Equal-Weight DW & Inflation Expectations ──────────────────
fig, ax1 = plt.subplots(figsize=(10,4))

# Debt-Worry (equal weight)
ax1.plot(DW_eq.index, DW_eq, color="C2", lw=1.5, label="Equal-Weight DW")
ax1.set_ylabel("Debt-Worry Index (z-score)", color="C2")
ax1.tick_params(axis="y", labelcolor="C2")

# second axis for inflation expectations
ax2 = ax1.twinx()
ax2.plot(data.index, data["Expect"], color="C3", lw=1.5, label="1-yr Exp. Inflation")
ax2.set_ylabel("Inflation Expectation (%)", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("Equal-Weight DW & 1-Year Inflation Expectations")
fig.tight_layout()
plt.show()


# ── 15) Plot Regression-Weight DW & Inflation Expectations ─────────────
fig, ax1 = plt.subplots(figsize=(10,4))

# Debt-Worry (regression weight)
ax1.plot(DW.index, DW, color="C0", lw=1.5, label="Regression-Weight DW")
ax1.set_ylabel("Debt-Worry Index (z-score)", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

# second axis for inflation expectations
ax2 = ax1.twinx()
ax2.plot(data.index, data["Expect"], color="C3", lw=1.5, label="1-yr Exp. Inflation")
ax2.set_ylabel("Inflation Expectation (%)", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("Regression-Weight DW & 1-Year Inflation Expectations")
fig.tight_layout()
plt.show()
