import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────
folder = Path("./GoogleTrendsData")
kw_files = {
    "debt"         : "debtMonthly.csv",
    "deficit"      : "deficitMonthly.csv",
    "repayment"    : "repaymentMonthly.csv",
    "gov_spending" : "GovernmentSpendingMonthly.csv",
   # "debt_ceiling" : "debtCeilingMonthly.csv",
    #"debt_to_gdp"  : "debtToGdpMonthly.csv",
    "public_debt"  : "publicDebtMonthly.csv",
}

def read_monthly(label, fn):
    df = pd.read_csv(
        folder/ fn,
        skiprows=1,
        header=0,
        parse_dates=["Month"],
        index_col="Month"
    )
    df.columns = [label]
    df[label] = df[label].replace("<1", "0.5").astype(float)
    return df

# ── 1) Read & concat ────────────────────────────────────────────────────
monthly = pd.concat(
    [read_monthly(label, fn) for label, fn in kw_files.items()],
    axis=1
).ffill()

# ── 2) Compute calendar-year means ──────────────────────────────────────
yearly = monthly.resample("Y").mean()
yearly.index = yearly.index.year  # simplify index to 2004, 2005, …

# ── 3) Plot yearly averages ─────────────────────────────────────────────
plt.figure(figsize=(12,6))
for col in yearly.columns:
    plt.plot(yearly.index, yearly[col], marker="o", label=col, linewidth=1.5)

plt.title("Yearly Average Google-Trends Series")
plt.xlabel("Year")
plt.ylabel("Average Search Interest")
plt.xticks(yearly.index, yearly.index.astype(int), rotation=0)
plt.legend(loc="upper left", ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 4) Compute Year-over-Year % Change ─────────────────────────────────
yearly_pct = yearly.pct_change() * 100

# ── 5) Plot yearly % changes ───────────────────────────────────────────
plt.figure(figsize=(12,6))
for col in yearly_pct.columns:
    plt.plot(yearly_pct.index, yearly_pct[col], marker="o", label=col, linewidth=1.5)

plt.title("Year-over-Year % Change in Google-Trends Series")
plt.xlabel("Year")
plt.ylabel("% Change from Prior Year")
plt.xticks(yearly_pct.index, yearly_pct.index.astype(int), rotation=0)
plt.axhline(0, color="gray", lw=0.8)  # zero‐growth line
plt.legend(loc="upper left", ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
