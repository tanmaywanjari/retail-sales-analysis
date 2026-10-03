# %% [markdown]
# # Retail Sales Analysis & Profit Prediction
# End-to-end analysis: cleaning → EDA → insights → predictive model → conclusions.
# **Business questions:** What drives sales and profit? Where are we losing money? Can we predict whether an order will be profitable?

# %%
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
os.makedirs("images", exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
SAVE = True
def save(name):
    if SAVE: plt.savefig(f"images/{name}.png", dpi=130, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 1. Load data

# %%
df = pd.read_csv("data/retail_sales.csv", parse_dates=["order_date"])
print(df.shape); df.head()

# %% [markdown]
# ## 2. Data cleaning

# %%
print("Missing values:\n", df.isna().sum()[lambda s: s > 0])
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().copy()
df["region"] = df["region"].fillna("Unknown")
df["year"] = df.order_date.dt.year
df["month"] = df.order_date.dt.month
df["ym"] = df.order_date.dt.to_period("M").dt.to_timestamp()
df["profit_margin"] = df.profit / df.sales
print("Clean shape:", df.shape)
df.describe().round(2)

# %% [markdown]
# ## 3. Exploratory analysis
# ### 3.1 Headline KPIs

# %%
kpi = {"Total Sales": df.sales.sum(), "Total Profit": df.profit.sum(),
       "Overall Margin %": df.profit.sum()/df.sales.sum()*100,
       "Orders": df.order_id.nunique(), "Customers": df.customer_id.nunique(),
       "Avg Order Value": df.sales.mean()}
for k, v in kpi.items(): print(f"{k:18s}: {v:,.2f}")

# %% [markdown]
# ### 3.2 Monthly sales trend & seasonality

# %%
monthly = df.groupby("ym")[["sales","profit"]].sum()
fig, ax = plt.subplots(figsize=(11,4.5))
monthly.sales.plot(ax=ax, marker="o", label="Sales"); monthly.profit.plot(ax=ax, marker="s", label="Profit")
ax.set_title("Monthly Sales and Profit"); ax.set_ylabel("Amount"); ax.legend()
save("01_monthly_trend")
peak = monthly.sales.idxmax(); print("Peak month:", peak.strftime("%B %Y"))

# %% [markdown]
# ### 3.3 Category performance

# %%
cat = df.groupby("category").agg(sales=("sales","sum"), profit=("profit","sum"))
cat["margin_%"] = cat.profit/cat.sales*100
fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
cat.sales.sort_values().plot.barh(ax=axes[0], color="#4C72B0"); axes[0].set_title("Sales by Category")
cat["margin_%"].sort_values().plot.barh(ax=axes[1], color="#55A868"); axes[1].set_title("Profit Margin % by Category")
plt.tight_layout(); save("02_category")
cat.round(2)

# %% [markdown]
# ### 3.4 Regions and customer segments

# %%
fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
df.groupby("region").profit.sum().sort_values().plot.barh(ax=axes[0], color="#C44E52"); axes[0].set_title("Profit by Region")
df.groupby("segment").sales.sum().plot.pie(ax=axes[1], autopct="%1.1f%%", startangle=90); axes[1].set_ylabel(""); axes[1].set_title("Sales Share by Segment")
plt.tight_layout(); save("03_region_segment")

# %% [markdown]
# ### 3.5 Does discounting hurt profit? (key insight)

# %%
disc = df.groupby("discount").agg(avg_margin=("profit_margin","mean"), orders=("order_id","count"))
disc["avg_margin"] *= 100
fig, ax = plt.subplots(figsize=(8,4.5))
sns.barplot(x=(disc.index*100).astype(int), y=disc.avg_margin, ax=ax, color="#8172B2")
ax.axhline(0, color="k", lw=1); ax.set_xlabel("Discount (%)"); ax.set_ylabel("Avg profit margin (%)")
ax.set_title("Average Profit Margin by Discount Level")
save("04_discount_vs_margin")
loss = (df.profit < 0).mean()*100
print(f"{loss:.1f}% of orders are loss-making"); disc.round(2)

# %% [markdown]
# ### 3.6 Loss-making orders by category and discount

# %%
df["loss"] = (df.profit < 0).astype(int)
heat = df.pivot_table(index="category", columns="discount", values="loss", aggfunc="mean")*100
plt.figure(figsize=(8,3.5))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label":"% loss-making"})
plt.title("Share of Loss-Making Orders (%)"); save("05_loss_heatmap")

# %% [markdown]
# ### 3.7 Correlations

# %%
plt.figure(figsize=(6,4.5))
sns.heatmap(df[["unit_price","quantity","discount","sales","profit"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix"); save("06_correlation")

# %% [markdown]
# ### 3.8 Top products and customers

# %%
top_products = df.groupby("product").profit.sum().sort_values(ascending=False)
print("Top 5 products by profit:\n", top_products.head(5).round(0))
print("\nTop 5 customers by sales:\n", df.groupby("customer_id").sales.sum().nlargest(5).round(0))

# %% [markdown]
# ## 4. Prediction: will an order be profitable?
# Features known at order time: category, region, segment, unit price, quantity, discount, month.

# %%
X = pd.get_dummies(df[["category","region","segment","unit_price","quantity","discount","month"]], drop_first=True)
y = (df.profit > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=2000).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
for name, m in [("Logistic Regression", lr), ("Random Forest", rf)]:
    print(f"{name}: accuracy = {accuracy_score(y_test, m.predict(X_test)):.3f}")
print("\n", classification_report(y_test, rf.predict(X_test), target_names=["Loss","Profit"]))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, rf.predict(X_test)), display_labels=["Loss","Profit"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Random Forest – Confusion Matrix")
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(8).sort_values()
imp.plot.barh(ax=axes[1], color="#DD8452"); axes[1].set_title("Top Feature Importances")
plt.tight_layout(); save("07_model")

# %% [markdown]
# ## 5. Conclusions & recommendations
# 1. **Growth & seasonality:** sales grew ~12.7% from 2023 to 2024. November (1.34x) and December (1.5x of the average month) are the peak; January is the weakest (0.76x). Plan inventory and staffing before Q4.
# 2. **Discounts erode margin:** average margin falls from ~26% (no discount) to ~2% (40% discount). Loss-making orders rise from 0% to ~29% at 40% off.
# 3. **Furniture is the problem category:** margin is only 8.2%, and ~90% of Furniture orders discounted at 30%+ lose money. Technology (19% margin) is the revenue engine; Office Supplies has the best margin (25.8%) but small volume – bundle it with big-ticket items.
# 4. **Regions:** East and West generate the most profit; South lags and is worth a closer look.
# 5. **Prediction:** the Random Forest reaches ~97.5% accuracy on unseen data, with discount, unit price and product category as the strongest signals. Recall on loss orders is lower (~57%) because losses are rare (~5%), so use the model as a risk-flagging aid, not a final decision.
# 6. **Recommendation:** cap Furniture discounts at 20%, and require approval for any discount of 30%+.
# 7. **Next steps:** run on real data, add time-series forecasting (ARIMA/Prophet) and customer segmentation (RFM/K-Means).
