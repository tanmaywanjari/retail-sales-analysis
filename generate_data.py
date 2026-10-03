"""Generates a realistic synthetic retail transactions dataset (2023-2024).
Replace data/retail_sales.csv with a real dataset (e.g. Kaggle 'Superstore') if you prefer;
see README for the column mapping."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 12000
cats = {"Technology": (["Laptop","Phone","Headphones","Monitor","Keyboard"], (40,1200), .22),
        "Furniture":  (["Chair","Desk","Bookcase","Lamp","Sofa"], (30,900), .12),
        "Office Supplies": (["Paper","Binder","Pens","Storage Box","Labels"], (2,60), .30)}
regions = {"West":.32,"East":.30,"Central":.22,"South":.16}
segments = {"Consumer":.52,"Corporate":.30,"Home Office":.18}
dates = pd.date_range("2023-01-01","2024-12-31")
# seasonality: Nov/Dec peak, mild summer bump, upward trend
w = np.array([1+0.6*(d.month in (11,12))+0.15*(d.month in (6,7))+0.0006*(d-dates[0]).days for d in dates])
od = rng.choice(dates, N, p=w/w.sum())
cat = rng.choice(list(cats), N, p=[.25,.2,.55])
rows=[]
for i in range(N):
    prods,(lo,hi),base = cats[cat[i]]
    price = round(float(np.exp(rng.uniform(np.log(lo),np.log(hi)))),2)
    qty = int(rng.integers(1,8))
    disc = float(rng.choice([0,.1,.2,.3,.4],p=[.45,.2,.2,.1,.05]))
    sales = round(price*qty*(1-disc),2)
    margin = base + 0.02 - 0.65*disc*(1 if cat[i]!="Technology" else .8) + rng.normal(0,.04)
    rows.append((f"ORD-{i+1:05d}", rng.choice(prods), price, qty, disc, sales, round(sales*margin,2)))
df = pd.DataFrame(rows, columns=["order_id","product","unit_price","quantity","discount","sales","profit"])
df.insert(1,"order_date",pd.to_datetime(od)); df.insert(2,"category",cat)
df["region"]=rng.choice(list(regions),N,p=list(regions.values()))
df["segment"]=rng.choice(list(segments),N,p=list(segments.values()))
df["customer_id"]=["CUST-%04d"%x for x in rng.integers(1,2200,N)]
# inject realistic messiness
df.loc[rng.choice(N,120,replace=False),"region"]=np.nan
df = pd.concat([df, df.sample(40, random_state=1)])  # duplicate rows
df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv("data/retail_sales.csv", index=False)
print(df.shape)
