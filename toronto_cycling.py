
import warnings, os, json
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.metrics           import (classification_report, roc_auc_score,
                                        ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection        import permutation_importance
import xgboost as xgb

np.random.seed(42)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT        = os.path.join(SCRIPT_DIR, "figures")
DATA_PATH  = "cycling-network_-_4326.csv"
YEAR_NOW   = 2024
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("1. LOAD & CLEAN")
print("=" * 62)

df = pd.read_csv(DATA_PATH)
print(f"  Raw records : {len(df):,}  |  Columns : {df.shape[1]}")

# Drop rows missing installation year (only 15)
df = df[df["INSTALLED"].notna()].copy()
df["INSTALLED"] = df["INSTALLED"].astype(int)

# Valid upgrade year = value > 2000 (some '1' placeholders exist)
df["UPGRADED_VALID"] = df["UPGRADED"].apply(
    lambda x: int(x) if pd.notna(x) and float(x) > 2000 else np.nan)

# ── Haversine segment length ─────────────────────────────────────────────────
def haversine_segment(coords):
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i];  lon2, lat2 = coords[i + 1]
        R    = 6_371_000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1);  dlam = np.radians(lon2 - lon1)
        a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        total += 2 * R * np.arcsin(np.sqrt(a))
    return total

lengths = []
centroids_lon, centroids_lat = [], []
for g in df["geometry"]:
    try:
        gj   = json.loads(g)
        segs = gj["coordinates"]
        total_len = sum(haversine_segment(s) for s in segs)
        all_pts   = [pt for s in segs for pt in s]
        clat = np.mean([p[1] for p in all_pts])
        clon = np.mean([p[0] for p in all_pts])
        lengths.append(total_len)
        centroids_lat.append(clat)
        centroids_lon.append(clon)
    except:
        lengths.append(np.nan)
        centroids_lat.append(np.nan)
        centroids_lon.append(np.nan)

df["length_m"]      = lengths
df["centroid_lat"]  = centroids_lat
df["centroid_lon"]  = centroids_lon
df.dropna(subset=["length_m"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  After cleaning  : {len(df):,} segments")
print(f"  Total network   : {df['length_m'].sum()/1000:.1f} km")
print(f"  Install range   : {df['INSTALLED'].min()} – {df['INSTALLED'].max()}")
print(f"  Upgraded (valid): {df['UPGRADED_VALID'].notna().sum()} segments "
      f"({df['UPGRADED_VALID'].notna().mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("2. FEATURE ENGINEERING")
print("=" * 62)

# Infrastructure safety tier (higher = safer, more separated)
INFRA_TIER = {
    "Sharrows"                           : 1,
    "Sharrows - Wayfinding"              : 1,
    "Sharrows - Arterial"                : 1,
    "Sharrows - Arterial - Connector"    : 1,
    "Signed Route (No Pavement Markings)": 2,
    "Park Road"                          : 2,
    "Bike Lane"                          : 3,
    "Bike Lane - Contraflow"             : 3,
    "Bike Lane - Buffered"               : 3,
    "Cycle Track"                        : 4,
    "Cycle Track - Contraflow"           : 4,
    "Bi-Directional Cycle Track"         : 4,
    "Multi-Use Trail"                    : 5,
    "Multi-Use Trail - Entrance"         : 5,
    "Multi-Use Trail - Boulevard"        : 5,
    "Multi-Use Trail - Existing Connector":5,
    "Multi-Use Trail - Connector"        : 5,
}

df["infra_tier"]   = df["INFRA_HIGHORDER"].map(INFRA_TIER).fillna(2)
df["age"]          = YEAR_NOW - df["INSTALLED"]
df["has_upgraded"] = df["UPGRADED_VALID"].notna().astype(int)

# Segments that were upgraded: upgrade lag
df["upgrade_lag"]  = df["UPGRADED_VALID"] - df["INSTALLED"]

# Decade installed
df["decade"]       = (df["INSTALLED"] // 10 * 10).astype(str) + "s"

# Low-safety flag: tier ≤ 2 AND age > 10
df["low_safety_old"] = ((df["infra_tier"] <= 2) & (df["age"] > 10)).astype(int)

# Priority score (prescriptive) = age × (6 − infra_tier) / length bonus
# Higher score = more urgently needs upgrade
df["priority_raw"]  = df["age"] * (6 - df["infra_tier"])
df["priority_score"]= (df["priority_raw"] / df["priority_raw"].max() * 100).round(1)

print(f"  Tiers assigned  : {df['infra_tier'].value_counts().sort_index().to_dict()}")
print(f"  Mean segment age: {df['age'].mean():.1f} yrs")
print(f"  Low-safety & old: {df['low_safety_old'].sum()} segments")

FEATURES = ["age", "infra_tier", "length_m", "low_safety_old",
            "centroid_lat", "centroid_lon"]
X = df[FEATURES]
y = df["has_upgraded"]
print(f"\n  Feature matrix  : {X.shape[0]} × {X.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 62)

# Colour palette per infra tier
TIER_COLORS = {1:"#d62728", 2:"#ff7f0e", 3:"#ffd166", 4:"#06d6a0", 5:"#457b9d"}
TIER_LABELS = {1:"Sharrows", 2:"Signed Route", 3:"Bike Lane",
               4:"Cycle Track", 5:"Multi-Use Trail"}

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Toronto Cycling Network – Exploratory Data Analysis",
             fontsize=15, fontweight="bold", y=1.01)

# 3a – Segments installed per year (stacked by tier)
ax = fig.add_subplot(gs[0, 0])
pivot = (df.groupby(["INSTALLED","infra_tier"])["SEGMENT_ID"]
           .count().unstack(fill_value=0))
bottom = np.zeros(len(pivot))
for tier in sorted(pivot.columns):
    ax.bar(pivot.index, pivot[tier], bottom=bottom,
           color=TIER_COLORS[tier], label=TIER_LABELS[tier], width=0.8)
    bottom += pivot[tier].values
ax.set_title("Segments Installed per Year"); ax.set_xlabel("Year"); ax.set_ylabel("Count")
ax.legend(fontsize=7, loc="upper left")

# 3b – Km of network by infra type
ax = fig.add_subplot(gs[0, 1])
km_by_infra = (df.groupby("INFRA_HIGHORDER")["length_m"].sum() / 1000).sort_values()
colors_bar  = [TIER_COLORS[INFRA_TIER.get(i, 2)] for i in km_by_infra.index]
ax.barh(km_by_infra.index, km_by_infra.values, color=colors_bar)
ax.set_title("Network Length by Infrastructure Type"); ax.set_xlabel("km")
ax.tick_params(axis="y", labelsize=7)

# 3c – Upgrade rate by infra tier
ax = fig.add_subplot(gs[0, 2])
upgrade_rate = df.groupby("infra_tier")["has_upgraded"].mean() * 100
bars = ax.bar([TIER_LABELS[t] for t in upgrade_rate.index], upgrade_rate.values,
              color=[TIER_COLORS[t] for t in upgrade_rate.index])
ax.set_title("% Segments Upgraded by Tier"); ax.set_ylabel("Upgrade Rate (%)")
ax.set_xticklabels([TIER_LABELS[t] for t in upgrade_rate.index],
                    rotation=20, ha="right", fontsize=8)
for bar, v in zip(bars, upgrade_rate.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")

# 3d – Age distribution by tier
ax = fig.add_subplot(gs[1, 0])
for tier in sorted(TIER_LABELS):
    sub = df[df["infra_tier"] == tier]["age"]
    ax.hist(sub, bins=20, alpha=0.65, color=TIER_COLORS[tier],
            label=TIER_LABELS[tier], edgecolor="white", lw=0.3)
ax.set_title("Age Distribution by Tier"); ax.set_xlabel("Segment Age (years)")
ax.set_ylabel("Count"); ax.legend(fontsize=7)

# 3e – Geographic map (dot per segment, colour = tier)
ax = fig.add_subplot(gs[1, 1])
for tier in sorted(TIER_LABELS):
    sub = df[df["infra_tier"] == tier]
    ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
               s=3, alpha=0.5, color=TIER_COLORS[tier], label=TIER_LABELS[tier])
ax.set_title("Geographic Distribution of Segments\n(Colour = Infrastructure Tier)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(fontsize=6, markerscale=2)
ax.tick_params(labelsize=7)

# 3f – Upgrade lag histogram for upgraded segments
ax = fig.add_subplot(gs[1, 2])
valid_lag = df[df["upgrade_lag"].notna() & (df["upgrade_lag"] >= 0)]["upgrade_lag"]
ax.hist(valid_lag, bins=25, color="#2d6a9f", edgecolor="white", lw=0.4)
ax.axvline(valid_lag.median(), color="red", ls="--", lw=1.8,
           label=f"Median = {valid_lag.median():.0f} yrs")
ax.set_title("Time to Upgrade (Upgraded Segments Only)")
ax.set_xlabel("Years Between Installation & Upgrade"); ax.set_ylabel("Count")
ax.legend()

plt.savefig(f"{OUT}/01_eda.png", bbox_inches="tight")
plt.close()
print("  Saved → figures/01_eda.png")

# Print some key stats
print(f"\n  Upgrade rate overall : {y.mean()*100:.1f}%")
print(f"  Median upgrade lag   : {valid_lag.median():.0f} years")
print(f"  Tier 1 (Sharrows) km : {df[df.infra_tier==1]['length_m'].sum()/1000:.1f} km")
print(f"  Tier 4+ km (protected): {df[df.infra_tier>=4]['length_m'].sum()/1000:.1f} km")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTIVE MODELLING – What drives segment upgrades?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("4. PREDICTIVE MODELLING")
print("=" * 62)

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression" : LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=300, max_depth=8,
                                                    min_samples_leaf=3, n_jobs=-1,
                                                    random_state=42),
    "XGBoost"             : xgb.XGBClassifier(n_estimators=300, max_depth=5,
                                               learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, use_label_encoder=False,
                                               eval_metric="logloss", n_jobs=-1,
                                               random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    Xin = X_sc if name == "Logistic Regression" else X
    roc_scores = cross_val_score(model, Xin, y, cv=cv, scoring="roc_auc")
    f1_scores  = cross_val_score(model, Xin, y, cv=cv, scoring="f1")
    model.fit(Xin, y)
    results[name] = {
        "model"    : model,
        "roc_mean" : roc_scores.mean(),
        "roc_std"  : roc_scores.std(),
        "f1_mean"  : f1_scores.mean(),
        "f1_std"   : f1_scores.std(),
        "proba"    : model.predict_proba(Xin)[:, 1],
    }
    print(f"  {name:<22} ROC-AUC={roc_scores.mean():.3f} ± {roc_scores.std():.3f}"
          f"   F1={f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

best_name = max(results, key=lambda k: results[k]["roc_mean"])
best      = results[best_name]
print(f"\n  ★  Best model : {best_name}  (ROC-AUC = {best['roc_mean']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(f"Model Diagnostics – {best_name}", fontsize=14, fontweight="bold")

best_model = best["model"]
Xin_best   = X_sc if best_name == "Logistic Regression" else X

# 5a – ROC curves for all models
ax = axes[0, 0]
for name, res in results.items():
    Xin = X_sc if name == "Logistic Regression" else X
    RocCurveDisplay.from_estimator(res["model"], Xin, y, ax=ax,
        name=f"{name} ({res['roc_mean']:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_title("ROC Curves (5-fold CV means)"); ax.legend(fontsize=8)

# 5b – Confusion matrix (best model)
ax = axes[0, 1]
ConfusionMatrixDisplay.from_estimator(best_model, Xin_best, y, ax=ax,
    colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix – {best_name}")

# 5c – ROC-AUC comparison bar
ax = axes[0, 2]
names  = list(results.keys())
aucs   = [results[n]["roc_mean"] for n in names]
colors = ["#e07b39" if n == best_name else "#a8c7e8" for n in names]
bars   = ax.bar(names, aucs, color=colors)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("ROC-AUC")
ax.set_title("Model ROC-AUC Comparison")
ax.set_xticklabels(names, rotation=15, ha="right")
for bar, v in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

# 5d – Feature importances (XGBoost)
ax = axes[1, 0]
xgb_model  = results["XGBoost"]["model"]
imp_df = pd.DataFrame({
    "feature"   : FEATURES,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=True)
ax.barh(imp_df["feature"], imp_df["importance"], color="#2d6a9f")
ax.set_title("Feature Importances – XGBoost")
ax.set_xlabel("Gain Importance")

# 5e – Upgrade probability distribution by tier (best model)
ax = axes[1, 1]
df["upgrade_prob"] = best["proba"]
for tier in sorted(TIER_LABELS):
    sub = df[df["infra_tier"] == tier]["upgrade_prob"]
    ax.hist(sub, bins=20, alpha=0.65, color=TIER_COLORS[tier],
            label=TIER_LABELS[tier], edgecolor="white", lw=0.3)
ax.set_title("Predicted Upgrade Probability by Tier")
ax.set_xlabel("P(Upgrade)"); ax.set_ylabel("Count"); ax.legend(fontsize=7)

# 5f – Age vs upgrade probability scatter
ax = axes[1, 2]
sc = ax.scatter(df["age"], df["upgrade_prob"],
                c=df["infra_tier"], cmap="RdYlGn",
                s=10, alpha=0.5, vmin=1, vmax=5)
plt.colorbar(sc, ax=ax, label="Infra Tier")
ax.set_title("Age vs Predicted Upgrade Probability")
ax.set_xlabel("Segment Age (years)"); ax.set_ylabel("P(Upgrade)")

plt.tight_layout()
plt.savefig(f"{OUT}/02_diagnostics.png", bbox_inches="tight")
plt.close()
print("\n  Saved → figures/02_diagnostics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRESCRIPTIVE – Upgrade Priority Roadmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("5. PRESCRIPTIVE – UPGRADE PRIORITY ROADMAP")
print("=" * 62)

# Only consider segments NOT yet upgraded
not_upgraded = df[df["has_upgraded"] == 0].copy()

# Combined priority score: age × safety gap × (1 + model's upgrade probability inverse)
# High score = old, low-safety, model says unlikely to be upgraded organically
not_upgraded["combined_priority"] = (
    not_upgraded["age"]
    * (6 - not_upgraded["infra_tier"])
    * (1.5 - not_upgraded["upgrade_prob"])   # model-adjusted urgency
    * (not_upgraded["length_m"] / 1000 + 0.1)
)
not_upgraded["combined_priority"] = (
    (not_upgraded["combined_priority"] - not_upgraded["combined_priority"].min())
    / (not_upgraded["combined_priority"].max() - not_upgraded["combined_priority"].min()) * 100
).round(1)

# Classify into tiers
p90 = not_upgraded["combined_priority"].quantile(0.90)
p70 = not_upgraded["combined_priority"].quantile(0.70)
p40 = not_upgraded["combined_priority"].quantile(0.40)

def priority_band(score):
    if score >= p90: return "CRITICAL"
    if score >= p70: return "HIGH"
    if score >= p40: return "MEDIUM"
    return "LOW"

not_upgraded["priority_band"] = not_upgraded["combined_priority"].apply(priority_band)

band_counts = not_upgraded["priority_band"].value_counts()
band_km     = not_upgraded.groupby("priority_band")["length_m"].sum() / 1000

print("\n  Priority Band Distribution (unupgraded segments):")
print(f"  {'Band':<12} {'Segments':>9} {'Network km':>12}")
print("  " + "-"*36)
for band in ["CRITICAL","HIGH","MEDIUM","LOW"]:
    if band in band_counts.index:
        print(f"  {band:<12} {band_counts[band]:>9}  {band_km[band]:>10.1f} km")

# Top 20 priority segments
top20 = (not_upgraded.nlargest(20, "combined_priority")
         [["STREET_NAME","FROM_STREET","TO_STREET","INSTALLED","infra_tier",
           "age","length_m","combined_priority","priority_band"]]
         .reset_index(drop=True))
top20.index += 1
top20.columns = ["Street","From","To","Yr Installed","Tier","Age (yrs)",
                 "Length (m)","Priority Score","Band"]
print("\n  Top 20 Segments for Priority Upgrade:")
print(top20[["Street","Yr Installed","Tier","Age (yrs)","Length (m)",
             "Priority Score","Band"]].to_string())

# Cost estimation
COST_PER_KM = {
    "Bike Lane"  : 150_000,    # city avg $/km (protected conversion)
    "Sharrows"   : 80_000,
    "Other"      : 120_000,
}
critical_km = band_km.get("CRITICAL", 0)
high_km     = band_km.get("HIGH", 0)
est_cost_critical = critical_km * 130_000   # blended rate
est_cost_high     = high_km     * 110_000
print(f"\n  Estimated upgrade cost – CRITICAL segments : ${est_cost_critical:,.0f}")
print(f"  Estimated upgrade cost – HIGH segments     : ${est_cost_high:,.0f}")
print(f"  Combined 5-year investment estimate        : ${(est_cost_critical+est_cost_high):,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRESCRIPTIVE VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
BAND_COLORS = {"CRITICAL":"#d62728","HIGH":"#ff7f0e","MEDIUM":"#ffd166","LOW":"#457b9d"}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Prescriptive Upgrade Roadmap – Toronto Cycling Network",
             fontsize=14, fontweight="bold")

# 7a – Geographic priority map
ax = axes[0, 0]
for band in ["LOW","MEDIUM","HIGH","CRITICAL"]:
    sub = not_upgraded[not_upgraded["priority_band"] == band]
    ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
               s=5, alpha=0.7, color=BAND_COLORS[band], label=band, zorder=3)
# Plot already-upgraded in light grey
upgraded_segs = df[df["has_upgraded"] == 1]
ax.scatter(upgraded_segs["centroid_lon"], upgraded_segs["centroid_lat"],
           s=2, alpha=0.3, color="#aaaaaa", label="Already Upgraded", zorder=1)
ax.set_title("Priority Map – Segments Needing Upgrade")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
legend = ax.legend(fontsize=8, markerscale=2, title="Priority Band",
                    title_fontsize=8)
ax.tick_params(labelsize=8)

# 7b – Priority score distribution
ax = axes[0, 1]
for band in ["LOW","MEDIUM","HIGH","CRITICAL"]:
    sub = not_upgraded[not_upgraded["priority_band"] == band]["combined_priority"]
    ax.hist(sub, bins=30, color=BAND_COLORS[band], alpha=0.75,
            label=band, edgecolor="white", lw=0.3)
ax.set_title("Priority Score Distribution"); ax.set_xlabel("Combined Priority Score")
ax.set_ylabel("Number of Segments"); ax.legend()

# 7c – Km by band × tier stacked
ax = axes[1, 0]
pivot_km = (not_upgraded.groupby(["priority_band","infra_tier"])["length_m"]
              .sum().unstack(fill_value=0) / 1000)
pivot_km = pivot_km.reindex(["LOW","MEDIUM","HIGH","CRITICAL"], fill_value=0)
bottom = np.zeros(len(pivot_km))
for tier in sorted(TIER_LABELS):
    if tier in pivot_km.columns:
        vals = pivot_km[tier].values
        ax.bar(pivot_km.index, vals, bottom=bottom,
               color=TIER_COLORS[tier], label=TIER_LABELS[tier], width=0.6)
        bottom += vals
ax.set_title("Network km to Upgrade by Band & Infrastructure Type")
ax.set_ylabel("km"); ax.set_xlabel("Priority Band")
ax.legend(fontsize=7, loc="upper right")

# 7d – 5-year phased investment timeline
ax = axes[1, 1]
phases = {
    "Year 1–2\n(CRITICAL)": {"km": critical_km, "cost": est_cost_critical, "color": "#d62728"},
    "Year 3–4\n(HIGH)"    : {"km": high_km,     "cost": est_cost_high,     "color": "#ff7f0e"},
    "Year 5+\n(MEDIUM)"   : {"km": band_km.get("MEDIUM",0),
                              "cost": band_km.get("MEDIUM",0)*90_000,
                              "color": "#ffd166"},
}
phase_labels = list(phases.keys())
phase_costs  = [phases[p]["cost"]/1e6 for p in phase_labels]   # in $M
phase_km     = [phases[p]["km"]       for p in phase_labels]
phase_colors = [phases[p]["color"]    for p in phase_labels]

bars = ax.bar(phase_labels, phase_costs, color=phase_colors, width=0.5)
ax2  = ax.twinx()
ax2.plot(phase_labels, phase_km, "o--", color="#2d6a9f", lw=2, ms=9, label="km upgraded")
ax2.set_ylabel("Network km Upgraded", color="#2d6a9f", fontsize=10)
ax2.tick_params(colors="#2d6a9f")
ax.set_title("Phased Investment Roadmap")
ax.set_ylabel("Estimated Investment ($M)")
for bar, v in zip(bars, phase_costs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"${v:.1f}M", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(phase_costs)*1.4)

plt.tight_layout()
plt.savefig(f"{OUT}/03_prescriptive.png", bbox_inches="tight")
plt.close()
print("\n  Saved → figures/03_prescriptive.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SENSITIVITY ANALYSIS – Impact of prioritization approach
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("6. SENSITIVITY ANALYSIS")
print("=" * 62)

# Compare three prioritization strategies by how much "Tier 1" (worst) gets addressed
strategies = {
    "Age-only"              : not_upgraded.nlargest(200, "age"),
    "Safety-tier-only"      : not_upgraded.nsmallest(200, "infra_tier"),
    "Combined (this model)" : not_upgraded.nlargest(200, "combined_priority"),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Sensitivity: Prioritization Strategy Comparison (Top 200 Segments)",
             fontsize=13, fontweight="bold")

for ax, (strat_name, subset) in zip(axes, strategies.items()):
    tier_counts = subset["infra_tier"].value_counts().sort_index()
    bars = ax.bar([TIER_LABELS[t] for t in tier_counts.index],
                  tier_counts.values,
                  color=[TIER_COLORS[t] for t in tier_counts.index])
    ax.set_title(strat_name, fontsize=11)
    ax.set_ylabel("Segments Selected"); ax.set_ylim(0, 160)
    ax.set_xticklabels([TIER_LABELS[t] for t in tier_counts.index],
                        rotation=20, ha="right", fontsize=8)
    low_count = sum(tier_counts.get(t,0) for t in [1,2])
    ax.text(0.5, 0.95, f"Low-safety selected: {low_count}",
            transform=ax.transAxes, ha="center", fontsize=9,
            fontweight="bold", color="red",
            bbox=dict(fc="white", boxstyle="round", alpha=0.8))

print("  Tier-1&2 segments selected by strategy:")
for strat_name, subset in strategies.items():
    low = (subset["infra_tier"] <= 2).sum()
    avg_age = subset["age"].mean()
    print(f"    {strat_name:<28} Low-safety={low:>3}  Avg age={avg_age:.1f} yrs")

plt.tight_layout()
plt.savefig(f"{OUT}/04_sensitivity.png", bbox_inches="tight")
plt.close()
print("\n  Saved → figures/04_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("FINAL SUMMARY")
print("=" * 62)
print(f"  Dataset          : {len(df):,} cycling segments, {df['length_m'].sum()/1000:.0f} km total")
print(f"  % upgraded       : {y.mean()*100:.1f}%  ({y.sum()} segments)")
print(f"  Best classifier  : {best_name}  (ROC-AUC = {best['roc_mean']:.3f})")
print(f"  CRITICAL band    : {band_counts.get('CRITICAL',0)} segments / {band_km.get('CRITICAL',0):.1f} km")
print(f"  HIGH band        : {band_counts.get('HIGH',0)} segments / {band_km.get('HIGH',0):.1f} km")
print(f"  Est. 2-yr invest : ${est_cost_critical:,.0f}  (CRITICAL segments)")
print(f"  Est. 4-yr invest : ${est_cost_critical+est_cost_high:,.0f}  (CRITICAL + HIGH)")
print("\nAll figures saved to ./figures/   Done ✓")
