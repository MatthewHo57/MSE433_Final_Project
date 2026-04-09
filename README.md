# MSE433 Final Project — Toronto Cycling Network Upgrade Prioritization

Predictive classification and prescriptive maintenance roadmap for the City of Toronto's cycling infrastructure, using the City of Toronto Open Data cycling network dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Reproducing All Figures & Results](#reproducing-all-figures--results)
- [How the Code Works](#how-the-code-works)
- [Output Reference](#output-reference)
- [Dependencies](#dependencies)

---

## Project Overview

Toronto's cycling network spans ~824 km across 1,523 segments, but **78% of segments have never been upgraded** since installation, with a mean age of ~15 years. Many early-2000s Sharrows and Signed Routes offer minimal separation from motor vehicle traffic and are misaligned with Vision Zero safety goals.

This project addresses two questions:

1. **Predict** — Which segments are least likely to receive an organic upgrade without policy intervention? *(binary classification)*
2. **Prescribe** — Given limited capital budgets, which segments should the City prioritize upgrading first, and what will it cost? *(priority scoring + phased roadmap)*

---

## Dataset

| Field | Value |
|-------|-------|
| **Name** | City of Toronto Cycling Network |
| **Source** | Toronto Open Data Portal |
| **URL** | https://open.toronto.ca/dataset/cycling-network/ |
| **File used** | `cycling-network_-_4326.csv` |
| **Records** | 1,538 segments (1,523 after cleaning) |
| **CRS** | EPSG:4326 (WGS84) |

**Key columns used:**

| Column | Description |
|--------|-------------|
| `INSTALLED` | Year the segment was installed |
| `UPGRADED` | Year the segment was last upgraded (blank if never) |
| `INFRA_HIGHORDER` | Infrastructure type (e.g. Bike Lane, Sharrows, Cycle Track) |
| `STREET_NAME` | Street the segment is on |
| `geometry` | GeoJSON MultiLineString — used to compute segment length and centroid |

---

## Folder Structure

```
mse433-toronto-cycling/
│
├── README.md                        ← You are here
├── mse433_toronto_cycling.py        ← Main analysis script (single file, fully self-contained)
├── MSE433_Toronto_Case_Report.md    ← Written case report (problem statement + summary)
│
├── data/
│   └── cycling-network_-_4326.csv   ← Place the dataset here (see Dataset section above)
│
└── figures/                         ← Auto-created when the script runs
    ├── 01_eda.png                   ← Exploratory data analysis (6-panel)
    ├── 02_diagnostics.png           ← Model diagnostics (6-panel)
    ├── 03_prescriptive.png          ← Prescriptive upgrade roadmap (4-panel)
    └── 04_sensitivity.png           ← Strategy sensitivity analysis (3-panel)
```

> **Note:** The `figures/` folder does not need to exist beforehand — the script creates it automatically.

---

## Setup & Installation

### Option A — Google Colab (recommended, no installation needed)

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook
2. Upload `mse433_toronto_cycling.py` and `cycling-network_-_4326.csv` using the 📁 sidebar
3. In the first cell, paste and run:
   ```python
   # Install missing library (Colab has the rest)
   !pip install xgboost -q
   ```
4. In the next cell, paste and run:
   ```python
   # Update the data path to match Colab's upload location
   exec(open("mse433_toronto_cycling.py").read().replace(
       'DATA_PATH= "/mnt/user-data/uploads/cycling-network_-_4326.csv"',
       'DATA_PATH= "/content/cycling-network_-_4326.csv"'
   ))
   ```
5. Figures will appear in the `/content/figures/` folder in the sidebar — right-click to download

---

### Option B — Local Python (Anaconda or pip)

**Step 1 — Install Python 3.9+**
Download from [python.org](https://www.python.org/downloads/) or install [Anaconda](https://www.anaconda.com/download).

**Step 2 — Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```
Or with conda:
```bash
conda install pandas numpy scikit-learn matplotlib seaborn
conda install -c conda-forge xgboost
```

**Step 3 — Set up the folder**
```
your-folder/
├── mse433_toronto_cycling.py
└── data/
    └── cycling-network_-_4326.csv
```

**Step 4 — Update the data path in the script**

Open `mse433_toronto_cycling.py` and change line ~30:
```python
# Before:
DATA_PATH= "/mnt/user-data/uploads/cycling-network_-_4326.csv"

# After (Windows):
DATA_PATH= "data/cycling-network_-_4326.csv"

# After (Mac/Linux):
DATA_PATH= "data/cycling-network_-_4326.csv"
```

---

### Option C — VS Code

1. Open the project folder in VS Code
2. Install the **Python** extension (from the Extensions panel)
3. Select your Python interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*)
4. Open a terminal in VS Code (`Ctrl+\``) and run `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
5. Update `DATA_PATH` as shown in Option B above
6. Click the **▶ Run** button in the top-right corner of the editor

---

## How to Run

Once setup is complete, run from the terminal:

```bash
python mse433_toronto_cycling.py
```

Expected runtime: **~60–90 seconds** (most time is spent on the 5-fold cross-validation).

You should see console output like:
```
==============================================================
1. LOAD & CLEAN
==============================================================
  Raw records : 1,538  |  Columns : 17
  After cleaning  : 1,523 segments
  Total network   : 823.8 km
  ...
==============================================================
4. PREDICTIVE MODELLING
==============================================================
  Logistic Regression    ROC-AUC=0.721 ...
  Random Forest          ROC-AUC=0.857 ...
  XGBoost                ROC-AUC=0.869 ...
  ★  Best model : XGBoost
...
All figures saved to ./figures/   Done ✓
```

---

## Reproducing All Figures & Results

All results are fully deterministic. A fixed random seed (`np.random.seed(42)`) is set at the top of the script, so every run produces **identical figures and numbers**.

| Figure | What it shows | How it's reproduced |
|--------|--------------|---------------------|
| `01_eda.png` | Installation trends, infrastructure type breakdown, upgrade rates by tier, age distributions, geographic dot map, time-to-upgrade histogram | Runs automatically in Section 3 of the script |
| `02_diagnostics.png` | ROC curves for all 3 models, confusion matrix, ROC-AUC bar chart, XGBoost feature importances, upgrade probability by tier, age vs. probability scatter | Runs automatically in Section 5 |
| `03_prescriptive.png` | Geographic priority map (colour = CRITICAL/HIGH/MEDIUM/LOW), priority score distribution, km-by-band stacked bar, phased investment timeline | Runs automatically in Section 7 |
| `04_sensitivity.png` | Side-by-side comparison of 3 prioritization strategies (age-only vs. safety-tier-only vs. combined model) for the top 200 segments | Runs automatically in Section 8 |

**To reproduce only the figures without re-training the models**, you can comment out the `cross_val_score` lines in Section 4 — the models will still train and predict, just without the CV loop, reducing runtime to ~10 seconds.

---

## How the Code Works

The script runs sequentially through 8 clearly labelled sections:

### Section 1 — Load & Clean
Reads the CSV, drops the 15 rows with missing installation years, and filters out placeholder upgrade years (values ≤ 2000 such as `1`). Segment lengths (in metres) and geographic centroids are computed from the embedded GeoJSON coordinates using the **Haversine formula**.

### Section 2 — Feature Engineering
Constructs 6 model features from the raw columns:

| Feature | How it's built | Why it matters |
|---------|---------------|----------------|
| `age` | `2024 − INSTALLED` | Older segments more likely to need upgrading |
| `infra_tier` | Ordinal mapping 1–5 (Sharrows → Multi-Use Trail) | Lower tier = lower safety = higher upgrade need |
| `length_m` | Haversine sum over coordinate pairs | Longer segments may face more upgrade complexity |
| `low_safety_old` | Binary: tier ≤ 2 AND age > 10 | Flags the most at-risk population |
| `centroid_lat/lon` | Mean of all coordinate points | Captures spatial clustering of upgrades |

The **binary target** is `has_upgraded`: 1 if the segment has a valid upgrade year recorded, 0 otherwise (78% of segments).

### Section 3 — Exploratory Data Analysis
Generates `01_eda.png` with 6 panels exploring the data from multiple angles: temporal trends, infrastructure composition, upgrade rates, age distributions, geography, and time-to-upgrade patterns.

### Section 4 — Predictive Modelling
Trains three classifiers using **5-fold stratified cross-validation** (stratified to handle the 22%/78% class imbalance). Primary metric is **ROC-AUC** (insensitive to class imbalance). XGBoost wins with ROC-AUC = 0.869.

```
Logistic Regression  →  ROC-AUC = 0.721  (baseline)
Random Forest        →  ROC-AUC = 0.857
XGBoost              →  ROC-AUC = 0.869  ★ best
```

### Section 5 — Model Diagnostics
Generates `02_diagnostics.png`. Key insight from feature importance: **segment age and infrastructure tier dominate**, followed by geographic location — suggesting upgrades have been spatially concentrated and that old low-tier segments in certain areas are systematically overlooked.

### Section 6 — Prescriptive Priority Scoring
Each unupgraded segment receives a **combined priority score**:

```
score = age × (6 − infra_tier) × (1.5 − P(upgrade)) × (length_km + 0.1)
```

- `age × (6 − infra_tier)` — rewards segments that are both old and low-safety
- `(1.5 − P(upgrade))` — up-weights segments the model predicts will *not* be upgraded organically
- `× length_km` — favours longer segments where an upgrade has more network impact

Scores are normalized 0–100 and classified into four bands at the **90th / 70th / 40th percentiles**:

| Band | # Segments | Network km |
|------|-----------|------------|
| CRITICAL | 120 | 207.7 km |
| HIGH | 243 | 171.2 km |
| MEDIUM | 387 | 126.8 km |
| LOW | 441 | 98.8 km |

### Section 7 — Prescriptive Visualisations
Generates `03_prescriptive.png`: a geographic priority map, score distribution histograms, a stacked bar of km-by-band broken down by infrastructure type, and a phased investment bar chart with a secondary axis showing km upgraded per phase.

### Section 8 — Sensitivity Analysis
Compares three alternative prioritization strategies on the top 200 segments:
- **Age-only** — selects oldest segments regardless of safety tier
- **Safety-tier-only** — selects lowest-tier segments regardless of age
- **Combined (this model)** — selects based on the full priority score

The combined approach captures 118 low-safety segments with an average age of 18.9 years — outperforming both single-dimension strategies.

---

## Output Reference

### Console output (key numbers)
```
Total network      : 823.8 km across 1,523 segments
Upgrade rate       : 21.8% (332 segments upgraded)
Median upgrade lag : 14 years (for segments that were upgraded)
Best model         : XGBoost  ROC-AUC = 0.869
CRITICAL segments  : 120 segments / 207.7 km
HIGH segments      : 243 segments / 171.2 km
Est. 2-yr invest   : ~$27.0M  (CRITICAL)
Est. 4-yr invest   : ~$45.8M  (CRITICAL + HIGH)
```

### Top 5 priority segments (CRITICAL band)
| Street | Installed | Tier | Age | Length |
|--------|-----------|------|-----|--------|
| Tommy Thompson Pk | 2001 | 2 | 23 yrs | 6,700 m |
| Broadway Ave | 2006 | 2 | 18 yrs | 3,298 m |
| Black Creek Trl | 2001 | 2 | 23 yrs | 1,876 m |
| Roselawn Ave | 2001 | 2 | 23 yrs | 1,850 m |
| Old Forest Hill Rd | 2005 | 2 | 19 yrs | 2,216 m |

---

## Dependencies

| Library | Version tested | Purpose |
|---------|---------------|---------|
| `pandas` | ≥ 1.5 | Data loading and manipulation |
| `numpy` | ≥ 1.23 | Numerical operations, Haversine formula |
| `scikit-learn` | ≥ 1.2 | Model training, cross-validation, metrics |
| `xgboost` | ≥ 1.7 | Best-performing classifier |
| `matplotlib` | ≥ 3.6 | All figure generation |
| `seaborn` | ≥ 0.12 | Heatmaps and distribution plots |

Install all at once:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```
