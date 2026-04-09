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
├── toronto_cycling.py        ← Analysis (single file, fully self-contained)
├── cycling-network_-_4326.csv       ← Data CSV
│
└── figures/                         ← Auto-created when the script runs
    ├── 01_eda.png                   ← Exploratory data analysis (6-panel)
    ├── 02_diagnostics.png           ← Model diagnostics (6-panel)
    ├── 03_prescriptive.png          ← Prescriptive upgrade roadmap (4-panel)
    └── 04_sensitivity.png           ← Strategy sensitivity analysis (3-panel)
```

> **Note:** The `figures/` folder doesn't need to exist beforehand — the script creates it automatically.

---

## Setup & Installation

### Option A — Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook
2. Upload `toronto_cycling.py` and `cycling-network_-_4326.csv` using the 📁 sidebar
3. In the first cell, paste and run:
   ```python
   # Install missing library (Colab has the rest)
   !pip install xgboost -q
   ```
4. In the next cell, paste and run:
   ```python
   # Update the data path to match Colab's upload location
   exec(open("toronto_cycling.py").read().replace(
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
├── toronto_cycling.py
└── cycling-network_-_4326.csv
```

**Step 4 — Update the data path in the script**

Open `toronto_cycling.py` and change line ~30:
```python

# Windows:
DATA_PATH= "data/cycling-network_-_4326.csv"

# Mac/Linux:
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
python toronto_cycling.py
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
