# 🚀 Project Structure Visualization

```
churn_prediction/
│
├── 📦 src/                          [Core Python Package]
│   ├── __init__.py                 
│   ├── config.py                   ⚙️  Configuration constants
│   ├── data_loader.py              📥 Load & prepare data
│   ├── preprocessing.py            🧹 Clean, balance, scale
│   ├── models.py                   🤖 Train & tune models
│   ├── evaluation.py               📊 Metrics & plots
│   ├── explainability.py           🔍 SHAP analysis
│   └── utils.py                    🛠️  Helper functions
│
├── 📂 scripts/
│   ├── run.sh                      🔄 Pipeline launcher
│   └── predict.py                  🎯 Inference script
│
├── 🔧 Configuration Files
│   ├── config.py                   (in src/)
│   ├── requirements_prod.txt       📋 Dependencies
│   └── .gitignore                  🚫 Skip patterns
│
├── 📚 Documentation
│   ├── README.md                   📖 User guide
│   └── PROJECT_STRUCTURE.txt       📋 Detailed structure
│
├── 🧬 Orchestration
│   ├── main.py                     ⚡ CLI entry point
│   └── train.ipynb                 📓 Notebook interface
│
└── 💾 Generated (after running)
    ├── best_model_gridsearch_rf.pkl
    ├── best_model_bayesian_rf.pkl
    ├── best_params_gridsearch.pkl
    ├── best_params_bayesian.pkl
    └── scaler.pkl

```

---

# 📋 Quick Reference

## Module Dependencies

```
main.py / train.ipynb
    ↓
├─→ data_loader.py (load CSV, encode features)
│   └─→ pandas, numpy
│
├─→ preprocessing.py (clean, balance, scale)
│   └─→ sklearn.preprocessing, imblearn.SMOTE
│
├─→ models.py (train, tune, save)
│   └─→ sklearn.ensemble, skopt.BayesSearchCV
│
├─→ evaluation.py (metrics, visualizations)
│   └─→ sklearn.metrics, matplotlib, seaborn
│
└─→ explainability.py (SHAP analysis)
    └─→ shap, TreeExplainer

```

## Execution Flow

```
1. data_loader.py      📥 Load WA_Fn-UseC_-Telco-Customer-Churn.csv
                           ↓
2. preprocessing.py    🧹 Remove outliers → Balance with SMOTE → Scale
                           ↓
3. models.py           🤖 GridSearch + Bayesian Tuning (parallel or sequential)
                           ↓
4. evaluation.py       📊 Compare models, plot confusion matrices & ROC
                           ↓
5. models.py           💾 Save pickle files (models + scaler)
                           ↓
6. explainability.py   🔍 SHAP feature importance analysis
                           ↓
✅ Done              Pipeline complete!
```

---

# 🎯 Running the Project

## Option 1: Python Script (Fastest)
```bash
python main.py
```

## Option 2: Shell Script
```bash
chmod +x scripts/run.sh
bash scripts/run.sh
```

## Option 3: Jupyter Notebook (Interactive)
```bash
jupyter notebook train.ipynb
# Run cells sequentially
```

## Option 4: Make Predictions
```bash
python scripts/predict.py --data new_customers.csv --model bayesian
```

---

# 🎓 Learning Path

For beginners understanding this structure:

1. **Start here**: `README.md` - Overview & quick start
2. **Config**: `src/config.py` - All constants in one place
3. **Flow**: `main.py` - See how modules connect
4. **Deep dive**: `src/` - Study each module independently
5. **Run it**: `python main.py` - See results
6. **Extend it**: Add new functions following the pattern

---

# ✅ Production Checklist

- [x] Modular design (separate concerns)
- [x] Configuration management (constants centralized)
- [x] No data leakage (SMOTE post-split)
- [x] Model persistence (pickle files)
- [x] Error handling (try-except blocks)
- [x] Logging (print statements)
- [x] Documentation (docstrings + README)
- [x] Reproducibility (fixed seeds)
- [x] Explainability (SHAP analysis)
- [x] Inference ready (`predict.py`)

---

# 📊 Model Files Generated

After running `main.py`, you get:

| File | Description |
|------|-------------|
| `best_model_gridsearch_rf.pkl` | GridSearch tuned Random Forest |
| `best_model_bayesian_rf.pkl` | Bayesian tuned Random Forest |
| `best_params_gridsearch.pkl` | GridSearch best hyperparameters |
| `best_params_bayesian.pkl` | Bayesian best hyperparameters |
| `scaler.pkl` | Fitted StandardScaler for new data |

Use these for production inference without retraining!

---

Created: Production-Ready Python ML Project
Status: ✅ Ready to Deploy
