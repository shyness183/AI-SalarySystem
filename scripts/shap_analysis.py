#!/usr/bin/env python3
"""
scripts/shap_analysis.py
Generate SHAP visualization figures for thesis
Output: models
Run: python scripts/shap_analysis.py
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
PROC_DIR   = "data/processed"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature name mapping (English display) ────────────────────────────────
FEATURE_NAME_MAP = {
    'exp_level_encoded':      'Experience Level',
    'years_experience':       'Years of Experience',
    'size_encoded':           'Company Size',
    'edu_level_encoded':      'Education Required',
    'remote_ratio':           'Remote Ratio',
    'skill_count':            'Skill Count',
    'job_description_length': 'JD Length',
    'benefits_score':         'Benefits Score',
    'days_to_deadline':       'Days to Deadline',
    'post_month':             'Post Month',
    'post_quarter':           'Post Quarter',
    'post_year':              'Post Year',
    'deadline_month':         'Deadline Month',
    'deadline_year':          'Deadline Year',
}

def map_feature_name(name):
    if name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[name]
    if name.startswith('ctry_'):
        return 'Country: ' + name[5:].title()
    if name.startswith('ind_'):
        return 'Industry: ' + name[4:]
    if name.startswith('emp_'):
        return 'EmpType: ' + name[4:]
    if name.startswith('skill_emb_'):
        return f'SkillVec[{name[10:]}]'
    return name

# ── 1. Load model and data ─────────────────────────────────────────────────
print("Loading model and data...")
model  = joblib.load(f"{MODEL_DIR}/xgboost_optimized.pkl")
X_test = pd.read_parquet(f"{PROC_DIR}/X_test_w2v.parquet")
y_test = pd.read_parquet(f"{PROC_DIR}/y_test.parquet").squeeze()

if 'skills_list' in X_test.columns:
    X_test = X_test.drop(columns=['skills_list'])

print(f"  Test set shape: {X_test.shape}")

# ── 2. Compute SHAP values ─────────────────────────────────────────────────
print("Computing SHAP values (TreeExplainer)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(f"  SHAP matrix shape: {shap_values.shape}")

display_names = [map_feature_name(c) for c in X_test.columns]
X_display     = X_test.copy()
X_display.columns = display_names

shap_importance = np.abs(shap_values).mean(axis=0)

# ── 3. Figure 1: Global Feature Importance Bar Chart (Top 20) ─────────────
print("Generating Figure 1: Feature importance bar chart...")
importance_df = pd.DataFrame({
    'feature':    display_names,
    'importance': shap_importance
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    importance_df['feature'][::-1],
    importance_df['importance'][::-1],
    color='#4472C4', alpha=0.85
)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_title('Global Feature Importance (SHAP) — Top 20', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, importance_df['importance'][::-1]):
    ax.text(bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_importance_bar.png")

# ── 4. Figure 2: Beeswarm Plot (Top 15) ───────────────────────────────────
print("Generating Figure 2: Beeswarm plot...")
top15_idx  = np.argsort(shap_importance)[::-1][:15]
shap_top15 = shap_values[:, top15_idx]
X_top15    = X_display.iloc[:, top15_idx]

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_top15, X_top15,
    plot_type='dot',
    max_display=15,
    show=False,
    color_bar_label='Feature value'
)
plt.title('SHAP Beeswarm Plot — Top 15 Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_beeswarm.png")

# ── 5. Figure 3: Dependence Plot — Experience Level ───────────────────────
print("Generating Figure 3: Experience level dependence plot...")
exp_idx = list(X_test.columns).index('exp_level_encoded')

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    X_test.iloc[:, exp_idx],
    shap_values[:, exp_idx],
    c=X_test.iloc[:, exp_idx],
    cmap='RdYlGn', alpha=0.4, s=8
)
ax.set_xlabel('Experience Level  (0=EN · 1=MI · 2=SE · 3=EX)', fontsize=11)
ax.set_ylabel('SHAP Value  (impact on log salary)', fontsize=11)
ax.set_title('SHAP Dependence Plot — Experience Level', fontsize=13, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['EN\n(Entry)', 'MI\n(Mid)', 'SE\n(Senior)', 'EX\n(Expert)'])
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_dependence_exp.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_dependence_exp.png")

# ── 6. Figure 4: Dependence Plot — Years of Experience ────────────────────
print("Generating Figure 4: Years of experience dependence plot...")
yr_idx = list(X_test.columns).index('years_experience')

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    X_test.iloc[:, yr_idx],
    shap_values[:, yr_idx],
    c=X_test.iloc[:, exp_idx],   # color by experience level
    cmap='RdYlGn', alpha=0.35, s=8
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Experience Level', fontsize=10)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(['EN', 'MI', 'SE', 'EX'])
ax.set_xlabel('Years of Experience', fontsize=11)
ax.set_ylabel('SHAP Value  (impact on log salary)', fontsize=11)
ax.set_title('SHAP Dependence Plot — Years of Experience', fontsize=13, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_dependence_years.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/shap_dependence_years.png")

# ── 7. Save SHAP summary data for Flask ───────────────────────────────────
print("Saving SHAP summary data...")
importance_df_full = pd.DataFrame({
    'feature':     display_names,
    'feature_raw': list(X_test.columns),
    'importance':  shap_importance
}).sort_values('importance', ascending=False)
importance_df_full.to_csv(f"{PROC_DIR}/shap_importance.csv", index=False)
print(f"  Saved: {PROC_DIR}/shap_importance.csv")

# ── 8. Console summary ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SHAP analysis complete!")
print(f"Figures saved to: {OUTPUT_DIR}/")
print("  shap_importance_bar.png   — bar chart  (thesis Fig 6.x)")
print("  shap_beeswarm.png         — beeswarm   (thesis Fig 6.x)")
print("  shap_dependence_exp.png   — exp level  (thesis Fig 6.x)")
print("  shap_dependence_years.png — years exp  (thesis Fig 6.x)")
print("=" * 55)

print("\nTop 10 SHAP Feature Importance:")
print(importance_df_full.head(10)[['feature', 'importance']].to_string(index=False))