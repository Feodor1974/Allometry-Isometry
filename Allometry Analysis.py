#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allometry Analysis V2 — 28-point structure
==========================================
Adapted for interactive file input.
Structure:
- 4 groups: Roach_I (5 ages), Roach_II (9 ages), Perch_I (6 ages), Perch_II (8 ages)
- 28 points total = age-class means (BL, Theta) averaged over 20 fish
- 20 regressions (4 groups × 5 theta)
- 15 ANCOVA comparisons (3 pairs × 5 theta) + bootstrap
- Power analysis (MDE)
- Three-level criterion for slopes
- All output in English
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')
import os

# ============================================================
# 0. INTERACTIVE FILE NAME INPUT
# ============================================================
FILE_NAME = input("Введите имя файла с данными (например, Data_FFF.xlsx): ").strip()
if not os.path.isfile(FILE_NAME):
    print(f"Файл '{FILE_NAME}' не найден. Проверьте имя и попробуйте снова.")
    exit(1)

# ============================================================
# 1. READ AND PARSE DATA FILE
# ============================================================

df_raw = pd.read_excel(FILE_NAME, sheet_name='Sheet 1', header=None)

# Row ranges for each parameter (20 fish each)
# ADJUSTED: all starts shifted -1 compared to original Data_F.xlsx
param_row_ranges = {
    'BL':  (2, 22),   # unchanged
    'Ed':  (23, 43),  # was 24 -> 23
    'Pod': (44, 64),  # was 45 -> 44
    'P1l': (65, 85),  # was 66 -> 65
    'Bd':  (86, 106), # was 87 -> 86
    'Cpd': (107, 127),# was 108 -> 107
}

# Column blocks for each group (indices are 0-based)
blocks = {
    'Roach_I':  {'cols': list(range(1, 6)),   'ages': ['a','b','c','d','e'], 'species':'Roach', 'stage':'I'},
    'Roach_II': {'cols': list(range(7, 16)),   'ages': ['1','2','3','4','5','6','7','8','9'], 'species':'Roach', 'stage':'II'},
    'Perch_I':  {'cols': list(range(17, 23)),  'ages': ['a','b','c','d','e','f'], 'species':'Perch', 'stage':'I'},
    'Perch_II': {'cols': list(range(24, 32)),  'ages': ['1','2','3','4','5','6','7','8'], 'species':'Perch', 'stage':'II'},
}

# Build long-form data (560 rows = 20 fish × 28 ages)
all_data = []
fish_id_counter = 0

for group_name, block_info in blocks.items():
    cols = block_info['cols']
    ages = block_info['ages']
    species = block_info['species']
    stage = block_info['stage']
    
    for fish_idx in range(20):
        fish_id = fish_id_counter + fish_idx
        for age_idx, age in enumerate(ages):
            col = cols[age_idx]
            row_data = {
                'Group': group_name,
                'Species': species,
                'Stage': stage,
                'Fish_ID': fish_id,
                'Age': age,
            }
            for param, (start, end) in param_row_ranges.items():
                row_data[param] = df_raw.iloc[start + fish_idx, col]
            all_data.append(row_data)
    
    fish_id_counter += 20

df_long = pd.DataFrame(all_data)

# ============================================================
# 2. COMPUTE THETA = Parameter / BL
# ============================================================

theta_params = ['Ed', 'Pod', 'P1l', 'Bd', 'Cpd']
for param in theta_params:
    df_long[f'Theta_{param}'] = df_long[param] / df_long['BL']

# ============================================================
# 3. BUILD 28-POINT DATASET (means by Age within Group)
# ============================================================

agg_dict = {'BL': 'mean'}
for param in theta_params:
    agg_dict[f'Theta_{param}'] = 'mean'
    agg_dict[param] = 'mean'

df_28 = df_long.groupby(['Group', 'Species', 'Stage', 'Age']).agg(agg_dict).reset_index()

# Add sample sizes
n_counts = df_long.groupby(['Group', 'Age']).size().reset_index(name='n')
df_28 = df_28.merge(n_counts, on=['Group', 'Age'])

# Order ages properly (corrected to match new file order)
age_orders = {
    'Roach_I': ['a','b','c','d','e'],
    'Roach_II': ['1','2','3','4','5','6','7','8','9'],
    'Perch_I': ['a','b','c','d','e','f'],
    'Perch_II': ['1','2','3','4','5','6','7','8'],
}
df_28['Age_Order'] = df_28.apply(lambda x: age_orders[x['Group']].index(x['Age']), axis=1)
df_28 = df_28.sort_values(['Group', 'Age_Order']).reset_index(drop=True)

# Create pooled species data for between-species ANCOVA
df_pooled = df_28.copy()
df_pooled.loc[df_pooled['Group'].isin(['Roach_I', 'Roach_II']), 'Pooled_Group'] = 'Roach_All'
df_pooled.loc[df_pooled['Group'].isin(['Perch_I', 'Perch_II']), 'Pooled_Group'] = 'Perch_All'

# ============================================================
# 4. REGRESSION RESULTS: 20 models (4 groups × 5 theta)
# Theta ~ BL on 28 points (5-9 points per regression)
# ============================================================

def run_regression(df, group, theta_param):
    """Run Theta ~ BL regression with diagnostics"""
    sub = df[df['Group'] == group].copy().sort_values('Age_Order')
    
    X = sm.add_constant(sub['BL'])
    y = sub[theta_param]
    model = sm.OLS(y, X).fit()
    
    # Diagnostics
    residuals = model.resid
    bp_test = het_breuschpagan(residuals, X)
    bp_stat, bp_pvalue, _, _ = bp_test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    return {
        'Group': group,
        'Theta': theta_param,
        'n': len(sub),
        'Intercept': model.params[0],
        'Slope': model.params[1],
        'SE_Slope': model.bse[1],
        't_Slope': model.tvalues[1],
        'p_Slope': model.pvalues[1],
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj,
        'F_stat': model.fvalue,
        'F_pvalue': model.f_pvalue,
        'RMSE': np.sqrt(model.mse_resid),
        'AIC': model.aic,
        'BIC': model.bic,
        'BP_Stat': bp_stat,
        'BP_pvalue': bp_pvalue,
        'SW_Stat': shapiro_stat,
        'SW_pvalue': shapiro_p,
    }

regression_results = []
for group in ['Roach_I', 'Roach_II', 'Perch_I', 'Perch_II']:
    for theta in ['Theta_Ed', 'Theta_Pod', 'Theta_P1l', 'Theta_Bd', 'Theta_Cpd']:
        regression_results.append(run_regression(df_28, group, theta))

df_regression = pd.DataFrame(regression_results)

# ============================================================
# 5. THREE-LEVEL CRITERION FOR SLOPES
# ============================================================

def classify_slope(row):
    """Three-level criterion: =0, ≠0, ≠ between groups"""
    p = row['p_Slope']
    slope = row['Slope']
    if p >= 0.05:
        return 'Not different from zero'
    else:
        return 'Different from zero'

df_regression['Slope_Criterion'] = df_regression.apply(classify_slope, axis=1)

# ============================================================
# 6. POWER ANALYSIS (MDE)
# ============================================================

def compute_mde(n, se_slope, alpha=0.05, power=0.80):
    """Minimum Detectable Effect for slope"""
    from scipy.stats import t as t_dist
    df = n - 2
    t_alpha = t_dist.ppf(1 - alpha/2, df)
    t_power = t_dist.ppf(power, df)
    mde = (t_alpha + t_power) * se_slope
    return mde, t_alpha, t_power

power_results = []
for _, r in df_regression.iterrows():
    mde, t_a, t_p = compute_mde(r['n'], r['SE_Slope'])
    f2 = r['R2'] / (1 - r['R2']) if r['R2'] < 1 else np.inf
    
    power_results.append({
        'Group': r['Group'],
        'Theta': r['Theta'],
        'n': r['n'],
        'Alpha': 0.05,
        'Target_Power': 0.80,
        't_critical': round(t_a, 3),
        't_power': round(t_p, 3),
        'SE_Slope': round(r['SE_Slope'], 6),
        'MDE_Slope': round(mde, 6),
        'Observed_Slope': round(r['Slope'], 6),
        'Detectable': 'Yes' if abs(r['Slope']) > mde else 'No',
        'R2': round(r['R2'], 4),
        'Cohens_f2': round(f2, 4) if np.isfinite(f2) else 'Inf',
    })

df_power = pd.DataFrame(power_results)

# ============================================================
# 7. ANCOVA: 15 comparisons (3 pairs × 5 theta)
# ============================================================

def ancova_test(df, group1, group2, theta_param):
    """ANCOVA: Test equality of slopes (interaction: Group × BL)"""
    sub = df[df['Group'].isin([group1, group2])].copy()
    sub['Group_code'] = (sub['Group'] == group2).astype(int)
    
    # Full model with interaction
    X_full = sm.add_constant(sub[['BL', 'Group_code']])
    X_full['BL_x_Group'] = sub['BL'] * sub['Group_code']
    y = sub[theta_param]
    model_full = sm.OLS(y, X_full).fit()
    
    # Reduced model (parallel slopes)
    X_red = sm.add_constant(sub[['BL', 'Group_code']])
    model_red = sm.OLS(y, X_red).fit()
    
    # F-test for interaction
    anova_table = sm.stats.anova_lm(model_red, model_full)
    f_stat = anova_table['F'][1]
    f_pvalue = anova_table['Pr(>F)'][1]
    
    # Individual slopes
    sub1 = sub[sub['Group'] == group1]
    sub2 = sub[sub['Group'] == group2]
    m1 = sm.OLS(sub1[theta_param], sm.add_constant(sub1['BL'])).fit()
    m2 = sm.OLS(sub2[theta_param], sm.add_constant(sub2['BL'])).fit()
    
    return {
        'Comparison': f"{group1} vs {group2}",
        'Theta': theta_param,
        'n1': len(sub1),
        'n2': len(sub2),
        'Slope1': m1.params[1],
        'SE1': m1.bse[1],
        'p1': m1.pvalues[1],
        'Slope2': m2.params[1],
        'SE2': m2.bse[1],
        'p2': m2.pvalues[1],
        'Slope_Diff': m2.params[1] - m1.params[1],
        'F_Interaction': f_stat,
        'p_Interaction': f_pvalue,
        'R2_Full': model_full.rsquared,
    }

ancova_results = []

# Within-species comparisons
for g1, g2 in [('Roach_I', 'Roach_II'), ('Perch_I', 'Perch_II')]:
    for theta in ['Theta_Ed', 'Theta_Pod', 'Theta_P1l', 'Theta_Bd', 'Theta_Cpd']:
        ancova_results.append(ancova_test(df_28, g1, g2, theta))

# Between-species comparison
for theta in ['Theta_Ed', 'Theta_Pod', 'Theta_P1l', 'Theta_Bd', 'Theta_Cpd']:
    sub = df_pooled.copy()
    sub['Group_code'] = (sub['Pooled_Group'] == 'Perch_All').astype(int)
    
    X_full = sm.add_constant(sub[['BL', 'Group_code']])
    X_full['BL_x_Group'] = sub['BL'] * sub['Group_code']
    y = sub[theta]
    model_full = sm.OLS(y, X_full).fit()
    
    X_red = sm.add_constant(sub[['BL', 'Group_code']])
    model_red = sm.OLS(y, X_red).fit()
    
    anova_table = sm.stats.anova_lm(model_red, model_full)
    f_stat = anova_table['F'][1]
    f_pvalue = anova_table['Pr(>F)'][1]
    
    roach = sub[sub['Pooled_Group'] == 'Roach_All']
    perch = sub[sub['Pooled_Group'] == 'Perch_All']
    m_r = sm.OLS(roach[theta], sm.add_constant(roach['BL'])).fit()
    m_p = sm.OLS(perch[theta], sm.add_constant(perch['BL'])).fit()
    
    ancova_results.append({
        'Comparison': 'Roach_All vs Perch_All',
        'Theta': theta,
        'n1': len(roach),
        'n2': len(perch),
        'Slope1': m_r.params[1],
        'SE1': m_r.bse[1],
        'p1': m_r.pvalues[1],
        'Slope2': m_p.params[1],
        'SE2': m_p.bse[1],
        'p2': m_p.pvalues[1],
        'Slope_Diff': m_p.params[1] - m_r.params[1],
        'F_Interaction': f_stat,
        'p_Interaction': f_pvalue,
        'R2_Full': model_full.rsquared,
    })

df_ancova = pd.DataFrame(ancova_results)

# ============================================================
# 8. BOOTSTRAP FOR ANCOVA (2,000 iterations)
# ============================================================

def bootstrap_ancova(sub1, sub2, theta_param, n_bootstrap=2000, seed=42):
    """Bootstrap confidence intervals for slope difference"""
    np.random.seed(seed)
    n1, n2 = len(sub1), len(sub2)
    bl1, th1 = sub1['BL'].values, sub1[theta_param].values
    bl2, th2 = sub2['BL'].values, sub2[theta_param].values
    
    m1 = sm.OLS(th1, sm.add_constant(bl1)).fit()
    m2 = sm.OLS(th2, sm.add_constant(bl2)).fit()
    obs_diff = m2.params[1] - m1.params[1]
    
    boot_diffs = []
    for _ in range(n_bootstrap):
        i1 = np.random.choice(n1, n1, replace=True)
        i2 = np.random.choice(n2, n2, replace=True)
        try:
            s1 = sm.OLS(th1[i1], sm.add_constant(bl1[i1])).fit()
            s2 = sm.OLS(th2[i2], sm.add_constant(bl2[i2])).fit()
            boot_diffs.append(s2.params[1] - s1.params[1])
        except:
            pass
    
    boot_diffs = np.array(boot_diffs)
    ci_l, ci_u = np.percentile(boot_diffs, [2.5, 97.5])
    pval = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    
    return obs_diff, ci_l, ci_u, pval, np.std(boot_diffs)

bootstrap_results = []

# Within-species
for g1, g2 in [('Roach_I', 'Roach_II'), ('Perch_I', 'Perch_II')]:
    sub1 = df_28[df_28['Group'] == g1]
    sub2 = df_28[df_28['Group'] == g2]
    for theta in ['Theta_Ed', 'Theta_Pod', 'Theta_P1l', 'Theta_Bd', 'Theta_Cpd']:
        obs, ci_l, ci_u, pval, se = bootstrap_ancova(sub1, sub2, theta)
        bootstrap_results.append({
            'Comparison': f"{g1} vs {g2}", 'Theta': theta,
            'Observed_Diff': round(obs, 6), 'Boot_CI_Lower': round(ci_l, 6),
            'Boot_CI_Upper': round(ci_u, 6), 'Boot_pvalue': round(pval, 4),
            'Boot_SE': round(se, 6), 'Significant': 'Yes' if pval < 0.05 else 'No',
        })

# Between-species
roach = df_pooled[df_pooled['Pooled_Group'] == 'Roach_All']
perch = df_pooled[df_pooled['Pooled_Group'] == 'Perch_All']
for theta in ['Theta_Ed', 'Theta_Pod', 'Theta_P1l', 'Theta_Bd', 'Theta_Cpd']:
    obs, ci_l, ci_u, pval, se = bootstrap_ancova(roach, perch, theta)
    bootstrap_results.append({
        'Comparison': 'Roach_All vs Perch_All', 'Theta': theta,
        'Observed_Diff': round(obs, 6), 'Boot_CI_Lower': round(ci_l, 6),
        'Boot_CI_Upper': round(ci_u, 6), 'Boot_pvalue': round(pval, 4),
        'Boot_SE': round(se, 6), 'Significant': 'Yes' if pval < 0.05 else 'No',
    })

df_bootstrap = pd.DataFrame(bootstrap_results)

# ============================================================
# 9. DESCRIPTIVE STATISTICS (28 age-class means)
# ============================================================

df_descriptive = df_28[['Group', 'Age', 'n', 'BL'] + [f'Theta_{p}' for p in theta_params]].copy()
for col in ['BL'] + [f'Theta_{p}' for p in theta_params]:
    df_descriptive[col] = df_descriptive[col].round(6)

# Group-level summary
group_summary = df_28.groupby('Group').agg({
    'BL': ['mean', 'std', 'min', 'max'],
    'Theta_Ed': ['mean', 'std'],
    'Theta_Pod': ['mean', 'std'],
    'Theta_P1l': ['mean', 'std'],
    'Theta_Bd': ['mean', 'std'],
    'Theta_Cpd': ['mean', 'std'],
}).round(6)

# ============================================================
# 10. SAVE ALL RESULTS TO EXCEL
# ============================================================

output_name = FILE_NAME.replace('.xlsx', '_Allometry_Results_V2.xlsx')
with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
    
    # Sheet 1: 28-point data
    df_28[['Group', 'Species', 'Stage', 'Age', 'n', 'BL', 
           'Theta_Ed', 'Theta_Pod', 'Theta_P1l', 'Theta_Bd', 'Theta_Cpd']].to_excel(
        writer, sheet_name='Age_Class_Means', index=False)
    
    # Sheet 2: Regression Results (20 models)
    df_regression.to_excel(writer, sheet_name='Regression_Results', index=False)
    
    # Sheet 3: Power Analysis (MDE)
    df_power.to_excel(writer, sheet_name='Power_Analysis', index=False)
    
    # Sheet 4: ANCOVA Comparisons
    df_ancova.to_excel(writer, sheet_name='ANCOVA_Comparisons', index=False)
    
    # Sheet 5: Bootstrap ANCOVA
    df_bootstrap.to_excel(writer, sheet_name='Bootstrap_ANCOVA', index=False)
    
    # Sheet 6: Descriptive Statistics
    df_descriptive.to_excel(writer, sheet_name='Descriptive_Stats', index=False)
    
    # Sheet 7: Group Summary
    group_summary.to_excel(writer, sheet_name='Group_Summary')

print("=" * 60)
print("ALLOMETRY ANALYSIS V2 COMPLETE")
print("=" * 60)
print(f"Input file:  {FILE_NAME}")
print(f"Output file: {output_name}")
print(f"\nStructure:")
print(f"  - Age_Class_Means:   28 rows (age-class means)")
print(f"  - Regression:        20 models (4 groups × 5 theta)")
print(f"  - Power Analysis:    MDE for each model")
print(f"  - ANCOVA:            15 comparisons")
print(f"  - Bootstrap:         2,000 iterations per comparison")
print(f"  - Descriptive:       28 rows + group summary")
print(f"\nThree-level slope criterion applied:")
print(df_regression[['Group', 'Theta', 'Slope', 'p_Slope', 'Slope_Criterion']].to_string())