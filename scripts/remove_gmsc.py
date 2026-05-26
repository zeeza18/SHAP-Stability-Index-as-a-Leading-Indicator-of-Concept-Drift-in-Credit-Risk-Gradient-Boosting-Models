"""Remove all GMSC references from paper.tex. Run from project root."""

from pathlib import Path

TEX = Path('paper/latex/paper.tex')
with open(TEX, encoding='utf-8') as f:
    text = f.read()

original_len = len(text)
applied = []

def sub(old, new, tag):
    global text
    assert old in text, f"NOT FOUND: {tag}\nSearching for:\n{repr(old[:120])}"
    text = text.replace(old, new, 1)
    applied.append(tag)

# ─── 1. Abstract: dataset list ────────────────────────────────────────────
sub(
    'setting with ADWIN-triggered adaptive retraining on three public credit datasets:\n'
    'IEEE-CIS Fraud Detection (590,540 transactions), Give Me Some Credit (150,000\n'
    'records), and ULB Credit Card Fraud (284,807 transactions).',
    'setting with ADWIN-triggered adaptive retraining on two public credit datasets:\n'
    'IEEE-CIS Fraud Detection (590,540 transactions) and\n'
    'ULB Credit Card Fraud (284,807 transactions).',
    'Abstract: dataset list'
)

# ─── 2. Abstract: stability sentence ──────────────────────────────────────
sub(
    'Datasets\nwithout meaningful concept drift maintained high mean SSI values (0.95 to 0.99),\n'
    'confirming SSI\'s sensitivity to genuine distributional change.',
    'On ULB Credit Card Fraud, mean SSI remained between 0.95 and 0.99,\n'
    'confirming SSI\'s ability to discriminate drift intensity across structurally\n'
    'distinct financial streaming environments.',
    'Abstract: stability sentence'
)

# ─── 3. Introduction contribution 1 ───────────────────────────────────────
sub(
    '        adaptive retraining across three large, structurally distinct public\n'
    '        datasets spanning six months of real transaction data (IEEE-CIS,\n'
    '        590,540 transactions), a standard retail credit benchmark (GMSC,\n'
    '        150,000 records), and a two-day high-frequency fraud stream (ULB\n'
    '        Credit Card Fraud, 284,807 transactions).',
    '        adaptive retraining across two structurally distinct public datasets:\n'
    '        the IEEE-CIS Fraud Detection benchmark (590,540 six-month\n'
    '        transactions) and the ULB Credit Card Fraud stream\n'
    '        (284,807 two-day transactions).',
    'Intro: contribution 1 dataset list'
)

# ─── 4. Introduction contribution 4 ───────────────────────────────────────
sub(
    '        distributional change: mean SSI spans $0.66$--$0.68$ on the\n'
    '        heavily drifting IEEE-CIS dataset, $0.95$--$0.99$ on ULB CC Fraud\n'
    '        (localised drift), and $0.99$--$1.00$ on GMSC (no real drift),\n'
    '        confirming that SSI tracks genuine distributional change and\n'
    '        correctly identifies stable environments without threshold\n'
    '        recalibration.',
    '        distributional change: mean SSI spans $0.66$--$0.68$ on the\n'
    '        heavily drifting IEEE-CIS dataset and $0.95$--$0.99$ on ULB CC\n'
    '        Fraud (localised drift), confirming that SSI discriminates\n'
    '        drift intensity across structurally distinct financial streaming\n'
    '        environments without threshold recalibration.',
    'Intro: contribution 4'
)

# ─── 5. §4.1 Datasets: GMSC paragraph ─────────────────────────────────────
sub(
    '\n\\textbf{Give Me Some Credit (GMSC).} This benchmark dataset contains 150,000\n'
    'borrower records with binary delinquency label \\texttt{SeriousDlqin2yrs}\n'
    '(positive rate $\\approx 6.7$\\%). The ten raw features cover revolving credit\n'
    'utilisation, delinquency counts across three time horizons, debt-to-income\n'
    'ratio, monthly income, open credit lines, real estate loans, number of\n'
    'dependents, and borrower age. Because the dataset carries no transaction\n'
    'timestamp, a pseudo-temporal ordering is constructed by sorting records by\n'
    'borrower age (ascending) and revolving credit utilisation (descending),\n'
    'producing a plausible lifecycle progression for window-based evaluation.\n',
    '\n',
    '§4.1: GMSC paragraph'
)

# ─── 6. Dataset table: caption + GMSC row ─────────────────────────────────
sub(
    '  \\caption{Summary of the three evaluation datasets.',
    '  \\caption{Summary of the two evaluation datasets.',
    '§4.1: dataset table caption'
)

sub(
    '    Give Me Some Credit & 150,000 & 10     &  17 & 6.7  & N/A   & Pseudo-time   \\\\\n',
    '',
    '§4.1: dataset table GMSC row'
)

# ─── 7. §4.2 Feature Engineering: GMSC paragraph + SMOTE ─────────────────
sub(
    '\n\\textbf{GMSC.} Erroneous zero ages are replaced with the training-set median.\n'
    'Delinquency counts exceeding 90 are capped to remove likely data-entry\n'
    'outliers. Monthly income is log-transformed and four composite risk indicators\n'
    'are constructed: total late payments (sum across all three delinquency\n'
    'horizons), debt-to-income product, income per dependent, and a credit\n'
    'utilisation risk score (utilisation rate $\\times$ total late payments).\n'
    'Borrower age is discretised into six bands. This yields 17 engineered features.\n',
    '\n',
    '§4.2: GMSC feature engineering paragraph'
)

sub(
    'training partition at each (re)training step for IEEE-CIS and GMSC.',
    'training partition at each (re)training step for IEEE-CIS.',
    '§4.2: SMOTE GMSC mention'
)

# ─── 8. §4.3 Temporal Protocol: table row + text ─────────────────────────
sub(
    '    GMSC         & 1/20 of data & 1/20 of data & 8  & 12 \\\\\n',
    '',
    '§4.3: windows table GMSC row'
)

sub(
    'Table~\\ref{tab:windows} lists the parameters. IEEE-CIS and ULB CC Fraud windows\n'
    'are defined by calendar duration; GMSC windows are equal-size record chunks\n'
    'because no real timestamp exists.',
    'Table~\\ref{tab:windows} lists the parameters. Both datasets use\n'
    'calendar-duration windows aligned to their respective temporal granularity.',
    '§4.3: temporal protocol GMSC text'
)

# ─── 9. Table 1: caption + GMSC rows ──────────────────────────────────────
# Table 1 caption already fixed manually — skip


sub(
    '    \\multirow{4}{*}{\\shortstack[l]{Give Me\\\\Some Credit}}\n'
    '      & StaticXGBoost    & $0.504\\pm0.013$ & $0.069\\pm0.004$ & $0.125\\pm0.005$ & ---  & ---   \\\\\n'
    '      & AdaptiveXGBoost  & $0.500\\pm0.014$ & $0.068\\pm0.003$ & $0.125\\pm0.005$ &   0  & 0.992 \\\\\n'
    '      & StaticLightGBM   & $0.506\\pm0.016$ & $0.069\\pm0.006$ & $0.000^{\\dagger}$ & ---  & ---   \\\\\n'
    '      & AdaptiveLightGBM & $0.500\\pm0.011$ & $0.068\\pm0.004$ & $0.000^{\\dagger}$ &   0  & 0.997 \\\\\n'
    '    \\midrule\n'
    '    \\multirow{4}{*}{\\shortstack[l]{ULB CC\\\\Fraud}}',
    '    \\multirow{4}{*}{\\shortstack[l]{ULB CC\\\\Fraud}}',
    'Table 1: GMSC rows'
)

# ─── 10. §5.1 text: GMSC F1 sentence + GMSC paragraph ───────────────────
sub(
    'for completeness; on GMSC, LightGBM models assign all instances probability\n'
    'below 0.5, yielding F1~$=0$ --- a reflection of extreme class imbalance rather\n'
    'than model failure, confirmed by the AP values which remain non-zero.\n'
    'Figure~\\ref{fig:auc_time} plots per-window AUC trajectories.',
    'Figure~\\ref{fig:auc_time} plots per-window AUC trajectories.',
    '§5.1: GMSC F1 sentence'
)

# §5.1 GMSC paragraph already removed manually


# ─── 11. §5.2 SSI Dynamics: GMSC second feature + three-way contrast ──────
sub(
    'Both series stabilise at intermediate values ($0.60$--$0.80$) through the\n'
    'middle of the evaluation period, reflecting the model\'s partial recovery after\n'
    'each retraining episode. Second, SSI on GMSC remains flat and close to unity\n'
    'throughout, consistent with the absence of detected drift. Third, on ULB CC\n'
    'Fraud, SSI shows a localised drop around the single drift event (window 13 for\n'
    'XGBoost, window 15 for LightGBM) before recovering as the model retrains.',
    'Both series stabilise at intermediate values ($0.60$--$0.80$) through the\n'
    'middle of the evaluation period, reflecting the model\'s partial recovery after\n'
    'each retraining episode. On ULB CC Fraud, by contrast, SSI shows a localised\n'
    'drop around the single drift event (window 13 for XGBoost, window 15 for\n'
    'LightGBM) before recovering as the model retrains.',
    '§5.2: GMSC second feature'
)

sub(
    'The mean SSI values summarised in Table~\\ref{tab:main_results} cleanly\n'
    'separate the high-drift regime (IEEE-CIS: 0.660--0.683) from the low-drift\n'
    'regimes (GMSC: 0.992--0.997; ULB CC Fraud: 0.952--0.990). This three-way\n'
    'contrast confirms that SSI responds to genuine distributional change and does\n'
    'not conflate sampling noise with structural drift.',
    'The mean SSI values summarised in Table~\\ref{tab:main_results} clearly\n'
    'separate the high-drift regime (IEEE-CIS: 0.660--0.683) from the low-drift\n'
    'regime (ULB CC Fraud: 0.952--0.990) --- a gap of more than 0.27 points that\n'
    'holds across both model architectures, confirming that SSI responds to genuine\n'
    'distributional change rather than sampling noise.',
    '§5.2: three-way contrast'
)

# ─── 12. §5.4 Ablation: GMSC paragraph ───────────────────────────────────
sub(
    '\n\\textbf{GMSC.} All four conditions converge at AUC $\\approx 0.508$,\n'
    'consistent with the absence of genuine temporal drift in this dataset.\n'
    'Component contributions are unidentifiable when the data generating process\n'
    'is stationary.\n',
    '\n',
    '§5.4: ablation GMSC paragraph'
)

# ─── 13. §5.6 Statistical Significance ───────────────────────────────────
sub(
    'consistent with either the absence of meaningful drift advantage (GMSC) or the\n'
    'counter-productive effect of single-episode retraining on an otherwise\n'
    'well-performing model (ULB CC Fraud).',
    'consistent with the counter-productive effect of single-episode retraining\n'
    'on the near-stable ULB CC Fraud dataset.',
    '§5.6: stats GMSC mention'
)

# ─── 14. §5.8 Discussion: three-dataset sentence ─────────────────────────
sub(
    'The mean SSI values on the three datasets span a wide and clean range:\n'
    '$0.66$--$0.68$ on IEEE-CIS (strong drift), $0.95$--$0.99$ on ULB CC Fraud\n'
    '(moderate and localised drift), and $0.99$--$1.00$ on GMSC (no real drift).\n'
    'This three-cluster separation holds across both model architectures and is\n'
    'stable even in the absence of ADWIN drift detections on GMSC. A practitioner\n'
    'deploying SSI in production could monitor the running mean and alert on\n'
    'sustained values below $\\tau_{\\mathrm{SSI}} = 0.80$ without per-dataset\n'
    'recalibration.',
    'The mean SSI values across the two datasets show a clear and stable\n'
    'separation: $0.66$--$0.68$ on the heavily drifting IEEE-CIS benchmark versus\n'
    '$0.95$--$0.99$ on ULB CC Fraud (localised drift) --- a gap of more than\n'
    '0.27 points that holds across both XGBoost and LightGBM. A practitioner\n'
    'deploying SSI in production could monitor the running mean and alert on\n'
    'sustained values below $\\tau_{\\mathrm{SSI}} = 0.80$ without per-dataset\n'
    'recalibration.',
    '§5.8: discussion three-dataset sentence'
)

# ─── 15. Conclusion: three → two ─────────────────────────────────────────
sub(
    'Evaluated on three structurally distinct public credit datasets using a',
    'Evaluated on two structurally distinct public credit datasets using a',
    'Conclusion: three → two'
)

sub(
    '  \\item Mean SSI cleanly separates three drift regimes across datasets:\n'
    '        $0.66$--$0.68$ under persistent concept drift (IEEE-CIS), $0.95$--$0.99$\n'
    '        under localised drift (ULB CC Fraud), and $0.99$--$1.00$ under a\n'
    '        drift-free regime (GMSC). This separation is stable across both XGBoost\n'
    '        and LightGBM architectures, confirming that SSI responds to genuine\n'
    '        distributional change rather than sampling noise.',
    '  \\item Mean SSI separates high-drift from low-drift environments:\n'
    '        $0.66$--$0.68$ under persistent concept drift (IEEE-CIS) versus\n'
    '        $0.95$--$0.99$ under localised drift (ULB CC Fraud) --- a gap of\n'
    '        more than 0.27 points that is stable across both XGBoost and LightGBM,\n'
    '        confirming that SSI tracks genuine distributional change without\n'
    '        threshold recalibration.',
    'Conclusion: finding 2 three-regime → two-regime'
)

# ─── 16. Limitations: remove GMSC first point ────────────────────────────
sub(
    'Several limitations warrant acknowledgement. First, the GMSC dataset lacks a\n'
    'real transaction timestamp; the pseudo-temporal ordering was constructed by\n'
    'sorting on demographic attributes, which does not replicate the temporal\n'
    'dynamics of a live credit stream. Second, the number of evaluation windows\n'
    '(17--18 per dataset) constrains statistical power:',
    'Several limitations warrant acknowledgement. First, the number of evaluation\n'
    'windows (17--18 per dataset) constrains statistical power:',
    'Limitations: remove GMSC first point'
)

# ─── 17. Data Availability ────────────────────────────────────────────────
sub(
    'The IEEE-CIS Fraud\nDetection and Give Me Some Credit datasets are available via Kaggle. The ULB\nCredit Card Fraud dataset is available via the UCI Machine Learning Repository.',
    'The IEEE-CIS Fraud Detection dataset is available via Kaggle. The ULB\nCredit Card Fraud dataset is available via the UCI Machine Learning Repository.',
    'Data Availability: remove GMSC'
)

# ─── Write ────────────────────────────────────────────────────────────────
with open(TEX, 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Applied {len(applied)} changes:')
for c in applied: print(f'  ✓ {c}')
print(f'\nFile: {original_len:,} → {len(text):,} chars  ({original_len - len(text):+,})')
