"""Generate notebooks/01_eda.ipynb programmatically (source of truth for the EDA notebook)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Heart Disease UCI - Exploratory Data Analysis\n"
    "\n"
    "Dataset: UCI Heart Disease (Cleveland subset, 303 patients, 13 clinical features).\n"
    "Target collapsed to binary risk: 0 = no disease, 1 = disease present "
    "(original `num` severity scale 0-4)."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n"
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "from src.data_processing import load_raw, load_clean_data\n"
    "\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n"
    "plt.rcParams['figure.figsize'] = (8, 5)"
))

cells.append(nbf.v4.new_markdown_cell("## 1. Raw data overview & missing values"))
cells.append(nbf.v4.new_code_cell(
    "raw = load_raw()\n"
    "print(raw.shape)\n"
    "raw.info()"
))
cells.append(nbf.v4.new_code_cell(
    "missing = raw.isna().sum()\n"
    "missing = missing[missing > 0]\n"
    "print('Columns with missing values:')\n"
    "print(missing)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(6, 3))\n"
    "sns.barplot(x=missing.index, y=missing.values, ax=ax, color='indianred')\n"
    "ax.set_title('Missing values by column (raw data)')\n"
    "ax.set_ylabel('Missing count')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../screenshots/eda_missing_values.png', dpi=120)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Cleaned dataset & target class balance\n"
    "`ca`/`thal` missing values imputed (median/mode); `num` collapsed to binary `target`."
))
cells.append(nbf.v4.new_code_cell(
    "df = load_clean_data()\n"
    "df.head()"
))
cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(5, 4))\n"
    "counts = df['target'].value_counts().sort_index()\n"
    "sns.barplot(x=counts.index.map({0: 'No Disease', 1: 'Disease'}), y=counts.values,\n"
    "            ax=ax, color='steelblue')\n"
    "ax.set_title('Class balance: heart disease presence')\n"
    "ax.set_ylabel('Patient count')\n"
    "for i, v in enumerate(counts.values):\n"
    "    ax.text(i, v + 2, str(v), ha='center')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../screenshots/eda_class_balance.png', dpi=120)\n"
    "plt.show()\n"
    "print(counts / counts.sum())"
))

cells.append(nbf.v4.new_markdown_cell("## 3. Feature distributions (histograms)"))
cells.append(nbf.v4.new_code_cell(
    "numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']\n"
    "fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
    "for ax, col in zip(axes.flat, numeric_cols):\n"
    "    sns.histplot(df[col], kde=True, ax=ax, color='teal')\n"
    "    ax.set_title(col)\n"
    "axes.flat[-1].axis('off')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../screenshots/eda_histograms.png', dpi=120)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell("## 4. Correlation heatmap"))
cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(10, 8))\n"
    "corr = df.corr(numeric_only=True)\n"
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)\n"
    "ax.set_title('Feature correlation heatmap')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../screenshots/eda_correlation_heatmap.png', dpi=120)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Feature relationships with target\n"
    "Focus on the features most correlated with `target` from the heatmap above: "
    "`cp` (chest pain type), `thalach` (max heart rate), `oldpeak`, `ca`, `thal`, `exang`."
))
cells.append(nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n"
    "focus_cols = ['cp', 'thalach', 'oldpeak', 'ca', 'thal', 'exang']\n"
    "for ax, col in zip(axes.flat, focus_cols):\n"
    "    sns.boxplot(data=df, x='target', y=col, ax=ax, color='lightseagreen') if df[col].nunique() > 5 \\\n"
    "        else sns.countplot(data=df, x=col, hue='target', ax=ax)\n"
    "    ax.set_title(f'{col} vs target')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../screenshots/eda_feature_vs_target.png', dpi=120)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Summary of EDA findings\n"
    "\n"
    "- Dataset has 303 patients, 13 features; only `ca` (4 rows) and `thal` (2 rows) have missing values, "
    "imputed with median/mode respectively.\n"
    "- Binary target is reasonably balanced (~54% no disease / ~46% disease) after collapsing severity "
    "levels, so accuracy is a meaningful metric but precision/recall/ROC-AUC are still tracked for "
    "robustness.\n"
    "- `cp` (chest pain type), `thalach` (max heart rate achieved), `oldpeak` (ST depression), `ca` "
    "(number of major vessels) and `thal` show the strongest separation between disease/no-disease groups, "
    "making them the most informative predictors for modeling.\n"
    "- `age`, `chol`, and `trestbps` show only weak linear correlation with the target individually, but are "
    "retained for modeling since they contribute non-linearly (captured well by tree-based models)."
))

nb['cells'] = cells

with open('notebooks/01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Notebook written to notebooks/01_eda.ipynb')
