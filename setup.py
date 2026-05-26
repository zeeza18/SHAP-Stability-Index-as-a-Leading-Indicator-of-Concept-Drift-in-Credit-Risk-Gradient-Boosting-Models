"""Setup configuration for the credit-drift research package."""

from setuptools import setup, find_packages

setup(
    name="credit-drift-research",
    version="0.1.0",
    description=(
        "Drift-Aware Gradient Boosting with Temporal Explainability "
        "for Credit Risk Scoring Under Distribution Shift"
    ),
    author="zeeza18",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "xgboost==2.0.3",
        "lightgbm==4.3.0",
        "scikit-learn==1.4.2",
        "imbalanced-learn==0.12.3",
        "river==0.21.0",
        "shap==0.45.0",
        "optuna==3.6.1",
        "pandas==2.2.1",
        "numpy==1.26.4",
        "scipy==1.13.0",
        "pyarrow==15.0.2",
        "pyyaml==6.0.1",
        "python-dotenv==1.0.1",
        "tqdm==4.66.2",
        "joblib==1.3.2",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
    ],
)
