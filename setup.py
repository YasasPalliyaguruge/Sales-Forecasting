from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="sales-forecasting-baseline",
    version="0.1.0",
    author="YasasPalliyaguruge",
    description="A reproducible mixed-type regression baseline for sales datasets",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/YasasPalliyaguruge/Sales-Forecasting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "joblib>=1.3",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    ],
)
