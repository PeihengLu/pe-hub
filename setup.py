from setuptools import setup, find_packages

setup(
    name="pe-db-project",
    version="0.1.0",
    description="Prime Editing Database and Ensemble Project",
    author="Peiheng",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    python_requires=">=3.8",
)