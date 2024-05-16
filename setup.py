import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jarvis_leaderboard",  # Replace with your own username
    version="2024.4.26",
    author="Kamal Choudhary",
    author_email="kamal.choudhary@nist.gov",
    description="jarvis_leaderboard",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
        "scikit-learn>=0.24.1",
        "pandas>=1.2.4",
        "rouge>=1.0.1",
        "mkdocs>=1.5.2",
        "mkdocs-material>=9.0.5",
        "pydantic>=2.3.0",
        "markdown>=3.2.1",
        "plotly",
        "absl-py==1.4.0",
        "nltk==3.8.1",
        # "evaluate==0.4.0",
        # "rouge-score==0.1.2",
        # "fsspec==2023.9.0",
        # "aiohttp==3.8.5",
        # "datasets==2.14.5",
        # "alignn>=2022.10.23",
        # "flake8>=3.9.1",
        # "pycodestyle>=2.7.0",
        # "pydocstyle>=6.0.0",
    ],
    scripts=[
        "jarvis_leaderboard/jarvis_populate_data.py",
        "jarvis_leaderboard/jarvis_upload.py",
        "jarvis_leaderboard/jarvis_serve.py",
        "jarvis_leaderboard/rebuild.py",
    ],
    package_data={
        "jarvis_leaderboard": [
            "benchmarks/benchmark_dois.json",
            "benchmarks/descriptions.csv",
            "benchmarks/ES/*/*.json.zip",
            "benchmarks/EXP/*/*.json.zip",
            "benchmarks/FF/*/*.json.zip",
            "benchmarks/QC/*/*.json.zip",
            "benchmarks/AI/SinglePropertyPrediction/*.json.zip",
            "benchmarks/AI/SinglePropertyClass/*.json.zip",
            "benchmarks/AI/ImageClass/*.json.zip",
            "benchmarks/AI/Spectra/*.json.zip",
            "benchmarks/AI/TextGen/*.json.zip",
            "benchmarks/AI/TextClass/*.json.zip",
            "benchmarks/AI/TextSummary/*.json.zip",
            "benchmarks/AI/TokenClass/*.json.zip",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knc6/jarvis_leaderboard",
    packages=setuptools.find_packages(),  # exclude=["*forces.json.zip","*csv.zip","*.md","*ipynb"]),
    # packages=setuptools.find_packages(exclude=["*/AI/MLFF/*","jarvis_leaderboard/contributions/*/*csv.zip","*.md"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
