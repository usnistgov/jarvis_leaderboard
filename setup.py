import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jarvis_leaderboard",  # Replace with your own username
    version="2023.01.17",
    author="Kamal Choudhary",
    author_email="kamal.choudhary@nist.gov",
    description="jarvis_leaderboard",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
        "scikit-learn>=0.24.1",
        "pandas==1.2.4",
        # "alignn>=2022.10.23",
        "mkdocs-material>=9.0.5",
        "pydantic>=1.8.1",
        "markdown==3.4.1",
        "flake8>=3.9.1",
        "pycodestyle>=2.7.0",
        "pydocstyle>=6.0.0",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knc6/jarvis_leaderboard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
