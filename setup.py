from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphplag",
    version="0.1.0",
    author="ZenleX-Dost",
    description="Semantic Graph-Based Plagiarism Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZenleX-Dost/GraphPlag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "spacy>=3.5.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "networkx>=3.0",
        "torch-geometric>=2.3.0",
        "grakel>=0.1.9",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "pyvis>=0.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "pandas>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "experiments": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
)
