"""Setup configuration for RSI-Bench."""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
      name="rsi-bench",
      version="0.1.0",
      author="Sung hun kwag",
      description="A Multi-Axis Benchmark for Recursive Self-Improvement in AI Systems",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/sunghunkwag/rsi-bench",
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=[
                "numpy>=1.21.0",
                "scipy>=1.7.0",
                "sortedcontainers>=2.4.0",
      ],
      extras_require={
                "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black", "flake8", "mypy"],
                "viz": ["matplotlib>=3.5.0", "seaborn>=0.12.0"],
      },
      classifiers=[
                "Development Status :: 3 - Alpha",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      entry_points={
                "console_scripts": [
                              "rsi-bench=rsi_bench.core:main",
                ],
      },
)
