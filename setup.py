"""
Setup script for URHGN package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="urhgn",
    version="1.0.0",
    author="Yan Xiaoqin",
    author_email="yanxiaoqin@stu.pku.edu.com",
    description="Urban Renewal Hierarchical Graph Network for Predictive Urban Planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yanxiaoqin/URHGN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "urhgn-train=scripts.train:main",
            "urhgn-predict=scripts.predict:main",
            "urhgn-explain=scripts.explain:main",
        ],
    },
    include_package_data=True,
    package_data={
        "urhgn": ["configs/*.yaml"],
    },
)