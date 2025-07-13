"""
PhantomHalo Browser Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phantomhalo-browser",
    version="1.0.0",
    author="PhantomHalo Development Team",
    description="Dark Halo OS Internet Browser with Post-Quantum Security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "asyncio-mqtt>=0.13.0",
        "pyyaml>=6.0",
        "aiohttp>=3.8.0",
        "pqcrypto>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "bandit>=1.7.5",
            "pylint>=2.17.0",
            "mypy>=1.5.0",
            "black>=23.7.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phantomhalo=phantomhalo_browser:main",
        ],
    },
)
