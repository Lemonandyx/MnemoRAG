from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
VERSION = {}
exec((ROOT / "mnemorag" / "version.py").read_text(encoding="utf-8"), VERSION)

requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="MnemoRAG",
    version=VERSION["__version__"],
    packages=find_packages(),
    url="https://github.com/Lemonandyx/MnemoRAG",
    license="MIT",
    description="Inter-request-aware retrieval-augmented generation serving",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
