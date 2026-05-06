from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements():
    path = Path("requirements.txt")
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="sensor-image-recon",
    version="0.1.0",
    description="Domain-neutral sensor-conditioned image reconstruction",
    packages=find_packages() + find_packages(where="src"),
    package_dir={"sensor_image_recon": "src/sensor_image_recon"},
    install_requires=_read_requirements(),
    python_requires=">=3.10",
)
