from setuptools import setup, find_packages

setup(
    name="tinyoct",
    version="1.0.0",
    description="TinyOCT: Anatomy-Guided Structured Projection Attention for Retinal OCT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.0",
        "omegaconf>=2.3.0",
        "medmnist>=2.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
        "Pillow>=10.0.0",
    ],
)
