from setuptools import setup, find_packages

setup(
    name="FAIRshake",
    version="0.1.1",
    author="Finley Holt",
    author_email="finley0454@gmail.com",
    description="A comprehensive data processing pipeline for FAIRshake.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FinleyHolt/SHAKE",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "Pillow",
        "click",
        "fabio==2024.9.0",
        "h5py",
        "imageio",
        "matplotlib",
        "numpy",
        "psutil",
        "pyFAI",
        "requests",
        "tensorflow",
        "torch",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'fairshake=fairshake.cli:main',
        ],
    },
)