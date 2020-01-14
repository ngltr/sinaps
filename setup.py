import setuptools

setuptools.setup(
    name="SiNAPS",
    version="0.0.3",
    author="Claire Guerrier",
    author_email="claire.guerrier@univ-cotedazur.fr",
    description="Neuronal simulation package",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "quantiphy",
        "scipy>=1.0.0",
        "matplotlib",
        "hvplot",
        "tqdm",
        "numba"
    ],
)
