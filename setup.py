import setuptools

setuptools.setup(
    name="SiNAPS",
    version="0.3.0",
    author="Claire Guerrier",
    author_email="claire.guerrier@univ-cotedazur.fr",
    description="Neuronal simulation package",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "param==1.10.0",
        "pandas",
        "quantiphy",
        "scipy>=1.0.0",
        "hvplot",
        "networkx",
        "datashader",
        "tqdm",
        "numba>=0.55",
    ],
)
