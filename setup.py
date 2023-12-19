import setuptools

setuptools.setup(
    name="SiNAPS",
    version="0.3.0",
    author="Claire Guerrier",
    author_email="claire.guerrier@univ-cotedazur.fr",
    description="Neuronal simulation package",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "param",
        "pandas",
        "quantiphy",
        "scipy",
        "hvplot",
        "networkx",
        "datashader",
        "tqdm",
        "numba",
        # "pygraphviz>=1.11"
    ],
)
