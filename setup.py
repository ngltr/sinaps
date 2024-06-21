import setuptools

setuptools.setup(
    name="SiNAPS",
    version="0.3.0",
    author="Claire Guerrier",
    author_email="claire.guerrier@univ-cotedazur.fr",
    description="Neuronal simulation package",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.26.4",
        "param==1.10.0",
        "pandas==2.2.1",
        "quantiphy==2.19",
        "scipy=1.11.4",
        "hvplot=0.9.0",
        "networkx==3.2.1",
        "datashader==0.16.0",
        "tqdm==4.66.1",
        "numba=0.58.1",
        "pygraphviz==1.12" # for macOS use brew install graphviz and build package using proper headers
    ],
)
