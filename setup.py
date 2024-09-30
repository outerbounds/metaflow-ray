from setuptools import setup, find_namespace_packages

version = "0.1.0"

def get_long_description() -> str:
    with open("README.md") as fh:
        return fh.read()

setup(
    name="metaflow-ray",
    version=version,
    description="An EXPERIMENTAL Ray decorator for Metaflow",
    author="Riley Hun",
    long_description=get_long_description(),
    author_email="riley.hun@autodesk.com",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)
