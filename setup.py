from setuptools import setup, find_namespace_packages

version = "0.0.5"

setup(
    name="metaflow-ray",
    version=version,
    description="An EXPERIMENTAL Ray decorator for Metaflow",
    author="Riley Hun",
    author_email="riley.hun@autodesk.com",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)
