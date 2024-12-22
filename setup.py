from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ccalign",
    version="0.0.1",
    author="Jonas Ewertz",
    author_email="jonas.ewertz@ikf.rub.de",
    description="A sentence-level text-audio alignment tool for conference calls.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/j-ewertz/ccalign",
    packages=find_packages(exclude=["tests*"]),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    license="MIT",
)
