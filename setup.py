import setuptools

with open("README.md", "r",encoding='utf-8') as f:
    long_description = f.read()




version= "0.0.0"

REPO_NAME = "Kidney_Disease_Classification"
AUTHOR_USER_NAME = "aayushkataria123"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "aayushkataria1910@gmail.com"  




setuptools.setup(
    name=REPO_NAME,
    version=version,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"))