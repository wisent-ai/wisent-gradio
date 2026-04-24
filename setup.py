from setuptools import setup, find_packages
setup(
    name="wisent-app",
    version="0.1.0",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Gradio UI for the wisent package family",
    url="https://github.com/wisent-ai/wisent-app",
    packages=find_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=["wisent>=0.10.0", "gradio>=4.0.0"],
    include_package_data=True,
    package_data={"wisent": ["app/*.png", "app/icons/*.svg"]},
)
