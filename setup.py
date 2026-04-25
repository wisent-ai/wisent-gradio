from setuptools import setup, find_namespace_packages
setup(
    name="wisent-gradio",
    version="0.1.1",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Gradio UI for the wisent package family",
    url="https://github.com/wisent-ai/wisent-gradio",
    packages=find_namespace_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=["wisent>=0.10.0", "gradio>=4.0.0"],
    include_package_data=True,
    package_data={"wisent": ["app/*.png", "app/icons/*.svg"]},
)
