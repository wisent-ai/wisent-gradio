from setuptools import setup, find_namespace_packages
setup(
    name="wisent-gradio",
    version="0.1.2",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Gradio UI for the wisent package family",
    url="https://github.com/wisent-ai/wisent-gradio",
    packages=find_namespace_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=[
        "wisent>=0.10.0",
        "gradio>=4.0.0",
        # The fleet's one failure envelope, pinned to an exact revision: the
        # vocabulary this console classifies with is not allowed to drift under
        # it between checkouts.
        "wisent-errors @ git+https://github.com/wisent-ai/wisent-errors"
        "@45a424c4f8e93488d2a461a1571df15795f4e612#subdirectory=python",
    ],
    include_package_data=True,
    package_data={"wisent": ["app/*.png", "app/icons/*.svg"]},
)
