from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cellula-ai-coding-assistant",
    version="1.0.0",
    author="Cellula Project",
    description="An Intelligent AI Coding Assistant using HuggingFace and ChromaDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/cellula-project",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    scripts=[
        "scripts/cellula_ai_coding_assistant.py"
    ]
)
