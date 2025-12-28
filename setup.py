from setuptools import setup, find_packages

setup(
    name="yolo_world",
    version="0.1.0",
    description="YOLO-World: Real-time Open Vocabulary Object Detection",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=["object detection"],
    author="Tencent AILab",
    author_email="ronnysong@tencent.com",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    packages=find_packages(
        include=["yolo_world*"],
        exclude=["docs*", "tests*", "third_party*", "assets*"]
    ),
    package_dir={"yolo_world": "yolo_world"},
    include_package_data=False,
    zip_safe=True,
    install_requires=[
        # Add your dependencies here
    ],
)