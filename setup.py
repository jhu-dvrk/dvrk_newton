from pathlib import Path

from setuptools import find_packages, setup


package_name = "dvrk_newton"
script_files = ["scripts/simulator.py"]


data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/share", ["share/newton.yaml"]),
    (f"share/{package_name}/share/scenes", [
        str(path) for path in sorted(Path("share/scenes").glob("*.yaml"))
    ]),
    (f"share/{package_name}/share/open-xr", [
        str(path) for path in sorted(Path("share/open-xr").glob("*"))
        if path.is_file()
    ]),
    (f"share/{package_name}/share/schemas", [
        "share/schemas/openxr-video.schema.json",
    ]),
    (f"share/{package_name}/launch", [
        "launch/open_xr.launch.py",
        "launch/simulator.launch.py",
    ]),
    (f"share/{package_name}/scripts", script_files),
]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Anton Deguet",
    maintainer_email="anton.deguet@jhu.edu",
    description="NVIDIA Newton backend for the common dVRK simulator runtime.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "simulator_node = dvrk_newton.node:main",
        ],
    },
)
