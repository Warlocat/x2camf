import os
import re
import sys
import platform
import subprocess
import shutil
import pathlib
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# Based on https://github.com/pybind/cmake_example

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        build_args = ["--config", cfg, "--", "-j"]

        # Ensure build temp dir exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # Locate .so (or .pyd) and move into x2camf/
        shared_libs = list(pathlib.Path(extdir).glob("libx2camf*.so"))  # adjust for Windows if needed
        if not shared_libs:
            raise RuntimeError(f"No libx2camf*.so found in {extdir}")

        target = pathlib.Path("x2camf") / shared_libs[0].name
        shutil.copy2(shared_libs[0], target)
        print(f"Copied {shared_libs[0]} → {target}")
        # Find built .so
        shared_libs = list(pathlib.Path(extdir).glob("libx2camf*.so"))
        if not shared_libs:
            raise RuntimeError(f"No libx2camf*.so found in {extdir}")
        
        # Move it into x2camf/ in the build directory
        target_package_dir = extdir / "x2camf"
        target_package_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(shared_libs[0]), target_package_dir / shared_libs[0].name)
        print(f"Moved {shared_libs[0]} -> {target_package_dir / shared_libs[0].name}")


# ----------------------------
# Package Metadata
# ----------------------------
AUTHOR = ''
PACKAGE_VERSION = '0.1'
DESCRIPTION = 'x2camf'
with open('README.md', 'r') as f:

    LONG_DESCRIPTION = f.read()

# ----------------------------
# Setup
# ----------------------------
setup(
    name='x2camf',
    version=PACKAGE_VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[CMakeExtension("libx2camf")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        "x2camf": ["libx2camf*.so"],  # ensure the built library is included in the wheel
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=["numpy"],
    python_requires=">=3.7",
)
