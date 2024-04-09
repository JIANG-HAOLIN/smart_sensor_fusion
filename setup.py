from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
long_description = ''.join(lines)

setup(
    name="smartsensorfusion",
    packages=[
        package for package in find_packages() if package.startswith("robomimic")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "h5py",
        "psutil",
        "tqdm",
        "termcolor",
        "tensorboard",
        "tensorboardX",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "egl_probe>=1.0.1",
        "torch",
        "torchvision",
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='==3.9',
    description="multimod smart sensor fusion",
    author="Haolin Jiang",
    url="",
    author_email="",
    version="0.3.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)