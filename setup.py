from setuptools import find_packages, setup

# from meinsweeper.version import PYTHON_REQUIRES, REQUIRES

setup(
    name='meinsweeper',
    # packages=find_packages(),
    python_requires = ">= 3.7",
    install_requires = [
        "nvidia-ml-py==11.450.51", 
        "asyncssh", 
        "rich",
        "psutil",
        "duecredit"
    ],
#     extras_require = {
#        'dev': ['pylint'],
#        'build': ['requests']
#    }
)