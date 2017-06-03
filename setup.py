
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
    name='reasoning-about-entailment',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    package_data={'reasoning-about-entailment': ['*.txt', 'data/*.txt']},
    include_package_date=True,
    description='Running through Deepmind\'s Reasoning About Entailment With Neural Attention.'
)
