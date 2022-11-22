from setuptools import find_packages, setup

setup(
    name='aesindy',
    packages=find_packages(include=['aesindy', 'aesindy.*']),
    version='0.1',
    description='description',
    author='Joseph Bakarji',
    license='MIT',
    install_requires=['mat73',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'pickle5',
                      'pysindy',
                      'scikit_learn',
                      'scipy',
                      'tensorflow-macos',
                      'tqdm'],
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)
