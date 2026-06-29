#########################################
#  __  __      _ _   _ ___ ___          #
# |  \/  |_  _| | |_(_) _ \ __|_ __     #
# | |\/| | || | |  _| |   / _|\ \ /     #
# |_|  |_|\_,_|_|\__|_|_|_\___/_\_\     #
# Planetary spectra generator           #
#########################################
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    # ######################################################################
    # BASIC DESCRIPTION
    # ######################################################################
    name='multirex',
    author='David Duque-Castaño and Jorge I. Zuluaga',
    author_email='dsantiago.duque@udea.edu.co',
    description='Massive planetary spectra generator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://pypi.org/project/multirex',
    keywords='exoplanets astrobiology astronomy spectroscopy',
    license='MIT',

    # ######################################################################
    # CLASSIFIER
    # ######################################################################
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        ],
    version='0.3.1',

    # ######################################################################
    # FILES
    # ######################################################################
    package_dir={'': '.'},
    packages=setuptools.find_packages(where='.', exclude=["multirex.data", "multirex.tests"]),

    
    # ######################################################################
    # ENTRY POINTS
    # ######################################################################
    entry_points={
        # 'console_scripts': ['install=pryngles.install:main'], # Removed pryngles legacy entry point
    },

    # ######################################################################
    # TESTS
    # ######################################################################
    test_suite='nose.collector',
    tests_require=['nose'],

    # ######################################################################
    # DEPENDENCIES
    # ######################################################################
    install_requires=[
        'numpy',
        'taurex>=3.1.14',
        'matplotlib',
        'tqdm',
        'pandas',
        'gdown',
        'pyarrow',
        'joblib',
        'astropy',
        'requests'
    ],
    
    extras_require={
        'ggchem': ['taurex-ggchem'],
    },

    # ######################################################################
    # OPTIONS
    # ######################################################################
    include_package_data=True,
    package_data={'': ['data/*.*', 'tests/*.*']},
    #scripts=['multirex/scripts/imultirex','multirex/scripts/multirex-test.py'],
)

