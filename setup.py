import codecs
from setuptools import setup, find_packages


def read_file(filename, cb):
    with codecs.open(filename, 'r', 'utf8') as f:
        return cb(f)


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kochat',
    version='0.6',
    description='Korean opensource chatbot framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hyunwoong Ko',
    author_email='gusdnd852@naver.com',
    url='https://github.com/Kochat-framework/kochat',
    install_requires=read_file('requirements.txt', lambda f: list(
        filter(bool, map(str.strip, f)))),
    packages=find_packages(exclude=[
        'kochat_config.py'
    ]),

    keywords=['chatbot', 'korean chatbot', 'kochat'],
    python_requires='>=3',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
)
