from setuptools import setup, find_packages

setup(
    name='YWVecUtil',
    version="1.0.3",
    description=('YunWen vector encoder and utils'),
    long_description=open('README.rst', encoding="utf8").read(),
    author='ZhangDun',
    author_email='dzhang@iyunwen.com',
    maintainer='ZhangDun',
    maintainer_email='dzhang@iyunwen.com',
    license='MIT',
    packages=find_packages(),
    url='',
    install_requires=['torch', "scipy", "scikit-learn", "numpy", "gensim", "transformers", ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords="Transformer Networks BERT XLNet PyTorch NLP deep learning"
)
