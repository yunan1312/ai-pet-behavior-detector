"""宠物行为智能分析软件 - 安装配置"""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='pet-behavior-analyzer',
    version='2.0.0',
    description='通过计算机视觉和语音交互技术，实时解读宠物的行为意图',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='PetBehaviorAnalyzer Team',
    python_requires='>=3.9',
    packages=find_packages(exclude=['tests*', 'resources*']),
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'PyQt6>=6.6.0',
        'numpy>=1.24.0',
        'Pillow>=10.0.0',
        'pyyaml>=6.0',
        'edge-tts>=6.1.10',
        'ultralytics>=8.0.0',
        'scikit-learn>=1.3.0',
        'onnxruntime>=1.16.0',
        'pygame>=2.5.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'pet-analyzer=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Video :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)