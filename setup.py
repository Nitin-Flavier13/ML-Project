# this file is used to make the project as a package

from typing import List
from setuptools import find_packages, setup 

HYPHEN_E_DOT = '-e .'

def get_requirements(filePath: int) -> List[str]:
    req = []

    with open(filePath) as file_obj:
        requirements = file_obj.readlines()
        req = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in req:
            req.remove(HYPHEN_E_DOT)
    
    return req


setup(
    name='mlproject',
    version='0.0.1',
    author='Nitin Flavier',
    author_email='nitinflavier13@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    

)