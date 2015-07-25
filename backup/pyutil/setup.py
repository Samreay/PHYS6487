from distutils.core import setup

setup(name='sam', 
      version='0.1',
	  package_dir={'sam': ''},
      packages=['sam'],
	  package_data = {'sam': ['wigglezold/*','wizcola/*/*', 'wizcola/*/*/*', 'wizcola/*/*/*/*']},
      )