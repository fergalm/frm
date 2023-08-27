
## The Frm Package

Modules with code I share across
many projects


## Sub projects
* base (My common tools)
* plots (anything that depends on matplotlib)
* gis (depends on gdal, geopandas etc)
* politics


## Installation
### Poetry

Create a new package with

```
poetry new $name
```

Edit the pyproject.toml with your 
dependencies. See https://python-poetry.org/docs/dependency-specification/

Build a distribution with

```
cd $name
poetry build
```

The tarball should show up in dist/


## To install to PyPi
poetry publish -u fergalm -p PASSWORD
I keep my password in my password manager
