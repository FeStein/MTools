# MTools
This library is a collection of modules I use to work with simulation software
(FEniCS, FEAP, etc.). The moudles are tailored to my own needs, so don't expect
a generality as well as a proper documentation.

## Modules

### inp
Library to parse abaqus mesh (input *.inp*) files. General use:
```python
Parser = inp.InpFileParser('test.inp',4)
abq_mesh = Parser.parse()
```
Note that 4 is specifies the number of nodes per element.

### VTKSampler
Module to create a dolfin mesh consistent of hexagonal elements given a *.vtu*
file.

