# MeinSweeper
<img src="meinsweeper/logo.png" align="right"
     alt="Minesweeper image taken from https://www.pngwing.com/en/free-png-vxhwi" width="80" height="80">

MeinSweeper is a light-weight framework for running experiments on arbitrary compute nodes

```diff
- This is still in alpha, and was written for research
- I.e. expect bugs and smelly code!
```

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MeinSweeper.

```bash
pip install meinsweeper
```

## Usage
### MeinSweeper consists of four components
### Using Provided Node-types
```python
import meinsweeper

cfg = {'target': 'ssh', ...}

meinsweeper.run(cfg)
```


### With Custom Nodes
#### Subclass and existing node
Override the initialization, running or logging behaviour of an existing node type.
If you would like to establish a generic, and robust, communication class which systematically utilizes STDOUT, that would be nice!.

#### Create a node class
```python
class MyNode(Node):


```

Specify new target as pascal_case version of class name 

```python

cfg = {'target':'my_node', ....}

meinsweeper.run(cfg)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
