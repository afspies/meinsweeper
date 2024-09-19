# MeinSweeper
<img src="meinsweeper/logo.png" align="right"
     alt="Minesweeper image taken from https://www.pngwing.com/en/free-png-vxhwi" width="80" height="80">

MeinSweeper is a lightweight framework for running experiments on arbitrary compute nodes, with built-in support for GPU management and job distribution.

```diff
- This is still in alpha, and was written for research
- I.e. expect bugs and smelly code!
```

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MeinSweeper:

  ```bash
  pip install meinsweeper
  ```

## Features
- Asynchronous job execution
- Support for multiple node types (SSH and Local)
- Automatic GPU management and allocation
- Retry mechanism for failed jobs and unavailable nodes
- Configurable via environment variables

## Usage
### Basic Usage
  ```python
  import meinsweeper

  targets = {
      'local_gpu': {'type': 'local_async', 'params': {'gpus': ['0', '1']}},
      'remote_server': {'type': 'ssh', 'params': {'address': 'example.com', 'username': 'user', 'key_path': '/path/to/key'}}
  }

  commands = [
      ("python script1.py", "job1"), 
      ("python script2.py", "job2"),
      # ... more commands
  ]

  meinsweeper.run_sweep(commands, targets)
  ```

### Node Types
1. **Local Async Node**: Executes jobs on the local machine, managing GPU allocation.
2. **SSH Node**: Connects to remote machines via SSH, manages GPU allocation, and executes jobs.

Both node types handle GPU checking, allocation, and release automatically.

### Configuration
MeinSweeper can be configured using environment variables:

- `MINIMUM_VRAM`: Minimum free VRAM required for a GPU to be considered available (in GB, default: 8)
- `USAGE_CRITERION`: Maximum GPU utilization for a GPU to be considered available (0-1, default: 0.8)
- `MAX_PROCESSES`: Maximum number of concurrent processes (-1 for no limit, default: -1)
- `RUN_TIMEOUT`: Timeout for each job execution (in seconds, default: 1200)
- `MAX_RETRIES`: Maximum number of retries for failed jobs (default: 3)
- `MEINSWEEPER_RETRY_INTERVAL`: Interval between retrying unavailable nodes (in seconds, default: 450)
- `MEINSWEEPER_DEBUG`: Enable debug logging (set to 'True' for verbose output)

Example:
  ```bash
  export MINIMUM_VRAM=10
  export USAGE_CRITERION=0.5
  export MEINSWEEPER_RETRY_INTERVAL=300
  python your_script.py
  ```

## Advanced Usage
### Custom Node Types
You can create custom node types by subclassing the `ComputeNode` abstract base class:

  ```python
  from meinsweeper.modules.nodes.abstract import ComputeNode

  class MyCustomNode(ComputeNode):
      async def open_connection(self):
          # Implementation
      
      async def run(self, command, label):
          # Implementation

  # Usage
  targets = {
      'custom_node': {'type': 'my_custom_node', 'params': {...}}
  }
  ```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT](https://choosealicense.com/licenses/mit/)
