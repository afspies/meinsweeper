# ToDos

## High Priority
1. Clear documentation on how to create a node class
   - Step-by-step guide with examples
   - Best practices and common pitfalls
2. Clear documentation on how to run sweeps
   - Basic usage examples
   - Advanced configuration options
3. Better handling of total-step counts
   - Allow each run in a sweep to have its own total number of steps
   - Implement dynamic progress tracking
4. Better crash logging
   - Improve error handling and reporting
   - Implement crash recovery mechanisms
5. A local async node
   - Implement a node type for using local GPUs with multiple jobs simultaneously
   - Optimize resource allocation for local runs

## Additional Suggestions
6. Implement a plugin system for easy extension of node types and functionalities
7. Add support for distributed logging and monitoring (e.g., integration with Prometheus/Grafana)
8. Improve configuration management (consider using a library like Hydra or OmegaConf)
9. Implement a web-based dashboard for monitoring and managing sweeps
10. Add support for automatic checkpointing and resuming of interrupted sweeps

## Existing Items
* Add logging option for transferring of files - i.e. use -p mode from rsync and add this as a separate process during initialization (and finalization) stage of a producer
* Restart partial rsync in the event of a failure during copying
* Add a target blacklist (rather than just recycling functioning targets)
    * Periodically re-check blacklisted targets
* Set up configuration scheme through a configuration class, assuming hydra or equivalent command line args 
* Add a slurm endpoint via https://github.com/facebookincubator/submitit