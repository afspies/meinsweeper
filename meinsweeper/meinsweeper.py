from pathlib import Path
import os
import numpy as np
import random
import pickle

import jax.numpy as jnp
import haiku as hk
import tensorflow as tf

import subprocess
import os

from .due import *
# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='slopfield')

# Standard haiku forward function
def forward_fn(x, training, net, cfg, debug=False):
    # Debug if a flag always set to true or false
    return net(cfg)(x, training, debug)

# -- Parameter Combination Functions -- #
def rename_treemap_branches(params, rename_tuples):
    """ Takes a haiku treemap datastructured (e.g. model params or state) and a list
    of the form [('old branch subname','new branch subname'), ...]
    
    Returns tree with renamed branches
    """

    if params is not None and len(rename_tuples) > 0: # Loaded model may have no associated state
        params = hk.data_structures.to_mutable_dict(params)
        initial_names = list(params.keys())
        for layer_name in initial_names:
            mapped_name = layer_name
            for (old_name, new_name) in rename_tuples:
                mapped_name = mapped_name.replace(old_name, new_name)
            params[mapped_name] = params[layer_name]
            if mapped_name != layer_name:
                params.pop(layer_name)
    return params

def _split_treemap(trainable_params, trainable_state, loaded_model, partition_string=None):
    loaded_params, loaded_state = loaded_model
    if loaded_params is not None:
        if partition_string is not None:
            # NOTE doesn't support fine-tuning - i.e. loaded_params = Frozen if we are partitioning
            trainable_params, _ = hk.data_structures.partition(
                                lambda m, n, p: partition_string in m, trainable_params)
            trainable_state, _ = hk.data_structures.partition(
                                lambda m, n, p: partition_string in m, trainable_state)
        else: # NOTE This assumes resuming from a checkpoint, but no option for pure testing  
            trainable_params = loaded_params
            tr9ainable_state = loaded_state
            loaded_params = None
            loaded_state = None
    return trainable_params, trainable_state, loaded_params, loaded_state

# -- Loading Functions -- #
# def load_params(path, load_from_step=None):
#     if load_from_step is None: # we'll take the one trained for the longest    
#         load_from_step = sorted(map(lambda x: int(x.split('_')[1][:-4]), os.listdir(path/'models')))[-1]
    
#     # Load parameters
#     with open(path/'models'/f'params_{load_from_step}.pkl', 'rb') as f:
#         params = pickle.load(f)
#     return params

# def load_config(path):
#     # Manually load config
#     # Initialize - must use relative path to cwd because hydra says so
#     initialize(config_path=os.path.relpath((path/'.hydra'), start=Path(os.path.realpath(__file__)).parent))
#     return compose(config_name="config")

# def load_saved_params(path, load_from_step=None):
#     path = Path(path)
#     cfg = load_config(path)
#     params = load_params(path, load_from_step=load_from_step)
#     return cfg, params

# def load_saved_model(path):
#     path = Path(path)
#     cfg, (params, state) = load_saved_params(path)
#     model_class = ModelClasses[cfg.model.name]
#     return cfg, model_class, params, state


# -- Misc. -- # 
def set_rng_seeds(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'


# This function should be called after all imports,
# in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2):
    """Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
                                              
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
    """
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    gpu_info = list(filter(lambda info: 'Used' in info, gpu_info))
    gpu_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_info] # Remove garbage
    gpu_info = gpu_info[:min(max_gpus, len(gpu_info))] # Limit to max_gpus
    # Assign free gpus to the current process
    gpus_to_use = ','.join([str(i) for i, x in enumerate(gpu_info) if x < threshold_vram_usage])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    print(f'Using GPUs {gpus_to_use}' if gpus_to_use else 'No free GPUs found')


def transform_to_jax(*args):
    out = []
    for arg in args:
        if isinstance(arg, dict): 
            out.append({k: jnp.array(v, dtype=jnp.float32) for k, v in arg.items()})
        else:
            out.append(jnp.array(arg, dtype=jnp.float32))
    return out