from meinsweeper import run_sweep

targets = []  # Machines
USERNAME = 'afs219'
SSH_KEY_PATH = '/media/home/alex/.ssh/doc'

for i in range(1, 20):
    targets.append(
        {
            'type': 'ssh',
            'params': {
                'address': f'ray{i:02}.doc.ic.ac.uk',
                'username': USERNAME,
                'key_path': SSH_KEY_PATH
            }
        }
    )
for i in range(1, 27):
    # if i not in [18]: # Kitty GPUs
    targets.append(
        {
            'type': 'ssh',
            'params': {
                'address': f'gpu{i:02}.doc.ic.ac.uk',
                'username': USERNAME,
                'key_path': SSH_KEY_PATH
            }
        }
    )

run_configurations = []
cmd = 'python -c "import time;[print(f\'[[LOG_ACCURACY TRAIN]] Step: {i+1}; Losses: Train:0.1\') for i in range(10) if time.sleep(3) is None]"'
# print(cmd)
for i in range(200):
    run_configurations.append((cmd, str(i)))

steps = 10
run_sweep(run_configurations, targets, steps=steps)

#   hydra.run.dir=\\"{path}\\" \
#   +run_base_dir=\\"{path}\\" \
#   debug={DEBUG} \
#   dataset_path={dataset_path} \
#   {stringify_stages(stages)} \
#   {' '.join([f'{k}={parse_yaml_value(v)}' for k, v in sweep_params.items()])}\
#   {' '.join([f'{k}={parse_yaml_value(v)}' for k, v in cfg['hydra_overrides'].items()])}
#     """

# def parse_yaml_value(val):
#     if val is None:
#         return 'null'
#     elif isinstance(val, str):
#         if '~' in val: # Tilde in path etc., not used as removal
#             return f"'{val}'"  # Escape the tilde
#         return f"'{val}'"
#     elif isinstance(val, list):
#         return f'[{",".join(map(parse_yaml_value, val))}]'
#     else:
#         return str(val)

# def dict_product(dicts):
#     return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

# def stringify_stages(stages):
#     return " ".join(stringify_recurse(stages, 'stages', []))

# def stringify_recurse(stages, prefix, output):
#     for key, value in stages.items():
#         if not isinstance(value, dict):
#             output.append(f"+{prefix}.{key}={parse_yaml_value(value)}".replace(" ", ''))
#         else:
#             stringify_recurse(value, f"{prefix}.{key}", output)
#     return output
