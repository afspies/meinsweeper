import os
os.environ['MEINSWEEPER_RETRY_INTERVAL'] = '15'  # Set to 15 seconds for testing
os.environ['MINIMUM_VRAM'] = '30'  # 30 GB minimum free VRAM
os.environ['USAGE_CRITERION'] = '0.1'  # 10% maximum GPU utilization
os.environ['MEINSWEEPER_DEBUG'] = 'True'

# Now import the rest of the modules
import pytest
from meinsweeper.meinsweeper import run_sweep
import asyncio
import tempfile
import subprocess
import time

# Override the retry interval for testing purposes
os.environ['MEINSWEEPER_RETRY_INTERVAL'] = '15'  # 30 seconds instead of 7.5 minutes
os.environ['MINIMUM_VRAM'] = '30'  # 10 GB minimum free VRAM
os.environ['USAGE_CRITERION'] = '0.1'  # 10% maximum GPU utilization

# Set the debug flag for the test
os.environ['MEINSWEEPER_DEBUG'] = 'True'

def create_gpu_occupying_script():
    return """     
import torch
import time

def occupy_gpu():
    device = torch.device('cuda')
    # Create a large tensor to occupy GPU memory
    tensor = torch.rand(10000, 10000, device=device)
    print(f'Occupying GPU with tensor of shape {tensor.shape}')
    
    # Run a simple operation in a loop to maintain GPU utilization
    for _ in range(300):  # Run for about 60 seconds
        result = torch.matmul(tensor, tensor)
        time.sleep(0.1)

if __name__ == '__main__':
    occupy_gpu()
    """

def create_job_script():
    return """
import torch
import time
from meinsweeper.modules.logger import MSLogger

def run_job():
    logger = MSLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running job on device: {device}')
    
    # Log initial step
    logger.log_loss(0.5, mode='train', step=0)
    
    # Try to create a tensor on GPU
    try:
        tensor = torch.rand(5000, 5000, device=device)
        print(f'Created tensor of shape {tensor.shape} on {device}')
        time.sleep(10)  # Simulate some work
    except RuntimeError as e:
        print(f'Failed to allocate tensor on GPU: {e}')
        logger.log_loss(1.0, mode='train', step=1)  # Log failure
        return
    
    # Log final step
    logger.log_loss(0.1, mode='train', step=100)
    print('Job completed successfully')

if __name__ == '__main__':
    run_job()
    """

@pytest.mark.asyncio
async def test_run_sweep_with_gpu_occupation():
    # Write the GPU occupying script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(create_gpu_occupying_script())
        gpu_occupy_script_path = temp_file.name

    # Write the job script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(create_job_script())
        job_script_path = temp_file.name

    # Start processes to occupy GPUs
    gpu_processes = []
    for i in range(2):  # Occupy both GPUs
        process = subprocess.Popen(['python', gpu_occupy_script_path])
        gpu_processes.append(process)

    print("Started GPU occupying processes. Waiting for 10 seconds...")
    time.sleep(10)  # Give some time for the GPUs to be occupied

    # Define targets (using local_async nodes)
    targets = {
        'gpu0': {'type': 'local_async', 'params': {'gpus': ['0']}},
        # 'gpu1': {'type': 'local_async', 'params': {'gpus': ['1']}}
    }

    # Create a list of commands for the sweep
    commands = [(f"python {job_script_path}", f"job{i}") for i in range(8)]  # Increased to 8 jobs

    # Run the sweep
    print("Starting the sweep...")
    await run_sweep(commands, targets, steps=2)

    print("Sweep completed. Cleaning up...")

    # Clean up the temporary files
    os.unlink(gpu_occupy_script_path)
    os.unlink(job_script_path)

    # Terminate the GPU occupying processes
    for process in gpu_processes:
        process.terminate()
        process.wait()

    print("Test completed.")

if __name__ == "__main__":
    asyncio.run(test_run_sweep_with_gpu_occupation())