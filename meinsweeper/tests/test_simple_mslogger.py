import os
import pytest
import asyncio
from meinsweeper.modules.nodes.local_async_node import LocalAsyncNode
from meinsweeper.modules.logger import MSLogger

# Set the debug flag for the test
os.environ['MEINSWEEPER_DEBUG'] = 'True'

@pytest.mark.asyncio
async def test_simple_mslogger():
    # Create a mock log queue
    log_q = asyncio.Queue()

    # Create a LocalAsyncNode with GPU 0 available
    node = LocalAsyncNode(log_q=log_q, available_gpus=['0'])

    # Open the connection
    connected = await node.open_connection()
    assert connected, "Failed to open connection to LocalAsyncNode"

    # Create a simple job command that uses MSLogger
    simple_job_cmd = """
from meinsweeper.modules.logger import MSLogger
import time

logger = MSLogger()

print("Starting simple job")
logger.log_loss(0.5, mode='train', step=0)
time.sleep(2)  # Short wait to ensure we can capture both logs
logger.log_loss(0.3, mode='train', step=1)
print("Simple job completed")
    """

    # Run the job
    success = await node.run(simple_job_cmd, "simple_job")
    assert success, "Job failed to complete successfully"

    # Check the log queue for expected messages
    expected_messages = [
        "Starting simple job",
        "[[LOG_ACCURACY TRAIN]] Step: 0; Losses: Train: 0.5",
        "[[LOG_ACCURACY TRAIN]] Step: 1; Losses: Train: 0.3",
        "Simple job completed"
    ]

    messages_found = {msg: False for msg in expected_messages}

    print("\nReceived log messages:")
    while not log_q.empty():
        msg = await log_q.get()
        print(f"Received: {msg}")
        if isinstance(msg, tuple) and len(msg) == 3:
            content = msg[0]
            if isinstance(content, str):
                print(f"Content: {content}")
                for expected in expected_messages:
                    if expected in content:
                        messages_found[expected] = True

    # Check if we found all expected messages
    for msg, found in messages_found.items():
        assert found, f"Expected message not found: {msg}"

    print("\nAll expected messages were found.")

if __name__ == "__main__":
    asyncio.run(test_simple_mslogger())