from abc import ABC, abstractmethod
import socket
import asyncio
import asyncssh
from pathlib import Path
# from rich.console import Console
# console = Console()
import xml.etree.ElementTree as ET

MINIMUM_VRAM =  8 # in GigaBytes
USAGE_CRITERION = 0.8 # percentage (float) or -1 => no processes other than xorg
MAX_PROCESSES = -1 # -1 => no limit, Otherwise number of processes = min(#nodes, #tbc_runs)
RUN_TIMEOUT = 120
MAX_RETRIES = 3

class Target():
    def __init__(self, target, retries=3) -> None:
        self.details = target
        self.retries = retries


    def get_target(self):
        return self.target 
    
    def retry(self):
        self.retries -= 1

    def not_failed_us(self):
        return self.retries > 0
    
    def __lt__(self, other):
        #To work with PQ => Return True if we have MORE retries left
        if self.retries > other.retries:
            return True


class RunManager:
    """ The RunManager spawns processes on the target nodes from the pool of available ones
    """
    def __init__(self, targets: dict, task_q: asyncio.Queue, log_q: asyncio.Queue) -> None:
        self.targets = targets
        self.max_proc = len(targets) if MAX_PROCESSES == -1 else MAX_PROCESSES
        self.log_q = log_q
        self.task_q = task_q
        self.running_proc = 0
        self.tasks = []

    async def start_run(self):
        # Create queue of available target nodes
        self.target_q = asyncio.PriorityQueue() #maxsize=MAX_PROCESSES) #! This should be num tasks, also check behaviour
        [await self.target_q.put(Target(target, retries=MAX_RETRIES)) for target in self.targets]
        
        # Spawn workers - this needs to be ongoing? 
        [self.tasks.append(asyncio.create_task(self.spawn_worker())) for _ in range(self.max_proc)]
        await asyncio.gather(*self.tasks)

    async def spawn_worker(self) -> None:
        while not self.task_q.empty():
            connected = False
            target=None
            retries=None
            while not connected:
                target = await self.target_q.get()
                # Create target
                if target.details['type'] == 'ssh':
                    node = SSHTarget(log_q=self.log_q, **target.details['params'])
                    connected = await node.open_connection()
                if not connected:
                    if target.not_failed_us():
                        target.retry()
                        await self.target_q.put(target)
                    await asyncio.sleep(5)

            # Run task once a connection was opened
            if self.task_q.empty(): # In case someone else was faster NOTE should instead timeout here
                self.target_q.task_done()
                return

            # Get a task and run it
            # NOTE should add timeout here in case of async funkiness
            cfg, label = await self.task_q.get() # Get next task from queue
            # cfg will be of form {'param1': ..., 'param2': ..., 'param3': ...}
            # cmd = ' '.join(f'{k}={v}' for k, v in cfg.items())
            # label = ','.join(f'{k.split(".")[-1]}={v}' for k, v in cfg.items())
                    #   export XLA_PYTHON_CLIENT_PREALLOCATE=false;
                    # XLA_PYTHON_CLIENT_MEM_FRACTION=.XX
            cmd = f"""source ~/.bashrc;
                      conda activate jax;
                      XLA_PYTHON_CLIENT_MEM_FRACTION=.7;
                      cd ./relational_slots;
                      {cfg}"""
            success = await node.run(cmd, label)
            if success:
                await self.target_q.put(target)
            else:
                if target.not_failed_us():
                    target.retry()
                    await self.target_q.put(target)
                await self.task_q.put((cfg, label)) # Need to do task again
            self.task_q.task_done()
            self.target_q.task_done()


class ComputeNode(ABC):
    def __init__(self, log_q: asyncio.Queue) -> None:
        pass

    @abstractmethod
    def run(self):
        pass
    
    # @abstractmethod
    # def put(self, src, target, recursive=True):
    #     pass

    # @abstractmethod
    # def get(self, src, target, recursive=True):
    #     pass


class SSHTarget(ComputeNode):
    def __init__(self, address, username, log_q, password=None, key_path=None, timeout=5):
        assert password or key_path, "Either password or key_path must be provided."
        self.log_q = log_q
        self.connection_info = {'address': address, 'username': username, 'password': password, 'key_path': key_path, 'timeout':timeout}
        

    async def open_connection(self):
        address, username, key_path, timeout = self.connection_info['address'], self.connection_info['username'], self.connection_info['key_path'], self.connection_info['timeout']
        try:
            self.conn = await asyncssh.connect(address,username=username, known_hosts=None, login_timeout=timeout, encoding='utf-8',
                                               client_keys=[key_path] if key_path else None, term_type='bash')
            #! this makes me sad                
            self.scp_conn = await asyncssh.connect(address,username=username, known_hosts=None, login_timeout=timeout, encoding='utf-8',
                                               client_keys=[key_path] if key_path else None)
        except asyncssh.PermissionDenied  as auth_err:
            print(f"didn't conenct to {address}")
            raise ValueError('Authentication failed. Check user name and SSH key configuration.') from auth_err
        except asyncssh.DisconnectError as ssh_err:
            print(f"disconnected from {address}")
            raise ValueError("No Session") from ssh_err
        except socket.gaierror as target_err:
            print(f"Could not connect to {address}") 
            return False

        # Check if GPU is free
        self.free_gpus = await self.check_gpu_free()
        if self.free_gpus is None:
            return False

        return True

    async def check_gpu_free(self):
        # files = await self.conn.run('ls')
        await asyncssh.scp(Path(__file__).parent/'check_gpu.py', (self.scp_conn, '/tmp/check_gpu.py'))
        gpus = await self.conn.run('python /tmp/check_gpu.py')
        gpus = gpus.stdout
        # self.scp_conn.close()
        if gpus.startswith('[[GPU INFO]]'):
            gpu_info = gpus[14:].split(']')[0]
            if gpu_info:
                return gpu_info
        return None




    async def run(self, command, label):
        """ 
        Should be non-blocking and return whever a signal is recieved
        """
        # try:
        async with self.conn.create_process(command) as proc: # stderr=asyncssh.STDOUT
            await self.log_q.put((({'status':'running'},'running'), self.connection_info['address'], label))

            async for line in timeout(proc.stdout.__aiter__(), RUN_TIMEOUT, 'TIMEOUT'):
                if line == 'TIMEOUT':
                    await self.log_q.put((({'status':'failed'},'failed'), self.connection_info['address'], label))
                    return False

                parsed_line = self.parse_log_line(line)
                await self.log_q.put(((parsed_line, line), self.connection_info['address'], label))

            async for err_line in proc.stderr: 
                if err_line != '':
                    print(f'got error {err_line} for label')
                    await self.log_q.put((({'status':'failed'},'failed'), self.connection_info['address'], label))
                    return False
        return True
               
    @staticmethod
    def parse_log_line(line):
        out = None
        if '[[LOG_ACCURACY TRAIN]]' in line:
            out = {}
            line = line.split('[[LOG_ACCURACY TRAIN]]')[1]
            line = line.split(';')
            for section in line:
                if 'Elapsed' in section:
                    continue
                elif "Losses" in section:
                    loss_terms = section.split('Losses:')[1].split(',')
                    for loss_term in loss_terms:
                        loss_name, loss_value = map(lambda x: x.strip(), loss_term.split(':'))
                        out[f'loss_{loss_name}'] = float(loss_value)
                elif 'Step' in section:
                    out['completed'] = int(section.split('Step:')[1])
        elif '[[LOG_ACCURACY TEST]]' in line:
            line = line.split(':')[1]
            out = {'test_acc': float(line.strip())}
        return out
    
    def scp_put(self, src, dst, recursive=True):
        self.scp.put(src, dst, recursive=recursive)
    
    def scp_get(self, src, dst, recursive=True):
        self.scp.get(src, dst, recursive=recursive)



# Taken from  https://stackoverflow.com/questions/50241696/how-to-iterate-over-an-asynchronous-iterator-with-a-timeout
from typing import *
T = TypeVar('T')
# async generator, needs python 3.6
async def timeout(it: AsyncIterator[T], timeo: float, sentinel: T) -> AsyncGenerator[T, None]:
    try:
        nxt = asyncio.ensure_future(it.__anext__())
        while True:
            try:
                yield await asyncio.wait_for(asyncio.shield(nxt), timeo)
                nxt = asyncio.ensure_future(it.__anext__())
            except asyncio.TimeoutError:
                yield sentinel
    except StopAsyncIteration:
        pass
    finally:
        nxt.cancel()  # in case we're getting cancelled our self
