from abc import ABC, abstractmethod
import socket
import asyncio
import asyncssh

import xml.etree.ElementTree as ET

MINIMUM_VRAM =  8 # in GigaBytes
USAGE_CRITERION = 0.8 # percentage (float) or -1 => no processes other than xorg
MAX_PROCESSES = -1 # -1 => no limit, Otherwise number of processes = min(#nodes, #tbc_runs)

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
        self.target_q = asyncio.Queue(maxsize=MAX_PROCESSES)
        [await self.target_q.put(target) for target in self.targets]
        
        # Spawn workers - this needs to be ongoing? 
        [self.tasks.append(asyncio.create_task(self.spawn_worker())) for _ in range(self.max_proc)]
        await asyncio.gather(*self.tasks)

    async def spawn_worker(self) -> None:
        while not self.task_q.empty():
            connected = False
            target=None
            while not connected:
                target = await self.target_q.get()
                # Create target
                if target['type'] == 'ssh':
                    node = SSHTarget(log_q=self.log_q, **target['params'])
                    connected = await node.open_connection()

            # Run task once a connection was opened
            if self.task_q.empty(): # In case someone else was faster NOTE should instead timeout here
                self.target_q.task_done()
                return

            # Get a task and run it
            # NOTE should add timeout here in case of async funkiness
            cfg = await self.task_q.get() # Get next task from queue
            success = await node.run('cd ./run_training;python -u run.py',cfg)
            if success:
                await self.target_q.put(target)
            else:
                await self.task_q.put(cfg) # Need to do task again
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
            self.conn = await asyncssh.connect(address,username=username, known_hosts=None, login_timeout=timeout,
                                               client_keys=[key_path] if key_path else None)
            # self.scp = SCPClient(self.client.get_transport())
        except asyncssh.AuthenticationException as auth_err:
            print(f"didn't conenct to {address}")
            raise ValueError('Authentication failed. Check user name and SSH key configuration.') from auth_err
        except asyncssh.SSHException as ssh_err:
            print(f"didn't conenct to {address}")
            raise ValueError("No Session") from ssh_err
        except socket.gaierror as target_err:
            print(f"Could not connect to {address}") 
            return False
        return True

    async def run(self, command, cfg):
        """ 
        Should be non-blocking and return whever a signal is recieved
        """
        async with self.conn.create_process(command) as proc:
            async for line in proc.stdout:
                await self.log_q.put((line, self.connection_info['address'], cfg))
        
        return True
                # proc.stdin.write(op + '\n')
                # result = await proc.stdout.readline()
                # print(result, end='\t')
        # stdin, stdout, stderr = self.client.exec_command(command, get_pty=True)
        # while not stdout.channel.exit_status_ready():
            # OUT = stdout.channel.recv(1024)
            # OUT = OUT.decode('utf-8').strip().rstrip().replace('\t','').replace('\n','')
            # if len(OUT) > 0:
                # print(OUT)

    @staticmethod
    def parse_out(ssh_stdout):
        # res = ''.join(ssh_stdout.readlines())
        res = ssh_stdout.decode('utf-8')
        try:
            return ET.fromstring(res)
        except ET.ParseError: # Unable to parse result, return error.
            return res
    
    def scp_put(self, src, dst, recursive=True):
        self.scp.put(src, dst, recursive=recursive)
    
    def scp_get(self, src, dst, recursive=True):
        self.scp.get(src, dst, recursive=recursive)
