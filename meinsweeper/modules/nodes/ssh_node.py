import logging
import asyncssh
from pathlib import Path

from meinsweeper.modules.utils import timeout_iterator
from .abstract import ComputeNode

# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class SSHNode(ComputeNode):
    def __init__(self, address, username, log_q, password=None, key_path=None, timeout=5):
        assert password or key_path, "Either password or key_path must be provided."
        self.log_q = log_q
        self.connection_info = {
            'address': address,
            'username': username,
            'password': password,
            'key_path': key_path,
            'timeout': timeout
        }
        self.RUN_TIMEOUT = timeout

    async def open_connection(self):
        address, username, key_path, timeout = self.connection_info['address'], self.connection_info[
            'username'], self.connection_info['key_path'], self.connection_info['timeout']
        try:
            self.conn = await asyncssh.connect(
                address,
                username=username,
                known_hosts=None,
                login_timeout=timeout,
                encoding='utf-8',
                client_keys=[key_path] if key_path else None,
                term_type='bash'
            )
            #! this makes me sad
            self.scp_conn = await asyncssh.connect(
                address,
                username=username,
                known_hosts=None,
                login_timeout=timeout,
                encoding='utf-8',
                client_keys=[key_path] if key_path else None
            )
        except asyncssh.PermissionDenied as auth_err:
            logging.warning(f"didn't conenct to {address}")
            # raise ValueError('Authentication failed. Check user name and SSH key configuration.') from auth_err
            return False
        except asyncssh.DisconnectError as ssh_err:
            logging.warning(f"disconnected from {address}")
            # raise ValueError("No Session") from ssh_err
            return False
        except OSError as err:
            print('Got ', err, f'when connecting to {address}')
            return False
        except TimeoutError:
            logging.warning(f"timeout from {address}")
            return False
        except socket.gaierror as target_err:
            logging.warning(f"Could not connect to {address}")
            return False

        # Check if GPU is free
        self.free_gpus = await self.check_gpu_free()
        if self.free_gpus is None:
            return False

        return True

    async def check_gpu_free(self):
        # files = await self.conn.run('ls')
        try:
            await asyncssh.scp(Path(__file__).parent.parent / 'check_gpu.py', (self.scp_conn, '/tmp/check_gpu.py'))
        except asyncssh.sftp.SFTPFailure:
            return None
        # logging.debug("CHECKING GPU")
        gpus = await self.conn.run('python /tmp/check_gpu.py')
        gpus = gpus.stdout
        # logging.debug(gpus)
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
        # logging.debug("RUNNING COMMAND")
        # logging.debug(command)
        # try:
        async with self.conn.create_process(command) as proc:  # stderr=asyncssh.STDOUT
            await self.log_q.put((({'status': 'running'}, 'running'), self.connection_info['address'], label))

            async for line in timeout_iterator(proc.stdout.__aiter__(), self.RUN_TIMEOUT, 'TIMEOUT'):
                if line == 'TIMEOUT':
                    await self.log_q.put((({'status': 'failed'}, 'failed'), self.connection_info['address'], label))
                    return False

                parsed_line = self.parse_log_line(line)
                if parsed_line == 'FAILED':  #! Hack
                    await self.log_q.put((({'status': 'failed'}, 'failed'), self.connection_info['address'], label))
                    return False
                await self.log_q.put(((parsed_line, line), self.connection_info['address'], label))

            async for err_line in proc.stderr:
                if err_line != '':
                    logging.warning(f'got error {err_line} for label')
                    await self.log_q.put((({'status': 'failed'}, 'failed'), self.connection_info['address'], label))
                    return False
        return True

    """
    LOG Format Description
    Log lines should contain [[LOG_ACCURACY TRAIN]] or [[LOG_ACCURACY TEST]]
    Followed by 'sections' divided with ';' and key value pairs divided with ':' (A list of K:V pairs for losses)
    For example:
        [[LOG_ACCURACY TRAIN]] Step: 10; Losses: Train: 0.25, Val: 0.65
    """
    
    @staticmethod
    def parse_log_line(line):
        # print line to stdout in spite of rich table
        # logging.debug(line)
        # logging.debug("PARSING LINE")

        out = None
        #! FIX this can fail if the log line is part of an error message - we will try to parse
        #! but the f-strings won't have been evaluated (meaning we can't convert {step} or {loss} to float)
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
        elif '[[LOG_ACCURACY TEST]]' in line:  #TODO something is broken here
            line = line.split(':')[1]
            out = {'test_acc': float(line.strip())}
        elif 'error' in line or 'RuntimeError' in line or 'failed' in line or 'Killed' in line:
            out = 'FAILED'
        return out

    def scp_put(self, src, dst, recursive=True):
        self.scp.put(src, dst, recursive=recursive)

    def scp_get(self, src, dst, recursive=True):
        self.scp.get(src, dst, recursive=recursive)
