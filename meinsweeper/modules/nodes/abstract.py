from abc import ABC, abstractmethod
import asyncio

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

