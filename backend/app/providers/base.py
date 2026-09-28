from abc import ABC, abstractmethod

class AgentProvider(ABC):
    @abstractmethod
    async def run(self, task: dict, configuration: str, policy: str) -> dict:
        raise NotImplementedError
