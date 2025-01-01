from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from typing import List, Optional

import redis
from loguru import logger

from .redis_keys import RedisKeys
from .streams import RedisStream
from .utils import generate_a_random_hex_str

RUNNER_ID_LENGTH = 4


class WorkerStarter(ABC):

    @abstractmethod
    def start_runner(self, name: str, model_id: str):
        pass


class Command(StrEnum):
    stop = "stop"
    task = "task"


class Runner:

    def __init__(self, rdb: redis.Redis, name: str) -> None:
        self._name = name
        self._rdb = rdb

    @property
    def name(self) -> str:
        return self._name

    @property
    def redis_client(self):
        return self._rdb

    @property
    def model_id(self) -> str:
        return self.redis_client.hget(RedisKeys.runner(self.name), "model_id").decode()

    @property
    def ctime(self) -> datetime:
        resp = self.redis_client.hget(RedisKeys.runner(self.name), "ctime")
        return datetime.fromisoformat(resp.decode())

    @property
    def utime(self) -> datetime:
        resp = self.redis_client.hget(RedisKeys.runner(self.name), "utime")
        return datetime.fromisoformat(resp.decode())

    @utime.setter
    def utime(self, dt: datetime):
        self.redis_client.hset(RedisKeys.runner(self.name), "utime", dt.isoformat())

    @property
    def is_busy(self) -> bool:
        busy = int(self.redis_client.hget(RedisKeys.runner(self.name), "busy"))
        return busy == 1

    @is_busy.setter
    def is_busy(self, busy: bool):
        self.redis_client.hset(RedisKeys.runner(self.name), "busy", 1 if busy else 0)

    @property
    def is_alive(self) -> bool:
        alive = int(self.redis_client.hget(RedisKeys.runner(self.name), "is_alive"))
        return alive == 1

    @is_alive.setter
    def is_alive(self, alive: bool):
        self.redis_client.hset(
            RedisKeys.runner(self.name), "is_alive", 1 if alive else 0
        )

    @property
    def task(self) -> Optional[str]:
        resp = self.redis_client.hget(RedisKeys.runner(self.name), "task")
        return resp.decode() if resp is not None else None

    @task.setter
    def task(self, tid: Optional[str]):
        if tid is None:
            self.redis_client.hdel(RedisKeys.runner(self.name), "task")
        else:
            self.redis_client.hset(RedisKeys.runner(self.name), "task", tid)

    @property
    def heartbeat(self) -> Optional[datetime]:
        resp = self.redis_client.get(RedisKeys.runner_heartbeat(self.name))
        return datetime.fromisoformat(resp.decode()) if resp is not None else None

    @property
    def stream(self) -> RedisStream:
        return RedisStream(
            RedisKeys.runner_stream(self.name),
            RedisKeys.runner_stream_readgroup(self.name),
            connection_pool=self.redis_client.connection_pool,
        )

    def stop(self):
        self.stream.publish({"cmd": Command.stop})

    def run_task(self, tid: str, obj_id: str):
        self.stream.publish({"cmd": Command.task, "tid": tid, "oid": obj_id})

    def update_heartbeat(self, dt: datetime, ttl: float):
        self.redis_client.set(
            RedisKeys.runner_heartbeat(self.name), dt.isoformat(), ex=ttl
        )

    def clean_heartbeat(self):
        self.redis_client.delete(RedisKeys.runner_heartbeat(self.name))


class RunnerPool(redis.Redis):

    def __init__(self, starter: WorkerStarter, **kws):
        super().__init__(**kws)
        self._starter = starter

    def get(self, name: str) -> Optional[Runner]:

        # Try find runner key in redis
        # If key in redis, runner may exists but dead.
        exists = int(self.exists(RedisKeys.runner(name)))
        if exists == 0:
            return None
        return Runner(rdb=redis.Redis(connection_pool=self.connection_pool), name=name)

    def new(self, model_id: str, name: str = None, ctime: datetime = None) -> Runner:
        is_specify_name = name is not None
        if name is None:
            name = generate_a_random_hex_str(length=RUNNER_ID_LENGTH)
        if ctime is None:
            ctime = datetime.now()

        # Check if runner already exists.
        #
        # If runner exists but we not require a name
        # make another name and retry.
        # Else if given a specified name, it can't create new runner.
        runner = self.get(name)
        if runner is not None:
            if is_specify_name:
                raise KeyError(f"runner named {name} already exists")
            while runner is not None:
                name = generate_a_random_hex_str(length=RUNNER_ID_LENGTH)
                runner = self.get(name)

        # Name ok, make a new runner.
        runner = Runner(
            rdb=redis.Redis(connection_pool=self.connection_pool), name=name
        )

        # Write runner metadata.
        self.hset(
            RedisKeys.runner(runner.name),
            mapping={
                "name": name,
                "model_id": model_id,
                "ctime": ctime.isoformat(),
                "utime": ctime.isoformat(),
                "busy": 0,
                "is_alive": 0,
            },
        )
        logger.debug(f"runner data, name [{name}], model id [{model_id}]")

        # Create runner stream and readgroup for command message.
        self.xgroup_create(
            RedisKeys.runner_stream(runner.name),
            RedisKeys.runner_stream_readgroup(runner.name),
            mkstream=True,
        )

        # Start runner worker.
        self._starter.start_runner(name=name, model_id=model_id)
        return runner

    def delete(self, name: str):
        r = self.get(name)
        if r is not None:
            r.stop()
        super().delete(
            RedisKeys.runner(name),
            RedisKeys.runner_heartbeat(name),
            RedisKeys.runner_stream(name),
        )

    def count(self) -> int:
        return len(self._get_all_runner_keys())

    def runners(self) -> List[Runner]:
        result = []

        runner_keys = self._get_all_runner_keys()
        for k in runner_keys:
            r = Runner(
                rdb=redis.Redis(connection_pool=self.connection_pool),
                name=k.decode().removesuffix(f"::{RedisKeys.runner_suffix}"),
            )
            result.append(r)

        return result

    def clean_dead_runners(self):
        for r in self.runners():
            if r.is_alive and r.heartbeat is not None:
                continue
            logger.debug(f"runner [{r.name}] is dead, clean.")
            self.delete(r.name)

    def _get_all_runner_keys(self):
        return self.keys(f"*::{RedisKeys.runner_suffix}")
