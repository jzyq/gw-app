from datetime import datetime

from loguru import logger
from redis import ConnectionPool, Redis

from .runner import RunnerPool
from .tasks import Task
from .redis_keys import RedisKeys
from abc import ABC, abstractmethod


class Dispatcher(ABC):

    @property
    @abstractmethod
    def runner_num(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def dispatch(self, task: Task):
        raise NotImplementedError()


class ModuleDispatcher(Dispatcher):

    def __init__(
        self,
        runner_pool: RunnerPool = None,
        rdb: Redis = None,
        connection_pool: ConnectionPool = None,
        max_runner: int = 10,
    ) -> None:

        if rdb is not None:
            self._rdb = rdb
        elif connection_pool is not None:
            self._rdb = Redis(connection_pool=connection_pool)
        else:
            raise TypeError("must have at least one redis client")

        if runner_pool is None:
            raise TypeError("must have a runner pool to manage runners.")

        self._runnerpool = runner_pool
        self.runner_num = max_runner

    @property
    def redis_client(self) -> Redis:
        return self._rdb

    @property
    def runner_num(self) -> int:
        return int(self.redis_client.get(RedisKeys.max_runner_num))

    @runner_num.setter
    def runner_num(self, num: int):
        self.redis_client.set(RedisKeys.max_runner_num, num)
        logger.debug(f"set max runner slots number to [{num}]")

    def dispatch(self, task: Task):
        logger.debug(f"dispatch task, id [{task.task_id}], " +
                     f"expect model [{task.model_id}]")

        # Try find a running which run the model task wanted.
        # If have, and it's currently no taks in progress, use this one.
        for r in self._runnerpool.runners():
            if r.model_id == task.model_id and not r.is_busy:
                r.run_task(task.task_id)
                logger.debug(f"find a running worker [{r.name}], " +
                             f"dispatch task [{task.task_id}]")
                return

        # No any runner running this model, start a new one.
        # And dispatch task to the new runner.
        if self._runnerpool.count() < self.runner_num:
            runner = self._runnerpool.new(task.model_id)
            runner.run_task(task.task_id)
            logger.debug(f"no runner running model {task.model_id}, " +
                         f"boot a new runner [{runner.name}], " +
                         f"dispatch task [{task.task_id}]")
            return

        # No any runner running this model, no free slot to start a new one.
        # Try find a runner currently no task in progress,
        # stop this runner to free a slot to start new runner.
        #
        # If have multiple runners idle, choice the oldest one by runer's update time.
        logger.debug(f"no running model [{task.model_id}] and free slot, " +
                     "try free one slot.")
        name = None
        last_utime = datetime.max
        for runner in self._runnerpool.runners():
            if not runner.is_busy:
                if runner.utime < last_utime:
                    last_utime = runner.utime
                    name = runner.name

        if name is not None:
            self._runnerpool.delete(name)
            runner = self._runnerpool.new(task.model_id)
            runner.run_task(task.task_id)
            logger.debug(f"find a idle runner [{name}] can free, stop this. " +
                         f"start a new runner with model [{task.model_id}], " +
                         f"dispatch task [{task.task_id}]")
            return

        # It is too busy to dispatch task currently
        # Just report a error and maybe try again later.
        logger.debug(f"no resource to dispatch task [{task.task_id}]")
        raise Exception("too busy.")


class ProcDispatcher(Dispatcher):

    def __init__(self, rdb: Redis, runner_pool: RunnerPool, max_runner: int = 10):
        self._rdb = rdb
        self._runnerpool = runner_pool
        self._max_runner_num = max_runner
        self._rdb.set(RedisKeys.max_runner_num, max_runner)
        
    @Dispatcher.runner_num.getter
    def runner_num(self) -> int:
        return int(self._rdb.get(RedisKeys.max_runner_num))

    def dispatch(self, task):
        pass