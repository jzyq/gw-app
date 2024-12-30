from typing import Optional

import redis

from .models import CreateInferenceTaskRequest
from .redis_keys import RedisKeys
from .settings import get_app_settings

_settings = get_app_settings()


class Task(redis.Redis):

    def __init__(self, tid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tid = tid

    @property
    def task_id(self) -> str:
        return self._tid

    @property
    def callback(self) -> str:
        return self.hget(RedisKeys.task(self.task_id), "callback").decode()

    @property
    def raw_request(self) -> CreateInferenceTaskRequest:
        data = self.hget(RedisKeys.task(self.task_id), "raw_request").decode()
        return CreateInferenceTaskRequest.model_validate_json(data)

    @property
    def inference_result(self) -> Optional[str]:
        res: bytes = self.get(RedisKeys.inference_result(self.task_id))
        return res.decode() if res is not None else res

    @property
    def ttl(self) -> int:
        return int(super().ttl(RedisKeys.task(self.task_id)))

    @inference_result.setter
    def inference_result(self, data: str):
        self.set(RedisKeys.inference_result(self.task_id), data, ex=self.ttl)


class TaskPool(redis.Redis):

    TASK_ID_LENGTH = 16

    def __init__(self, task_ttl: int = _settings.task_lifetime_s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_ttl = task_ttl

    def new(self, task_id: str, callback: str, raw_request: CreateInferenceTaskRequest) -> Task:
        self.hset(RedisKeys.task(task_id), mapping={
            "task_id": task_id,
            "callback": callback,
            "raw_request": raw_request.model_dump_json()
        })
        self.expire(RedisKeys.task(task_id), self._task_ttl)
        return Task(tid=task_id, connection_pool=self.connection_pool)

    def get(self, task_id: str) -> Optional[Task]:
        exists = int(self.exists(RedisKeys.task(task_id)))
        if exists != 1:
            return None
        return Task(task_id, connection_pool=self.connection_pool)

    def delete(self, task_id: str):
        super().delete(RedisKeys.task(task_id),
                       RedisKeys.inference_result(task_id),
                       RedisKeys.postprocess_result(task_id))
