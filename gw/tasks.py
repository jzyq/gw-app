from enum import StrEnum
from typing import Dict, List, Optional

import redis

from .models import (
    CreateInferenceTaskRequest,
    InferenceObject,
    InferenceResult,
    TaskResults,
)
from .redis_keys import RedisKeys
from .settings import get_app_settings

_settings = get_app_settings()


class InferenceState(StrEnum):
    pending = "pending"
    running = "running"
    complete = "complelte"
    failed = "failed"


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
    def ttl(self) -> int:
        return int(super().ttl(RedisKeys.task(self.task_id)))

    @property
    def object_list(self) -> List[InferenceObject]:
        return self.raw_request.object_list

    def get_object(self, name: str) -> Optional[InferenceObject]:
        for obj in self.object_list:
            if obj.object_id == name:
                return obj
        return None

    def update_inference_state(
        self, obj: InferenceObject, model: str, state: InferenceState
    ):
        name = RedisKeys.task_inference_state(self.task_id, obj.object_id)
        super().hset(name, mapping={model: str(state)})
        super().expire(name, self.ttl)

    def get_inference_state(self, obj: InferenceObject, model: str) -> InferenceState:
        if model in obj.type_list:
            name = RedisKeys.task_inference_state(self.task_id, obj.object_id)
            res = super().hget(name, model)
            if res is None:
                return InferenceState.pending
            return InferenceState(res)
        raise Exception(f"object no such model in model list {model}")

    def get_inference_result(
        self, obj: InferenceObject, model: str
    ) -> Optional[InferenceResult]:
        name = RedisKeys.task_inference_result(self.task_id, obj.object_id)
        res = super().hget(name, model)
        if res is None:
            return None
        return InferenceResult.model_validate_json(res)

    def set_inference_result(
        self, obj: InferenceObject, model: str, res: InferenceResult
    ):
        name = RedisKeys.task_inference_result(self.task_id, obj.object_id)
        super().hset(name, mapping={model: res.model_dump_json(by_alias=True)})
        super().expire(name, self.ttl)

    def set_postprocess_result(self, res: TaskResults):
        name = RedisKeys.postprocess_result(self.task_id)
        super().set(name, res.model_dump_json(by_alias=True), ex=self.ttl)

    def get_postprocess_result(self) -> Optional[TaskResults]:
        name = RedisKeys.postprocess_result(self.task_id)
        res = super().get(name)
        if res is None:
            return None
        return TaskResults.model_validate_json(res)


class TaskPool(redis.Redis):

    TASK_ID_LENGTH = 16

    def __init__(self, task_ttl: int = _settings.task_lifetime_s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_ttl = task_ttl

    def new(
        self, task_id: str, callback: str, raw_request: CreateInferenceTaskRequest
    ) -> Task:
        self.hset(
            RedisKeys.task(task_id),
            mapping={
                "task_id": task_id,
                "callback": callback,
                "raw_request": raw_request.model_dump_json(by_alias=True),
            },
        )
        self.expire(RedisKeys.task(task_id), self._task_ttl)
        return Task(tid=task_id, connection_pool=self.connection_pool)

    def get(self, task_id: str) -> Optional[Task]:
        exists = int(self.exists(RedisKeys.task(task_id)))
        if exists != 1:
            return None
        return Task(task_id, connection_pool=self.connection_pool)

    def delete(self, task_id: str):
        super().delete(RedisKeys.task(task_id), RedisKeys.postprocess_result(task_id))
