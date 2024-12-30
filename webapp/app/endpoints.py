import redis
from fastapi import APIRouter, Request
from fastapi.responses import Response
from loguru import logger

from gw.settings import AppSettings
from gw.streams import RedisStream
from gw.tasks import TaskPool

from gw import models

router = APIRouter()


def get_global_config(req: Request) -> AppSettings:
    return req.app.state.app_settings


def get_task_pool(req: Request) -> TaskPool:
    return req.app.state.taskpool


def get_task_create_stream(req: Request) -> RedisStream:
    return req.app.state.stream


@router.post("/picAnalyse")
async def create_task(task: models.CreateInferenceTaskRequest, req: Request):

    logger.info("receive inference request.")
    logger.debug(f"request body: {task.model_dump()}")

    # Make callback url, it has a static form/.
    callback = "http://{}:{}/picAnalyseRetNotify".format(
        task.request_host_ip, task.request_host_port
    )
    logger.info(f"callback path: {callback}")

    try:
        t = get_task_pool(req).new(
            task_id=task.request_id, callback=callback, raw_request=task
        )
        get_task_create_stream(req).publish({"task_id": t.task_id})
        logger.info(f"push new inference task to queue. request id: {task.request_id}")

    except redis.ConnectionError as e:
        logger.error(f"create new task error, {str(e)}")
        return Response(content="redis disconnected", status_code=500)

    return Response(status_code=200)
