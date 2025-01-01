import json
import signal
import threading

import redis
import requests
from loguru import logger

from gw.settings import get_app_settings
from gw.streams import Streams
from gw.tasks import TaskPool
from gw.utils import generate_a_random_hex_str
from gw.utils import initlize_logger
from gw.models import TaskResults

def make_signal_handler(evt: threading.Event):
    def handler(signum, frame):
        evt.set()
    return handler


def main():
    settings = get_app_settings()

    initlize_logger("notifier")

    # Connect redis.
    rdb = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db
    )
    logger.info(f"connect redis {settings.redis_host}:{settings.redis_port}, " +
                f"use db {settings.redis_db}")

    # Connect task pool.
    taskpool = TaskPool(
        connection_pool=rdb.connection_pool,
        task_ttl=settings.task_lifetime_s)

    # Connect stream. receive message from task finish stream.
    # Make a unique consumer name.
    consumer = f"{generate_a_random_hex_str(length=8)}::notifier::consumer"
    stream = Streams(connection_pool=rdb.connection_pool).task_finish
    logger.info(f"use stream {stream.stream} receive message, readgroup {stream.readgroup}, " +
                f"consumer name {consumer}")

    # Make loop stop flag and register signal handler
    stop_flag = threading.Event()
    signal.signal(signal.SIGTERM, make_signal_handler(stop_flag))
    signal.signal(signal.SIGINT, make_signal_handler(stop_flag))

    logger.info("start event loop.")
    while not stop_flag.is_set():

        # Pull one message from stream once, block wating 1000 ms.
        # Which in order to check if stop flag set every 1 second.
        # If stop flag set, stop message loop.
        messages = stream.pull(consumer, count=1, block=1 * 1000)

        # Ignore when timeout or stream broken.
        if len(messages) == 0:
            continue

        msg = messages[0]
        mid = msg.id
        tid = msg.data["task_id"].decode()
        logger.info(f"recieve message {mid}, task id {tid}")

        # Read task data, ignore if task invalid
        # Need consume this message.
        task = taskpool.get(tid)
        if task is None:
            msg.ack()
            continue

        try:
            res = task.get_postprocess_result()
            resp = requests.post(task.callback, json=res.model_dump(by_alias=True))
        except requests.exceptions.InvalidURL as e:
            logger.error(f"invalid url: {e}")
            # TODO: task have a invalid callback url, what should do ?
            msg.ack()
            continue
        except requests.exceptions.ConnectionError as e:
            logger.error(f"send request error: {e}")
            
            # Ack this message to prevent retry. We will retry later.
            msg.ack()

            # TODO: mark this task, it need retry callback.
            continue

        if resp.status_code == 200:
            logger.info(f"call {task.callback} send result.")
            msg.ack()
        else:
            # TODO: handle request failed.
            msg.ack()
            pass

        # Every things ok, delete this task and its results.
        taskpool.delete(task.task_id)

    # Stop loop, do cleanup.
    logger.info("recieve stop signal, cleanup...")
    rdb.close()


if __name__ == "__main__":
    logger.info("start norifier app...")
    main()
    logger.info("notifier app shutdown.")
