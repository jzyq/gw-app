import signal
import threading

import redis
from loguru import logger

from gw.models import ComposedResult, TaskResults
from gw.settings import get_app_settings
from gw.streams import Streams
from gw.tasks import InferenceState, Task, TaskPool
from gw.utils import generate_a_random_hex_str


def make_signal_handler(evt: threading.Event):
    def handler(signum, frame):
        evt.set()

    return handler


def compose_results(task: Task) -> bool:
    # Check if all inference under a object are completed.
    # And try to compose inference result.
    composed_results = []

    # Check each object and each model.
    for obj in task.object_list:

        # Use to track inference results.
        results = []

        for model in obj.type_list:
            if task.get_inference_state(obj, model) == InferenceState.complete:
                res = task.get_inference_result(obj, model)
                if res is None:
                    return False
                results.append(res)

            # If any inference not complete, compose failed.
            else:
                return False

        # Compose inference result about this object, push to result list.
        compose_res = ComposedResult(objectId=obj.object_id, results=results)
        composed_results.append(compose_res)

    # So all inference are completed.
    # Write final result into redis.
    res = TaskResults(requestId=task.task_id, requestList=composed_results)
    task.set_postprocess_result(res)

    # Return successed.
    return True


def main():

    # Connect to redis.
    settings = get_app_settings()
    rdb = redis.Redis(
        host=settings.redis_host, port=settings.redis_port, db=settings.redis_db
    )
    logger.info(
        f"connect redis {settings.redis_host}:{settings.redis_port}, "
        + f"use db {settings.redis_db}"
    )

    # Connect message streams, and make a consumer name.
    # in stream use to pull message from runner to notify that inference complete.
    # out stream use to send postprocess complete message to next step,
    consumer = f"{generate_a_random_hex_str(length=8)}::postprocess::consumer"
    streams_maker = Streams(connection_pool=rdb.connection_pool)
    in_stream = streams_maker.task_inference_complete
    out_stream = streams_maker.task_finish
    logger.info(
        f"use input stream {in_stream.stream}, readgroup {in_stream.stream}, "
        + f"consumer name {consumer}. "
        + f"output stream {out_stream.stream}, readgroup {out_stream.readgroup}."
    )

    # Connect to task pool.
    taskpool = TaskPool(connection_pool=rdb.connection_pool)

    # Make stop flag and register signal handler.
    stop_flag = threading.Event()
    signal.signal(signal.SIGTERM, make_signal_handler(stop_flag))
    signal.signal(signal.SIGINT, make_signal_handler(stop_flag))

    logger.info("register signal handler and start message loop.")
    while not stop_flag.is_set():

        # Pull a message from input stream, block wait 1000 ms.
        # If no message we just continue.
        # This is in order to check if stop flag was set.
        # and if set we will stop loop.
        messages = in_stream.pull(consumer, count=1, block=1 * 1000)
        if len(messages) == 0:
            continue

        msg = messages[0]
        mid = msg.id
        tid = msg.data["task_id"].decode()
        logger.info(f"message received, message id {mid}, task id {tid}")

        # Read task data from task pool,
        # if task invalid, ignore and consume message.
        task = taskpool.get(tid)
        if task is None:
            msg.ack()
            continue

        # compose results.
        # if complete, notify next processer, or ignore.
        if compose_results(task):
            out_stream.publish({"task_id": task.task_id})
            logger.info(f"task {task.task_id} result compose complete")

        msg.ack()

    # Clean worker pool, terminate all working process.
    # Any unfinished post process will be drop.
    logger.info("recieve stop signal, cleanup...")
    rdb.close()


if __name__ == "__main__":
    from gw.utils import initlize_logger

    initlize_logger("postprocess")

    logger.info("start post process app...")
    main()
    logger.info("post process app shutdown.")
