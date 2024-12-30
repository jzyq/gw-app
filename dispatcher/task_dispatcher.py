import signal
import subprocess
import threading

import redis
from loguru import logger

from gw.dispatcher import ProcDispatcher
from gw.runner import RunnerPool, WorkerStarter
from gw.settings import get_app_settings
from gw.streams import Streams
from gw.tasks import TaskPool
from gw.utils import generate_a_random_hex_str


# Use to fork runer process.
class SubprocessStarter(WorkerStarter):
    def start_runner(self, name, model_id):
        subprocess.Popen(["python", "task_runner.py", name, model_id])


def make_signal_handler(evt: threading.Event):
    def handler(signum, frame):
        evt.set()
    return handler


def main():

    settings = get_app_settings()
    logger.info(
        f"init dispatcher, app root {settings.app_root}, models root {settings.pt_model_root}")

    # Connect redis.
    rdb = redis.Redis(host=settings.redis_host,
                      port=settings.redis_port,
                      db=settings.redis_db)
    logger.info(f"connect to redis {settings.redis_port}:{settings.redis_port}, " +
                f"use db {settings.redis_db}")

    # Connect task pool which use to read task data.
    taskpool = TaskPool(connection_pool=rdb.connection_pool,
                        task_ttl=settings.task_lifetime_s)
    logger.info("connect task pool, task lifetime set to ",
                f"{settings.task_lifetime_s} second(s)")

    # Initlize runner pool.
    # We'll use subprocess to start new runner.
    starter = SubprocessStarter()
    runnerpool = RunnerPool(
        connection_pool=rdb.connection_pool, starter=starter)
    logger.info(f"connect runner pool, use {type(starter)} as starter.")

    # Initlize dispatcher.
    dispatcher = ProcDispatcher(rdb=rdb,
                                runner_pool=runnerpool,
                                max_runner=settings.runner_slot_num)
    logger.info(f"init dispatcher, max runner number {dispatcher.runner_num}")

    # Make consumer name to receive message.
    # Recive task create messag from this stream.
    consumer = f"{generate_a_random_hex_str(length=8)}::dispatcher::consumer"
    task_create_stream = Streams(rdb=rdb).task_create
    logger.info(f"use task create stream, stream name {task_create_stream.stream}, " +
                f"readgroup {task_create_stream.readgroup}, consumer {consumer}")

    # Make stop flag and register signal handler.
    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, make_signal_handler(stop_evt))
    signal.signal(signal.SIGINT, make_signal_handler(stop_evt))

    logger.info("start message loop.")
    while not stop_evt.is_set():

        # Pull one message from stream, block 1000 ms,
        # Which give use a chance to check if stop flag set.
        messages = task_create_stream.pull(consumer, count=1, block=1 * 1000)

        # Ignore if no message come.
        if len(messages) == 0:
            continue

        msg = messages[0]
        tid = msg.data["task_id"].decode()
        logger.info(f"receive message {msg.id}, task id {tid}")

        # Ignore if task invalid, but consume this message.
        task = taskpool.get(task_id=tid)
        if task is None:
            msg.ack()
            continue

        # Then dispatch inference task.
        # Clean up dead runner before dispatch task.
        try:
            runnerpool.clean_dead_runners()
            dispatcher.dispatch(task)
            logger.info(f"task dispatch, id {task.task_id}")
            msg.ack()
        except Exception as e:
            # TODO: handle exceptions.
            logger.error(f"dispatch failed, {e}")
            pass

    logger.info("stop message loop, cleanup...")
    rdb.close()


if __name__ == "__main__":
    from gw.utils import initlize_logger

    initlize_logger("dispatcher")

    logger.info("start dispatcher app.")
    main()
    logger.info("dispatcher app shutdown.")
