from enum import StrEnum


class RedisKeys(StrEnum):

    max_runner_num = "runner_num::dispatcher::gw"

    def postprocess_result(x): return f"{x}::postprocess:result::gw"

    task_suffix = "task::gw"
    def task(x): return f"{x}::task::gw"
    def task_inference_state(x, o): return f"state::{o}::{x}::task::gw"
    def task_inference_result(x, o): return f"result::{o}::{x}::task::gw"

    runner_suffix = "runner::gw"
    def runner(x): return f"{x}::runner::gw"
    def runner_heartbeat(x): return f"{x}::runner::heartbeat::gw"
    def runner_stream(x): return f"{x}::runner::stream::gw"
    def runner_stream_readgroup(x): return f"{x}::runner::readgroup::gw"

    stream_task_create = "task_create::stream::gw"
    stream_readgroup_task_create = "task_create::readgroup::gw"

    stream_inference_complete = "inference_complete::stream::gw"
    stream_readgroup_inference_complete = "inference_complete::readgroup::gw"

    stream_postprocess_complete = "postprocess_complete::stream::gw"
    stream_readgroup_postprocess_complete = "postprocess_complete::readgroup::gw"
