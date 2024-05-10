import asyncio
from concurrent.futures import Future
from typing import Callable

import util


async def run_async(partial_functions: list[Callable[[], Future[any]]]):
    async def run():
        tasks = _evalp(partial_functions)
        results = await asyncio.gather(*tasks)
        print("Finished all tasks")
        for i, r in enumerate(results):
            print(f"task {i} -> {r}")

    try:
        await run()
    except BaseException as ex:
        msg = f"Finished tasks because exception {util.message(ex)}"
        raise RuntimeError(msg) from ex


def _evalp(pfs: list[Callable[[], Future[any]]]) -> list:
    evals = []
    for pf in pfs:
        evals.append(pf())
    return evals
