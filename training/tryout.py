import asyncio
import functools as ft
from dataclasses import dataclass

import asyncrunner as ar
import simdb as db


@dataclass
class StatusIndicator:
    running: bool = True


def all_sims():
    with db.create_client() as c:
        for s in db.find_all(c):
            print(s)


def start_async():
    indicator = StatusIndicator()
    partials = [
        ft.partial(loop1, "A", 50, 0.1, indicator),
        ft.partial(loop1, "B", 2, 0.07654, indicator),
        ft.partial(loop_err, "C", 3, 0.06654, indicator),
        ft.partial(loop, "D", 10, 0.08654, indicator),
        ft.partial(loop_master, "E", 45, 0.044, indicator),
        ft.partial(loop, "F", 2, 0.02654, indicator),
    ]
    asyncio.run(ar.run_async(partials))


async def loop(name: str, n: int, wait: float, indicator) -> str:
    for i in range(n + 1):
        await asyncio.sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")
    return f"# finished {name} {i} of {n}"


async def loop1(name: str, n: int, wait: float, indicator):
    for i in range(n + 1):
        await asyncio.sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")


async def loop_err(name: str, n: int, wait: float, indicator) -> str:
    for i in range(n + 1):
        await asyncio.sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")
    raise RuntimeError(f"# exception {name} {i} of {n}")


async def loop_master(name: str, n: int, wait: float, indicator) -> str:
    for i in range(n + 1):
        await asyncio.sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")
    indicator.running = False
    return f"# master finished {name} {i} of {n}"


def main():
    start_async()
