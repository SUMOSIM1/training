from dataclasses import dataclass
from time import sleep

import simdb as db
import udp
import util


@dataclass
class StatusIndicator:
    running: bool = True


def tryout_all_sims():
    with db.create_client() as c:
        for s in db.find_all(c):
            print(s)


def tryout_concurrent():
    print("---> concurrent")
    indicator = StatusIndicator()
    functions = [
        (loop, "A", 5, 1.0, indicator),
        (loop1, "B", 2, 0.2, indicator),
        (loop, "C", 20, 0.4, indicator),
        (loop, "D", 4, 0.1, indicator),
        (loop_master, "E", 3, 0.2, indicator),
    ]
    results = util.run_concurrent(functions, 4)
    for result in results:
        print(f"<--- result {result}")


def loop(name: str, n: int, wait: float, indicator) -> str:
    i = 0
    for i in range(n + 1):
        sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")
    return f"# finished {name} {i} of {n}"


def loop1(name: str, n: int, wait: float, indicator):
    i = 0
    for i in range(n + 1):
        sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")


def loop_err(name: str, n: int, wait: float, indicator) -> str:
    i = 0
    for i in range(n + 1):
        sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")
    raise RuntimeError(f"# exception {name} {i} of {n}")


def loop_master(name: str, n: int, wait: float, indicator) -> str:
    i = 0
    for i in range(n + 1):
        sleep(wait)
        if not indicator.running:
            return f"# finished because indication {name} {i} of {n}"
        print(f"- - - {name} {i} of {n}")
    indicator.running = False
    return f"# master finished {name} {i} of {n}"


tryout_port = 6000


def tryout_client():
    print("---> tryout client")
    udp.send_and_wait("hallo", tryout_port)


def tryout_server():
    def handler1(msg: str) -> str:
        sleep(0.5)
        return f"response to <{msg}>"

    print("---> tryout server")
    udp.open_socket(tryout_port, handler1)


def main():
    # tryout_all_sims()
    tryout_client()
