import select
import socket
from typing import Callable

host = "0.0.0.0"


def send_and_wait(data: str, port: int, timeout_sec: int) -> str:
    server_address_port = (host, port)
    buffer_size = 1024
    print(f"---> {port} Sendig: {data}")
    my_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    my_socket.settimeout(timeout_sec)
    try:
        bytes_to_send = str.encode(data)
        my_socket.sendto(bytes_to_send, server_address_port)
        msg_from_server = my_socket.recvfrom(buffer_size)
        return msg_from_server[0].decode("ascii")
    finally:
        my_socket.close()
        print(f"---- {port} Socket closed")


def open_socket(port: int, handler: Callable[[str], str]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, port))
        print(f"---- Server listening on {host}:{port}")
        TIMEOUT = 3
        active = True
        while active:
            # wait for request
            readable, writable, exceptional = select.select([sock], [], [], TIMEOUT)
            print(f"---- readable - {readable}, {type(readable)}")
            print(f"---- writable - {writable}, {type(writable)}")
            print(f"---- exeptional - {exceptional}, {type(exceptional)}")
            if readable:
                data, address = sock.recvfrom(1024)
                msg = data.decode("utf-8")
                print(f"---- Received data: {msg} from {address}")
                resp = handler(msg)
                bytes_to_send = str.encode(resp)
                sock.sendto(bytes_to_send, address)

            else:
                print(f"---- No data received for {TIMEOUT} s --> EXIT")
                active = False
