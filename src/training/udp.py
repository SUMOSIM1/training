import socket
import threading


host = "0.0.0.0"

send_lock = threading.Lock()


def send_and_wait(data: str, port: int, timeout_sec: int) -> str:
    try:
        send_lock.acquire_lock()
        udp_socket = open_socket(timeout_sec)
        server_address_port = (host, port)
        buffer_size = 1024
        bytes_to_send = str.encode(data)
        udp_socket.sendto(bytes_to_send, server_address_port)
        msg_from_server = udp_socket.recvfrom(buffer_size)
        result = msg_from_server[0].decode("ascii")
    except TimeoutError as ex:
        print(f"Timout error on port:{port}")
        raise ex
    finally:
        send_lock.release_lock()
    return result


def open_socket(timeout_sec: int) -> socket.socket:
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.settimeout(timeout_sec)
    return udp_socket
