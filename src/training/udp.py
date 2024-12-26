import socket

host = "0.0.0.0"


def send_and_wait(data: str, port: int, timeout_sec: int) -> str:
    with open_socket(timeout_sec) as udp_socket:
        server_address_port = (host, port)
        buffer_size = 1024
        bytes_to_send = str.encode(data)
        udp_socket.sendto(bytes_to_send, server_address_port)
        msg_from_server = udp_socket.recvfrom(buffer_size)
        return msg_from_server[0].decode("ascii")


def open_socket(timeout_sec: int) -> socket.socket:
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.settimeout(timeout_sec)
    return udp_socket
