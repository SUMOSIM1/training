import socket
from dataclasses import dataclass

@dataclass
class Config():
    host: str = "127.0.0.0"
    port: int = 4445

def sendAndWait(data: str, config: Config) -> str:
    print(f"---> Sendig: {data}")
    msgFromClient = "start"
    bytesToSend = str.encode(msgFromClient)
    serverAddressPort = (config.host, config.port)
    bufferSize = 1024
    client = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    client.settimeout(2)
    try:
        client.sendto(bytesToSend, serverAddressPort)
        msgFromServer = client.recvfrom(bufferSize)
        bmsg = msgFromServer[0].decode('ascii')
        print(f"---> Response from server {bmsg}")
    except socket.timeout:
        print("---- timeout closing socket")
    finally:
        client.close()


if __name__ == "__main__":
    msgFromClient = "start"
    bytesToSend = str.encode(msgFromClient)
    serverAddressPort = ("127.0.0.1", 4445)
    bufferSize = 1024
    client = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    client.settimeout(2)
    try:
        client.sendto(bytesToSend, serverAddressPort)
        while True:
            msgFromServer = client.recvfrom(bufferSize)
            bmsg = msgFromServer[0].decode('ascii')
            print(bmsg)
            if bmsg == 'stop':
                break
            answer = f"Answer for: [{bmsg}]"
            bytesToSend = str.encode(answer)
            client.sendto(bytesToSend, serverAddressPort)
    except socket.timeout:
        print("timeout closing socket")
    finally:
        client.close()

