import socket

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

