import socket
import pickle


def Main():

    host = "192.168.0.160"  # client ip
    port = 4005

    server = ("192.168.0.154", 4000)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    d = [1, 392, 19238]
    msg = pickle.dumps(d)

    s.sendto(msg, server)
    s.close()


if __name__ == "__main__":
    Main()
