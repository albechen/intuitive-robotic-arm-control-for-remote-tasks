import socket
import pickle


def Main():

    host = "***REMOVED***"  # client ip
    port = 4005

    server = ("***REMOVED***", 4000)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    d = [1, 392, 19238]
    msg = pickle.dumps(d)

    s.sendto(msg, server)
    s.close()


if __name__ == "__main__":
    Main()
