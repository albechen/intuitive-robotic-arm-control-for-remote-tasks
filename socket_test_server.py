import socket
import pickle


def Main():

    host = "***REMOVED***"  # Server ip
    port = 4000

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    print("Server Started")
    while True:
        data, addr = s.recvfrom(1024)
        data = pickle.loads(data)
        print("Message from: " + str(addr))
        print("From connected user: ", data)
    c.close()


if __name__ == "__main__":
    Main()
