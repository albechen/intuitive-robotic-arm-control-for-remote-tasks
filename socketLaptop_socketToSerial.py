# https://stackoverflow.com/questions/11352855/communication-between-two-computers-using-python-socket
### https://github.com/PowerBroker2/pySerialTransfer ####

from pySerialTransfer import pySerialTransfer as txfer
from time import sleep
import pickle
import socket


def Main():

    #### SOCKET CONNECTION ######
    host = "***REMOVED***"  # Server ip
    port = 4000
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    print("Server Started")

    #### ARDUINO CONNECTIOn ######
    link_nema17 = txfer.SerialTransfer("COM3")
    link_nema17.open()
    link_28BYJ = txfer.SerialTransfer("COM4")
    link_28BYJ.open()
    sleep(3)

    while True:
        data, addr = s.recvfrom(1024)
        steps_list = pickle.loads(data)
        print(str(addr), " - RECEIVED: ", steps_list)

        ###################################################################
        # Send a list
        ###################################################################
        send_size_nema17 = 0
        list_size_nema17 = link_nema17.tx_obj(steps_list[0:3])
        if list_size_nema17 is not None:
            send_size_nema17 += list_size_nema17
        else:
            list_size_nema17 = 0
        ###################################################################
        # Transmit all the data to send in a single packet
        ###################################################################
        link_nema17.send(send_size_nema17)
        ###################################################################
        # Wait for a response and report any errors while receiving packets
        ###################################################################
        while not link_nema17.available():
            if link_nema17.status < 0:
                if link_nema17.status == txfer.CRC_ERROR:
                    print("ERROR: CRC_ERROR")
                elif link_nema17.status == txfer.PAYLOAD_ERROR:
                    print("ERROR: PAYLOAD_ERROR")
                elif link_nema17.status == txfer.STOP_BYTE_ERROR:
                    print("ERROR: STOP_BYTE_ERROR")
                else:
                    print("ERROR: {}".format(link_nema17.status))
        ###################################################################
        # Parse response list
        ###################################################################
        rec_list_nema17 = link_nema17.rx_obj(
            obj_type=type(steps_list),
            obj_byte_size=list_size_nema17,
            list_format="i",
        )
        ###################################################################
        # Display the received data
        ###################################################################
        print(
            "NEMA17 --- SENT: {}    RCVD: {}".format(
                steps_list[0:3],
                rec_list_nema17,
            )
        )

        ###################################################################
        # Send a list
        ###################################################################
        send_size_28BYJ = 0
        list_size_28BYJ = link_28BYJ.tx_obj(steps_list[3:7])
        if list_size_28BYJ is not None:
            send_size_28BYJ += list_size_28BYJ
        else:
            list_size_28BYJ = 0
        ###################################################################
        # Transmit all the data to send in a single packet
        ###################################################################
        link_28BYJ.send(send_size_28BYJ)
        ###################################################################
        # Wait for a response and report any errors while receiving packets
        ###################################################################
        while not link_28BYJ.available():
            if link_28BYJ.status < 0:
                if link_28BYJ.status == txfer.CRC_ERROR:
                    print("ERROR: CRC_ERROR")
                elif link_28BYJ.status == txfer.PAYLOAD_ERROR:
                    print("ERROR: PAYLOAD_ERROR")
                elif link_28BYJ.status == txfer.STOP_BYTE_ERROR:
                    print("ERROR: STOP_BYTE_ERROR")
                else:
                    print("ERROR: {}".format(link_28BYJ.status))
        ###################################################################
        # Parse response list
        ###################################################################
        rec_list_28BYJ = link_28BYJ.rx_obj(
            obj_type=type(steps_list),
            obj_byte_size=list_size_28BYJ,
            list_format="i",
        )
        ###################################################################
        # Display the received data
        ###################################################################
        print(
            "28BYJ --- SENT: {}    RCVD: {}".format(
                steps_list[3:6],
                rec_list_28BYJ,
            )
        )


if __name__ == "__main__":
    Main()
