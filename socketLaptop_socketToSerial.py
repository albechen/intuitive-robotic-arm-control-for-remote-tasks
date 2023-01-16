from pySerialTransfer import pySerialTransfer as txfer
from time import sleep


link_nema17 = txfer.SerialTransfer("COM5")
link_nema17.open()
link_28BYJ = txfer.SerialTransfer("COM6")
link_28BYJ.open()

sleep(3)  # allow some time for the Arduino to completely reset


### https://github.com/PowerBroker2/pySerialTransfer ####
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
list_size_28BYJ = link_28BYJ.tx_obj(steps_list[3:6])
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
