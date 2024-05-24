import socket
import queue
import threading

# Define the server's hostname and port number
server_hostname = 'localhost'  # Listen on all available interfaces
server_port = 12346
directory = "ls24_live_dataset/last_phase"

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((server_hostname, server_port))

# Listen for incoming connections
server_socket.listen(1)

log_file = open("ls-output-log.txt", "w")
log_file.write("listening on port 12346\n")
log_file.flush()

# Shared queue for received messages
message_queue = queue.Queue()

is_exit = False

import re

# coalescing repetitive tokens
token_pattern = r'\b(\d\d)(?:\s+\1){4,}\b'
token_replacement = r'\1 9'

# coalescing repetitive time tokens
time_pattern = r'\b(\d)(?:\s+\1){4,}\b'
time_replacement = r'\1 8'

def coalesce_sentence(sentence):
    token_replaced = re.sub(token_pattern, token_replacement, sentence)
    time_replaced = re.sub(time_pattern, time_replacement, token_replaced)
    return time_replaced

def handle_connection(connection, client_address):
    try:
        log_file.write(f"Connection from {client_address}\n")
        received_data = b""
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            received_data += chunk

        received_messages = received_data.decode("utf-8").split("\n")
        if received_messages[-1] == "":
            received_messages = received_messages[:-1]
        
        for message in received_messages:
            message_queue.put(message)

    finally:
        connection.close()

def consume_messages():
    global is_exit
    while True:
        message = message_queue.get()
        
        if message.startswith("exit"):
            log_file.write("exiting\n")
            log_file.flush()
            is_exit = True
        start_time, end_time, ip1, ip2, msg = message.split(",")
        msg = coalesce_sentence(msg).rstrip()
        file_name = directory + "/" + ip1 + ".txt"
        with open(file_name, "a") as f:
            f.write(ip2 + "," + start_time + "," + end_time + "," + msg + "\n")
            f.flush()
        
consumer_thread = threading.Thread(target=consume_messages)
consumer_thread.start()

while not is_exit:
    connection, client_address = server_socket.accept()
    connection_thread = threading.Thread(target=handle_connection, args=(connection, client_address))
    connection_thread.start()
