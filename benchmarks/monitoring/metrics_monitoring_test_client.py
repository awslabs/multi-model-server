#!/usr/bin/env python3

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Test client for MMS Metrics Monitoring Server
"""

import sys, socket, select


def client():
    host = sys.argv[1]
    port = int(sys.argv[2])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((host, port))
    except:
        print('Unable to connect')
        sys.exit()

    print('Connected to remote host. You can start sending messages')
    sys.stdout.write('[Me] ');
    sys.stdout.flush()

    while 1:
        socket_list = [sys.stdin, s]

        # Get the list sockets which are readable
        read_sockets, write_sockets, error_sockets = select.select(socket_list, [], [])

        for sock in read_sockets:
            if sock == s:
                # incoming message from remote server, s
                data = sock.recv(4096).decode()
                if not data:
                    print('\nDisconnected from the server')
                    sys.exit()
                else:
                    # print data
                    sys.stdout.write(data)
                    sys.stdout.write('[Me] ');
                    sys.stdout.flush()

            else:
                # user entered a message
                msg = sys.stdin.readline()
                s.send(msg.encode("latin-1"))
                sys.stdout.write('[Me] ');
                sys.stdout.flush()


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print('Usage : python chat_client.py hostname port')
        sys.exit()

    sys.exit(client())
