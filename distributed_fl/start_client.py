# --- START OF FILE distributed_fl/start_client.py ---

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flwr.client import start_client
from sec_agg.Plaintext.client_app import client_fn

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Client ID not provided.")
        sys.exit(1)
        
    cid = int(sys.argv[1])
    
    # [REMOVED] The hardcoded run_config is no longer needed here.
    
    # Create the client instance using only the client ID
    client = client_fn(cid)

    # Read server host from shared file
    server_host_file = "/homes/01/sxbhattarai/sec-agg/server_host.txt"
    while not os.path.exists(server_host_file):
        print(f"Waiting for server host file: {server_host_file}")
        time.sleep(5)

    with open(server_host_file, "r") as f:
        server_address = f.read().strip()
    
    print(f"Client {cid} connecting to server at {server_address}")
    start_client(server_address=server_address, client=client)
