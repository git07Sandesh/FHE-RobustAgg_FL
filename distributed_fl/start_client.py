# --- START OF FILE distributed_fl/start_client.py ---

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flwr.client import start_client
from sec_agg.Plaintext.client_app import client_fn
import torch
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Client ID not provided.")
        sys.exit(1)
        
    cid = int(sys.argv[1])
    print(f"[DEBUG] Client {cid}: Script started.", flush=True)    
    # [REMOVED] The hardcoded run_config is no longer needed here.
    try:
        cuda_ok = torch.cuda.is_available()
        print(f"[DEBUG] Client {cid}: torch.cuda.is_available() check returned: {cuda_ok}", flush=True)
        if not cuda_ok:
            print("[DEBUG] Client {cid}: CUDA not available according to PyTorch.", flush=True)
    except Exception as e:
        print(f"[DEBUG] Client {cid}: Error during torch.cuda.is_available() check: {e}", flush=True)

    print(f"[DEBUG] Client {cid}: Calling client_fn factory...", flush=True)
    client = client_fn(cid)
    print(f"[DEBUG] Client {cid}: client_fn factory returned.", flush=True)


    # Read server host from shared file
    server_host_file = "/homes/01/sxbhattarai/sec-agg/server_host.txt"
    while not os.path.exists(server_host_file):
        print(f"Waiting for server host file: {server_host_file}")
        time.sleep(5)

    with open(server_host_file, "r") as f:
        server_address = f.read().strip()
    
    print(f"[DEBUG] Client {cid}: About to call start_client...", flush=True)
    start_client(server_address=server_address, client=client)
    print(f"[DEBUG] Client {cid}: start_client finished.", flush=True)
