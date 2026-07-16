

from argparse import ArgumentParser
import os
import sys
import time
import subprocess
import zmq

def main():
    ap = ArgumentParser()
    ap.add_argument("--hosts", type=str, nargs="+", required=True)
    ap.add_argument("--nwork", type=int, help="GPU devices per node (default=1)", default=1)
    args = ap.parse_args()
    try:
        hosts = args.hosts
    except RuntimeError as e:
        print(f"Error: {e}")
        print("For testing outside SLURM, hardcoding ['localhost']...")
        hosts = ["localhost"]

    base_port = 5550   # Matches worker_daemon.py port-base
    
    context = zmq.Context()
    
    # Connect to all workers and register them in a Poller
    # We map socket -> endpoint_string to keep track of who is who
    worker_sockets = {}
    poller = zmq.Poller()
    
    print("Connecting to persistent GPU workers...")
    for host in hosts:
        for local_id in range(args.nwork):
            endpoint = f"tcp://{host}:{base_port + local_id}"
            
            # REQ socket to send tasks to the worker's REP socket
            sock = context.socket(zmq.REQ)
            sock.connect(endpoint)
            
            worker_sockets[sock] = endpoint
            # Register the socket with the poller to monitor for incoming ACKs (responses)
            poller.register(sock, zmq.POLLIN)
            print(f" -> Connected to worker: {endpoint}")

    # Track which workers are idle and ready for a task
    # Initially, we assume all workers are idle and ready
    idle_workers = list(worker_sockets.keys())
    busy_workers = {} # Maps socket -> current_task metadata
    
    print(f"\nInitialized {len(idle_workers)} workers. Ready to process events.")
    
    # Simulate or initialize your psana stream
    # import psana
    # ds = psana.DataSource(......)
    # event_stream = ds.events()
    
    # this will actually need to be dynamic, reading from rghte FFB
    total_events = 500
    event_stream = ({"run": 123, "event": i} for i in range(total_events))
    # ------------------------------------------------------------------

    events_sent = 0
    events_completed = 0
    start_time = time.time()

    try:
        # Loop runs as long as there are events to send OR workers still processing
        while events_sent < total_events or len(busy_workers) > 0:
            
            # If we have idle workers AND events left to process, assign them immediately
            while idle_workers and events_sent < total_events:
                worker_sock = idle_workers.pop(0)
                event_data = next(event_stream)
                
                # Send task to worker (non-blocking call)
                worker_sock.send_json(event_data)
                
                busy_workers[worker_sock] = event_data
                events_sent += 1

            # Wait (poll) for any busy worker to finish and reply with "READY"
            # We timeout after 10 milliseconds to keep the loop highly responsive
            socks = dict(poller.poll(timeout=10))
            
            for sock in socks:
                if socks[sock] == zmq.POLLIN:
                    # Receive the "READY" acknowledgment from the worker
                    reply = sock.recv_json()
                    
                    # Remove from busy pool and return to idle pool
                    completed_task = busy_workers.pop(sock)
                    idle_workers.append(sock)
                    events_completed += 1
                    
                    if events_completed % 50 == 0 or events_completed == total_events:
                        elapsed = time.time() - start_time
                        rate = events_completed / elapsed
                        print(f"Progress: {events_completed}/{total_events} events processed. ({rate:.1f} ev/sec)")

    except KeyboardInterrupt:
        print("\nShutdown requested by user. Stopping master...")
    finally:
        # Clean up sockets
        print("Closing connections.")
        for sock in worker_sockets.keys():
            sock.close()
        context.term()

if __name__ == "__main__":
    main()
