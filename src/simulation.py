import flwr as fl
import argparse
import os
import logging
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def client_fn(cid):
    """Create and return a client instance"""
    from client import FLClient
    train_txt = os.path.join("data", "partitions", f"client_{cid}_train.txt")
    test_txt = os.path.join("data", "partitions", "test.txt")
    return FLClient(int(cid), train_txt, test_txt, "cpu").to_client()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()

    # Configure strategy
    strategy = FedAvg(
        min_available_clients=args.num_clients,
        min_fit_clients=args.num_clients,
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs}
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )

if __name__ == "__main__":
    main()