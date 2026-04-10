"""Entry point for abstract optimization experiments.

Handles vLLM lifecycle, scorer setup, method dispatch, and result saving.
Every candidate is scored by every scorer. The optimize target determines selection.

Usage:
    python run_optimize.py --config configs/bon_llama_cot.yaml
"""
import argparse
import atexit
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from serve import launch_server, wait_for_health, find_free_port
from cot_scorer import score as cot_score

RESULTS_DIR = "/nlp/scr/nathu/latent_rewrite/results"


def make_client(endpoint):
    """Create OpenAI client from endpoint string."""
    if endpoint == "openai":
        return OpenAI()
    else:
        base_url = endpoint.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return OpenAI(base_url=base_url, api_key="unused")


@dataclass
class Scorer:
    """A configured scorer: client + model + judge settings."""
    name: str
    client: OpenAI
    model: str
    judge: str
    cot: bool
    k_select: int
    k_eval: int
    temperature: float
    top_p: float = None

    def score(self, title, abstract):
        return cot_score(self.client, self.model, title, abstract,
                         judge=self.judge, cot=self.cot,
                         k_select=self.k_select, k_eval=self.k_eval,
                         temperature=self.temperature, top_p=self.top_p)


def load_papers(data_path, limit=None, shuffle=True, seed=42):
    """Load papers from JSON or parquet."""
    if data_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(data_path)
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        papers = df.to_dict("records")
    else:
        with open(data_path) as f:
            papers = json.load(f)
    if limit:
        papers = papers[:limit]
    return papers


def setup_servers_and_scorers(cfg):
    """Parse config, launch vLLM servers as needed, return (scorers_dict, procs).

    Servers are deduplicated by (model, port) — multiple scorers can share one.
    """
    procs = []
    # Track launched servers: gpu -> (port, client)
    active_servers = {}

    def get_client(server_cfg):
        """Get or create a client for a server config."""
        endpoint = server_cfg.get("endpoint", "openai")

        if endpoint == "openai":
            return make_client("openai")

        if endpoint == "launch":
            gpu = server_cfg["gpu"]
            # Normalize to a hashable key for dedup
            gpu_key = tuple(gpu) if isinstance(gpu, list) else gpu
            if gpu_key in active_servers:
                return active_servers[gpu_key][1]

            model = server_cfg["model"]
            port = find_free_port()
            gpu_mem = server_cfg.get("gpu_memory_utilization", 0.90)
            max_len = server_cfg.get("max_model_len", 4096)

            proc = launch_server(model, gpu, port, gpu_mem, max_len)
            if not wait_for_health(port, 300, model, proc):
                proc.terminate()
                sys.exit(1)

            client = make_client(f"http://localhost:{port}")
            active_servers[gpu_key] = (port, client)
            procs.append(proc)
            return client

        if endpoint == "existing":
            if not active_servers:
                raise RuntimeError("No server launched yet")
            # If only one server, just use it
            if len(active_servers) == 1:
                return next(iter(active_servers.values()))[1]
            # Multiple servers — need gpu to disambiguate
            gpu = server_cfg.get("gpu")
            if gpu is None:
                raise RuntimeError("Multiple servers running, specify 'gpu' to pick one")
            for key, (port, client) in active_servers.items():
                if key == gpu or (isinstance(key, tuple) and gpu in key):
                    return client
            raise RuntimeError(f"No server found for GPU {gpu}")

        # Direct URL
        return make_client(endpoint)

    # Build scorers
    scorers = {}
    for name, scorer_cfg in cfg["scorers"].items():
        client = get_client(scorer_cfg)
        scorers[name] = Scorer(
            name=name,
            client=client,
            model=scorer_cfg["model"],
            judge=scorer_cfg.get("judge", "harsh_nodim"),
            cot=scorer_cfg.get("cot", True),
            k_select=scorer_cfg.get("k_select", 5),
            k_eval=scorer_cfg.get("k_eval", 5),
            temperature=scorer_cfg.get("temperature", 0.6),
            top_p=scorer_cfg.get("top_p", None),
        )

    return scorers, procs, get_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup scorers + servers
    scorers, procs, get_client = setup_servers_and_scorers(cfg)
    atexit.register(lambda: [p.terminate() for p in procs if p.poll() is None])

    # Rewriter client (reuse get_client so it can find existing servers)
    rw_cfg = cfg.get("rewriter", {})
    rw_client = get_client(rw_cfg)
    rw_model = rw_cfg.get("model", "gpt-5-mini")

    # Data
    data_cfg = cfg.get("data", {})
    papers = load_papers(
        data_cfg.get("path", "data/iclr2026_subsample.parquet"),
        limit=data_cfg.get("limit"),
    )
    print(f"Loaded {len(papers)} papers")

    # Optimize config
    opt_cfg = cfg["optimize"]
    target_names = opt_cfg["target"]
    if isinstance(target_names, str):
        target_names = [target_names]
    aggregate = opt_cfg.get("aggregate", "mean")

    # Method config
    method_cfg = cfg["method"]
    method = method_cfg["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name", f"{method}_{timestamp}")

    # Incremental save
    out_path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    all_results = []

    def save():
        with open(out_path, "w") as f:
            json.dump({"config": cfg, "results": all_results}, f, indent=2)

    if method == "bon":
        from methods.bon import run_bon
        results = run_bon(
            rewriter_client=rw_client,
            rewriter_model=rw_model,
            scorers=scorers,
            target_names=target_names,
            aggregate=aggregate,
            papers=papers,
            n=method_cfg.get("n", 16),
            style=method_cfg.get("style", "open"),
            goal=method_cfg.get("goal", "diverse"),
            rewrite_temperature=method_cfg.get("rewrite_temperature", 1.0),
            on_paper_done=lambda i, r: (all_results.append(r), save()),
        )
    elif method == "opro":
        raise NotImplementedError("OPRO coming soon")
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\nDone! Results saved to {out_path}")
    print(f"  Papers: {len(results)}")


if __name__ == "__main__":
    main()
