"""Runs on a Grid'5000 frontend (stdlib only, one light API sweep); prints JSON.

Summarises every GPU cluster of every site from the Reference API: GPU model, memory,
compute capability, GPUs per node, node count, queues and exotic flag.
"""

import json
import urllib.request

API = "https://api.grid5000.fr/stable"


def get(path):
    with urllib.request.urlopen(f"{API}/{path}", timeout=60) as response:
        return json.load(response)


def cluster_summary(site, cluster):
    nodes = get(f"sites/{site}/clusters/{cluster}/nodes")["items"]
    gpu_nodes = [n for n in nodes if n.get("gpu_devices")]
    if not gpu_nodes:
        return None
    node = gpu_nodes[0]
    gpus = list(node["gpu_devices"].values())
    first = gpus[0]
    major, _, minor = str(first.get("compute_capability", "0.0")).partition(".")
    return {
        "site": site,
        "cluster": cluster,
        "gpu": first.get("model", "unknown"),
        "vendor": first.get("vendor", ""),
        "memory_mib": int(first.get("memory", 0)) // (1024 * 1024),
        "compute_capability": [int(major or 0), int(minor or 0)],
        "gpus_per_node": len(gpus),
        "nodes": len(gpu_nodes),
        "queues": node.get("supported_job_types", {}).get("queues", []),
        "exotic": bool(node.get("exotic")),
        "max_walltime": node.get("supported_job_types", {}).get("max_walltime"),
    }


def main():
    out = []
    for site in (s["uid"] for s in get("sites")["items"]):
        for cluster in (c["uid"] for c in get(f"sites/{site}/clusters")["items"]):
            summary = cluster_summary(site, cluster)
            if summary:
                out.append(summary)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
