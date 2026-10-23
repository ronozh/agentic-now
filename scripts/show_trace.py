"""Print the latest Langfuse trace (or a given trace id) as a tree.

Run: uv run python scripts/show_trace.py [trace_id]
"""

import sys

from langfuse import get_client

import agentic_now.config  # noqa: F401  (loads .env so the Langfuse client gets its keys)


def main() -> None:
    api = get_client().api
    trace_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else api.observations.get_many(limit=1, is_root_observation=True)
        .data[0]
        .trace_id
    )
    observations = api.observations.get_many(
        trace_id=trace_id, limit=100, fields="core,basic,usage,model"
    ).data
    children: dict[str | None, list] = {}
    for obs in sorted(observations, key=lambda o: o.start_time):
        children.setdefault(obs.parent_observation_id, []).append(obs)

    def show(parent_id: str | None, depth: int) -> None:
        for obs in children.get(parent_id, []):
            usage = obs.usage_details or {}
            tokens = (
                f"  tokens in/out {usage.get('input', 0)}/{usage.get('output', 0)}"
                if usage
                else ""
            )
            level = f"  [{obs.level}]" if obs.level not in (None, "DEFAULT") else ""
            print(
                f"{'  ' * depth}{obs.type:<10} {obs.name:<28} {obs.latency or 0:6.2f}s{tokens}{level}"
            )
            show(obs.id, depth + 1)

    print(f"trace {trace_id}")
    show(None, 0)


if __name__ == "__main__":
    main()
