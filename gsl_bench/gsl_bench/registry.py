"""Agent lookup: by shipped name, by entry point, or by dotted path 'pkg.module:Class'."""
import importlib
from importlib.metadata import entry_points


def _agent_entry_points():
    eps = entry_points()
    # Python 3.10+ returns a SelectableGroups; 3.8/3.9 returns a plain dict.
    if hasattr(eps, 'select'):
        return list(eps.select(group='gsl_bench.agents'))
    return list(eps.get('gsl_bench.agents', []))


def available_agents():
    return sorted(ep.name for ep in _agent_entry_points())


def load_agent_class(spec: str):
    """Resolve an agent spec to a class.

    'gpulaika'                    -> shipped/registered entry point
    'mypkg.my_agent:MyAgent'      -> dotted path, no registration needed
    """
    if not spec:
        raise KeyError("No agent given. Use a registered name or 'pkg.module:ClassName'.")
    if ':' in spec:
        mod, cls = spec.split(':', 1)
        return getattr(importlib.import_module(mod), cls)
    for ep in _agent_entry_points():
        if ep.name == spec:
            return ep.load()
    raise KeyError(
        f"Unknown agent '{spec}'. Available: {available_agents()} "
        f"or use 'pkg.module:ClassName'.")
