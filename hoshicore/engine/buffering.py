"""Static queue compensation for reconverging, position-preserving streams."""

from typing import Any

from ..ops.base import BaseOp
from .build import ValidatedDag, _iter_node_src_links, _parse_link
from .flatten import INACTIVE_MARKER


def plan_input_buffers(
        dag: ValidatedDag, registry: dict[str, type[BaseOp]],
        effective_configs: dict[str, Any]) -> dict[tuple[str, str], int]:
    """Map (consumer, port) to capacity; never count worker-local storage.

    Global input feeders are independent. Configuration barriers and variable
    streams do not propagate finite, position-based lookahead.
    """
    lineage = {}
    capacities = {}
    for name in dag.exec_order:
        spec = dag.nodes[name]
        cls = registry[spec['op']]
        configs = {
            key: (effective_configs[key] if entry.get('global')
                  and key in effective_configs else entry['default'])
            for key, entry in cls.CONFIGS.items() if 'default' in entry or (
                entry.get('global') and key in effective_configs)
        }
        incoming = []
        for loc, src in _iter_node_src_links(spec):
            section, port = loc.split('.', 1)
            if src == INACTIVE_MARKER:
                continue
            parsed = _parse_link(src)
            if section == 'configs':
                configs.pop(port, None)
                if parsed[0] == 'configs' and parsed[1] in effective_configs:
                    configs[port] = effective_configs[parsed[1]]
                continue
            if (section != 'inputs' or parsed[0] != 'node'
                    or cls.INPUTS.get(port, {}).get('type') != 'sequence'):
                continue
            parent, output = parsed[1:]
            parent_cls = registry[dag.nodes[parent]['op']]
            if parent_cls.OUTPUTS.get(output, {}).get('type') != 'sequence':
                continue
            incoming.append((port, lineage[parent]))
            capacities[name, port] = cls.MAX_SIZE
        try:
            lookahead = cls.required_lookahead(configs)
        except KeyError as error:
            raise ValueError(
                f'{name}: lookahead requires static config {error}') from error
        if isinstance(lookahead,
                      bool) or not isinstance(lookahead, int) or lookahead < 0:
            raise ValueError(
                f'{name}: required_lookahead must be a nonnegative integer')
        offsets = {}
        if not cls.VARIABLE_OUTPUT:
            for _, ancestors in incoming:
                for ancestor, offset in ancestors.items():
                    offsets[ancestor] = max(offsets.get(ancestor, 0), offset)
            for port, ancestors in incoming:
                # Lookahead is an upper bound: excluded/edge windows can emit
                # earlier. Subtracting this branch's upper bound would therefore
                # underallocate when another branch actually needs its full span.
                gap = max(
                    (offset
                     for other_port, other in incoming if other_port != port
                     for ancestor, offset in other.items()
                     if ancestor in ancestors),
                    default=0)
                current = capacities[name, port]
                if current != 0:
                    capacities[name, port] = max(current, gap + 1)
        lineage[name] = {
            a: offset + lookahead
            for a, offset in offsets.items()
        }
        # Different output ports still share the producer's broadcast backpressure.
        lineage[name][name] = 0
    return capacities
