"""Engine registry. Add new adapters here only after the licence audit in CONTRIBUTING.md."""

from __future__ import annotations

from .base import Engine, EngineUnavailable
from .chatterbox_mtl import ChatterboxMultilingualEngine
from .selftest import SelftestEngine

_FACTORIES = {
    ChatterboxMultilingualEngine.name: ChatterboxMultilingualEngine,
    SelftestEngine.name: SelftestEngine,
}
_INSTANCES: dict[str, Engine] = {}


def engine_names() -> list[str]:
    return list(_FACTORIES)


def get_engine(name: str) -> Engine:
    if name not in _FACTORIES:
        raise ValueError(f"Motor desconhecido '{name}'. Opções: {', '.join(_FACTORIES)}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _FACTORIES[name]()
    return _INSTANCES[name]


__all__ = ["Engine", "EngineUnavailable", "engine_names", "get_engine"]
