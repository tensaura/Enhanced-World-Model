from utils.registry import discover_modules

MEMORY_REGISTRY = discover_modules("memory")


def test_memory_creation(memory_input_args):
    for name, cls in MEMORY_REGISTRY.items():
        memory = cls(**memory_input_args)
        assert memory is not None
        assert name == memory.__class__.__name__
        assert isinstance(memory.tags, frozenset), f"Tags must be a frozenset for {name}"
