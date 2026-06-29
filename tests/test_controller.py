from src.utils.registry import discover_modules

CONTROLLER_REGISTRY = discover_modules("controller")


def test_controller_creation(controller_input_args):
    for name, cls in CONTROLLER_REGISTRY.items():
        controller = cls(**controller_input_args)
        assert controller is not None
        assert name == controller.__class__.__name__
        assert isinstance(controller.tags, frozenset), f"Tags must be a frozenset for {name}"
