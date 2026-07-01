from utils.registry import discover_modules

VISION_REGISTRY = discover_modules("vision")


def test_vision_creation(vision_input_args):
    for name, cls in VISION_REGISTRY.items():
        vision = cls(**vision_input_args)
        assert vision is not None
        assert name == vision.__class__.__name__
        assert isinstance(vision.tags, frozenset), f"Tags must be a frozenset for {name}"
