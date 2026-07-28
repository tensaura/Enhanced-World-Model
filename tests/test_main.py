import pytest


def test_main_fast(monkeypatch, main_input_args):
    """Test if main runs in multiple configurations.
    TODO: Bypass PPO completely for this test to run faster."""

    for correct_args in main_input_args["correct"]:
        monkeypatch.setattr("sys.argv", correct_args + ["--epochs", "1", "--ppo-epochs", "1", "--rollout-steps", "1"])
        from main import main

        main()

    for incorrect_args in main_input_args["error"]:
        monkeypatch.setattr("sys.argv", incorrect_args + ["--epochs", "1", "--ppo-epochs", "1", "--rollout-steps", "1"])
        from main import main

        with pytest.raises(RuntimeError):
            main()


@pytest.mark.slow
def test_main(monkeypatch, main_input_args):
    """Test if main runs in multiple configurations for longer. Needed to know if training is working across epochs."""
    for correct_args in main_input_args["correct"]:
        monkeypatch.setattr("sys.argv", correct_args + ["--epochs", "5"])
        from main import main

        main()

    for incorrect_args in main_input_args["error"]:
        monkeypatch.setattr("sys.argv", incorrect_args + ["--epochs", "5"])
        from main import main

        with pytest.raises(RuntimeError):
            main()
