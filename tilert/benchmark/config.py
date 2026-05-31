"""TileRT configuration file loading.

Reads model weights paths from ~/.tilert/config.toml so that benchmark scripts
and regression workflows do not need hardcoded paths.

Config file format (~/.tilert/config.toml):

    [weights]
    deepseek_v3_2 = "/path/to/tilert_weights/DeepSeek-V32"
    deepseek_v3_2_v2 = "/path/to/tilert_weights/DeepSeek-V32-v2"
"""

import tomllib
from pathlib import Path

CONFIG_DIR = Path.home() / ".tilert"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def get_config_path() -> Path:
    """Return the path to the TileRT config file."""
    return CONFIG_FILE


def get_weights_dir(model: str, cli_override: str | None = None) -> str:
    """Resolve the weights directory for *model*.

    Resolution order (highest priority first):
      1. *cli_override* (from ``--model-weights-dir`` CLI flag)
      2. ``~/.tilert/config.toml`` → ``[weights].<model>``

    Raises ``FileNotFoundError`` / ``KeyError`` with a user-friendly message
    when the config file or key is missing.
    """
    if cli_override is not None:
        return cli_override

    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"No --model-weights-dir provided and config file not found at {config_path}.\n"
            f"Create it with:\n\n"
            f"  mkdir -p {CONFIG_DIR}\n"
            f"  cat > {config_path} << 'EOF'\n"
            f"  [weights]\n"
            f'  deepseek_v3_2 = "/path/to/DeepSeek-V32"\n'
            f"  EOF\n"
        )

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(
            f"Failed to parse {config_path}: {e}\n" f"Please check the file for syntax errors."
        ) from e

    weights = config.get("weights", {})
    if model not in weights:
        available = ", ".join(weights.keys()) if weights else "(none)"
        raise KeyError(
            f"Model {model!r} not found in {config_path} [weights] section.\n"
            f"Available models: {available}\n"
            f"Add it with:\n\n"
            f"  [weights]\n"
            f'  {model} = "/path/to/{model}/weights"\n'
        )

    return str(weights[model])
