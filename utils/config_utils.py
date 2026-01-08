# config_utils.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def dump_yaml(obj: Any, path: str) -> None:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _parse_value(val_str: str) -> Any:
    import yaml
    return yaml.safe_load(val_str)


def set_by_dotted_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_overrides(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Supported forms:
      --a.b 123
      --a.b=123
      a.b=123
    """
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t.startswith("--") and "=" in t[2:]:
            k, v = t[2:].split("=", 1)
            pairs.append((k.strip(), v.strip()))
            i += 1
            continue

        if t.startswith("--"):
            k = t[2:].strip()
            if i + 1 >= len(tokens):
                raise ValueError(f"Override '{t}' missing value")
            v = tokens[i + 1]
            if v.startswith("--"):
                raise ValueError(f"Override '{t}' missing value (next token looks like flag '{v}')")
            pairs.append((k, v))
            i += 2
            continue

        if "=" in t:
            k, v = t.split("=", 1)
            pairs.append((k.strip(), v.strip()))
            i += 1
            continue

        raise ValueError(f"Unrecognized override token: '{t}'")

    return pairs


def apply_overrides(cfg: Dict[str, Any], unknown_tokens: List[str]) -> Dict[str, Any]:
    if not unknown_tokens:
        return cfg
    pairs = parse_overrides(unknown_tokens)
    for key, val_str in pairs:
        set_by_dotted_path(cfg, key, _parse_value(val_str))
    return cfg
