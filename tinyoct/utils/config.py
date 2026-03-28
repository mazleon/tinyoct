"""Config loading and merging utilities."""
from pathlib import Path

try:
    from omegaconf import OmegaConf
    OMEGACONF = True
except ImportError:
    OMEGACONF = False


def load_config(path: str):
    """Load a YAML config, resolving 'defaults' inheritance like Hydra."""
    config_dir = Path(path).parent

    if OMEGACONF:
        raw = OmegaConf.load(path)
        # Resolve 'defaults' list: load each base and merge in order
        defaults = OmegaConf.to_container(raw.get("defaults", []), resolve=False)
        if defaults:
            base_cfg = OmegaConf.create({})
            for entry in defaults:
                base_name = entry if isinstance(entry, str) else list(entry.values())[0]
                base_path = config_dir / f"{base_name}.yaml"
                if base_path.exists():
                    base_cfg = OmegaConf.merge(base_cfg, OmegaConf.load(base_path))
            # Merge override config on top (excluding 'defaults' key)
            override = OmegaConf.create({k: v for k, v in OmegaConf.to_container(raw, resolve=False).items() if k != "defaults"})
            cfg = OmegaConf.merge(base_cfg, override)
        else:
            cfg = raw
        return cfg
    else:
        import yaml
        from types import SimpleNamespace

        def dict_to_ns(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_ns(i) for i in d]
            return d

        with open(path) as f:
            raw = yaml.safe_load(f)
        # Handle defaults inheritance
        defaults = raw.pop("defaults", [])
        merged = {}
        for entry in defaults:
            base_name = entry if isinstance(entry, str) else list(entry.values())[0]
            base_path = config_dir / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as f:
                    base = yaml.safe_load(f) or {}
                merged.update(base)
        merged.update(raw)
        return dict_to_ns(merged)


def merge_ablation(base_cfg, ablation_overrides: dict):
    """Apply ablation overrides to base config."""
    if OMEGACONF:
        override_list = [f"{k}={v}" for k, v in ablation_overrides.items()
                         if k != "description"]
        return OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(override_list))
    else:
        # Simple dot-path setter for SimpleNamespace
        import copy
        cfg = copy.deepcopy(base_cfg)
        for dotpath, value in ablation_overrides.items():
            if dotpath == "description":
                continue
            parts = dotpath.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        return cfg
