"""Config loading and merging utilities."""
from pathlib import Path

try:
    from omegaconf import OmegaConf
    OMEGACONF = True
except ImportError:
    OMEGACONF = False


def load_config(path: str):
    """Load a YAML config file. Falls back to SimpleNamespace if OmegaConf not available."""
    if OMEGACONF:
        cfg = OmegaConf.load(path)
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
        return dict_to_ns(raw)


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
