from src.NLSeeker.config import NLSeekerConfig

_LAZY_NAMES = {
    "NLSeeker": ("src.Operators.Seekers.NLSeeker", "NLSeeker"),
    "NLSeekerStandalone": ("src.NLSeeker.standalone", "NLSeekerStandalone"),
    "nl_search": ("src.NLSeeker.standalone", "nl_search"),
}


def __getattr__(name):
    if name in _LAZY_NAMES:
        import importlib
        module_name, attr = _LAZY_NAMES[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'src.NLSeeker' has no attribute {name!r}")


def __dir__():
    return sorted({"NLSeekerConfig", *_LAZY_NAMES.keys()})


__all__ = ["NLSeekerConfig", "NLSeeker", "NLSeekerStandalone", "nl_search"]
