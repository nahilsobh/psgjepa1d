"""PSG-JEPA-1D. Torch-dependent symbols are imported lazily so that `world` and `data`
(pure NumPy) can be used without torch installed."""
__all__ = ["JEPA1D","training_step","REG","LAM0","PSGGroundingHeads","grounding_loss"]
def __getattr__(name):
    if name in ("JEPA1D","training_step","REG","LAM0"):
        from . import model as _m; return getattr(_m, name)
    if name in ("PSGGroundingHeads","grounding_loss"):
        from . import grounding as _g; return getattr(_g, name)
    raise AttributeError(name)
