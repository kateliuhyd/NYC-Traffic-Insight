# Lazy re-export to avoid heavy sklearn import at package init time.
# Usage: from src.models import SegmentedModel


def __getattr__(name):
    if name == "SegmentedModel":
        from src.models.segmented_model import SegmentedModel
        return SegmentedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
