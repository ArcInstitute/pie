"""PIE: predicting transcriptional responses to perturbations from biological knowledge sources."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arc-pie")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"
