"""any-company-llm — scan hardware, pick a GGUF model, serve it locally.

Zero cloud dependency after the initial model download.
"""

from any_company_llm.hardware_scanner import HardwareProfile, scan_hardware
from any_company_llm.model_selector import ModelSelection, select_model

__version__ = "0.1.0"

__all__ = [
    "HardwareProfile",
    "ModelSelection",
    "scan_hardware",
    "select_model",
    "__version__",
]
