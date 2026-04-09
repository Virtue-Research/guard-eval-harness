"""Code vulnerability detection dataset adapters.

Submodules must be imported here so that
``@dataset_registry.register()`` decorators execute during
plugin discovery.  The ``import_submodules`` helper in
``plugins/discovery.py`` imports packages but does **not**
recurse into submodules automatically.
"""

from guard_eval_harness.datasets.code_vuln.local_cwe_json import (  # noqa: F401
    LocalCweJsonDataset,
)
from guard_eval_harness.datasets.code_vuln.vulnllm_r import (  # noqa: F401
    VulnLLMRApplication,
    VulnLLMRFunctionLevel,
    VulnLLMRRepoLevel,
)
