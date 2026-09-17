"""Convert an axon workflow spec (goal + stages) into an ELIO `feature.yaml`.

    python examples/to_elio_feature.py [spec.yaml] > my.feature.yaml

With no argument it converts the built-in design->implement->test->commit pipeline.
The emitted YAML is an `elio/v1` Feature that ELIO's loader (`@elio/sdk`
loadFeaturePack) accepts and its runtime can back — so the SAME pipeline the axon
`run_workflow` tool executes can also be run by the TypeScript ELIO engine.

Validate the output against the real loader (from the elio repo):

    node -e "import('@elio/sdk').then(m=>console.log(!!m.loadFeaturePackFromFile(process.argv[1])))" my.feature.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

from axon.elio import to_elio_feature, to_yaml


def main() -> int:
    if len(sys.argv) > 1:
        spec = Path(sys.argv[1]).read_text()
        fid = None
    else:
        from examples.run_elio_workflow import DESIGN_IMPL_COMMIT
        spec = DESIGN_IMPL_COMMIT
        fid = "calc.add"

    pack = to_elio_feature(spec, feature_id=fid, artifact_kind="python-module", eval_gate="test_passes")
    sys.stdout.write(to_yaml(pack))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
