#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q alginate_ratio_transfer_IDE.py v62_core.py DP_spectrum_process_Class.py alginate_V66_ALL_NCE_OVERNIGHT_IDE.py experiments tests
for ver in 66 67 68 69 70; do
  echo "Running synthetic acceptance V${ver}"
  python "tests/test_v${ver}_synthetic.py"
done
