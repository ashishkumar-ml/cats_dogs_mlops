#!/usr/bin/env bash
# smoke_test.sh — post-deploy smoke test
# Usage: bash smoke_test.sh [BASE_URL]
# Exit code 0 = all tests passed, non-zero = failure

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PASS=0; FAIL=0

ok()   { echo "[PASS] $1"; ((PASS++)); }
fail() { echo "[FAIL] $1"; ((FAIL++)); }

echo "================================================"
echo "  Smoke Tests  ->  $BASE_URL"
echo "================================================"

# ── 1. Health check ──────────────────────────────────────────────────────────
echo ""
echo "1. GET /health"
HEALTH=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo "{}")
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
MODEL_LOADED=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_loaded',''))" 2>/dev/null || echo "")

[ "$STATUS" = "ok" ]       && ok "Health status = ok"         || fail "Health status not ok (got: $STATUS)"
[ "$MODEL_LOADED" = "True" ] && ok "Model loaded = True"       || fail "Model not loaded (got: $MODEL_LOADED)"

# ── 2. Prediction ─────────────────────────────────────────────────────────────
echo ""
echo "2. POST /predict"

# Generate a small synthetic JPEG
TMPIMG=$(mktemp /tmp/smoke_XXXXXX.jpg)
python3 - <<PYEOF
from PIL import Image
import numpy as np, sys
arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
Image.fromarray(arr, 'RGB').save("$TMPIMG")
PYEOF

PRED=$(curl -sf -X POST "$BASE_URL/predict" \
  -F "file=@${TMPIMG};type=image/jpeg" 2>/dev/null || echo "{}")
rm -f "$TMPIMG"

PRED_CLASS=$(echo "$PRED" | python3 -c "import sys,json; print(json.load(sys.stdin).get('predicted_class',''))" 2>/dev/null || echo "")
CONFIDENCE=$(echo "$PRED" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('confidence',0))" 2>/dev/null || echo "0")
PROBS=$(echo "$PRED" | python3 -c "import sys,json; print(json.load(sys.stdin).get('probabilities',{}))" 2>/dev/null || echo "")

[ "$PRED_CLASS" = "Cat" ] || [ "$PRED_CLASS" = "Dog" ] \
  && ok "Predicted class = $PRED_CLASS" \
  || fail "Unexpected predicted_class: '$PRED_CLASS'"

python3 -c "exit(0 if float('$CONFIDENCE') > 0 else 1)" 2>/dev/null \
  && ok "Confidence = $CONFIDENCE (>0)" \
  || fail "Confidence not positive"

[ -n "$PROBS" ] && ok "Probabilities dict present" || fail "Probabilities missing"

# ── 3. Metrics endpoint ──────────────────────────────────────────────────────
echo ""
echo "3. GET /metrics"
METRICS=$(curl -sf "$BASE_URL/metrics" 2>/dev/null || echo "")
echo "$METRICS" | grep -q "inference_requests_total" \
  && ok "Prometheus counter inference_requests_total present" \
  || fail "Prometheus metrics not exposed"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "================================================"

[ "$FAIL" -eq 0 ] || exit 1
