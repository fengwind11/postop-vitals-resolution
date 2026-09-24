from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling_common import fit_tabular_preprocessor, apply_tabular_preprocessor, metric_values, set_seed

set_seed(20260918)
rng = np.random.default_rng(20260918)
x = rng.normal(size=(80, 6))
x[rng.random(x.shape) < 0.1] = np.nan
y = np.array([0] * 64 + [1] * 16)
p = np.clip(0.08 + 0.6 * (y + rng.normal(0, 0.5, len(y))) / 2, 1e-4, 1 - 1e-4)
prep = fit_tabular_preprocessor(x, scale=True)
xt = apply_tabular_preprocessor(x, prep)
metrics = metric_values(y, p)
assert xt.shape == (80, 6), xt.shape
assert all(np.isfinite(list(metrics.values()))), metrics
assert 0 <= metrics["auroc"] <= 1
print("SYNTHETIC_SMOKE_TEST_PASS")
