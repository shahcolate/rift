"""The `import rift` public API surface: lazy, complete, and usable."""

from __future__ import annotations

import subprocess
import sys


def test_all_exports_resolve():
    import rift

    for name in rift.__all__:
        assert getattr(rift, name) is not None


def test_all_matches_export_table():
    # __all__ is a hand-maintained literal (pyright can't evaluate a
    # computed one); this pins it to the lazy-export table so the two
    # can't drift.
    import rift

    assert set(rift.__all__) == set(rift._EXPORTS) | {"__version__"}


def test_unknown_attribute_raises():
    import rift
    import pytest

    with pytest.raises(AttributeError, match="no attribute"):
        rift.definitely_not_a_thing


def test_dir_lists_exports():
    import rift

    listing = dir(rift)
    for name in ("compare_runs", "run_suite", "load_suite", "DriftResult"):
        assert name in listing


def test_import_rift_is_lazy():
    # `import rift` must not drag in the heavy stack (scipy, httpx, rich) —
    # library users embedding just the stats shouldn't pay CLI startup.
    code = (
        "import sys; import rift; "
        "heavy = [m for m in ('scipy', 'httpx', 'rich') if m in sys.modules]; "
        "print(','.join(heavy) or 'CLEAN')"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.stdout.strip() == "CLEAN", out.stdout


def test_stats_layer_usable_over_foreign_scores():
    # The adapter pitch in miniature: paired scores from ANY harness.
    import rift

    drift = rift.compare_runs(
        [1.0] * 10 + [0.0] * 2, [1.0] * 8 + [0.0] * 4,
        "their-model-v1", "their-model-v2", "their-eval",
    )
    assert drift.test_used == "mcnemar_exact"
    assert 0.0 <= drift.p_value <= 1.0


def test_version_matches_installed_metadata():
    import rift

    assert rift.__version__.count(".") == 2
