"""Cross-check with CPS-02: the same synthetic navigation chain.

CPS-02 (``cpsrbd`` v1.0.0, GMCavalheri/cyber-physical-rbd-attack-trees) models the chain as a
unified RBD + attack-tree DAG. Its components and the attack-tree leaves whose *effects* this
project simulates must exist there under the names used here, and the integrity-monitor miss
probability it assumes is the number Phase 4 measures.
"""

import pytest

from fdicps.dual_anomaly_data_generation.fdi import (
    CPS02_BUS_INJECTION,
    CPS02_GNSS_SPOOF,
    CPS02_INTEGRITY_MISS,
)
from fdicps.dual_anomaly_data_generation.sensors import UNIT_COMPONENT

cpsrbd = pytest.importorskip("cpsrbd")

from cpsrbd.consequence_driven_analysis.case_study import load_parameters  # noqa: E402

ROWS = {row.name: row for row in load_parameters()}


def test_pinned_version() -> None:
    assert cpsrbd.__version__ == "1.0.0"


def test_navigation_units_exist_in_cps02() -> None:
    # CPS-02 names the hardware-failure events <component>_hw; air_data is this project's
    # labelled addition and must not be claimed to exist there.
    for component in set(UNIT_COMPONENT.values()) - {"air_data"}:
        assert ROWS[f"{component}_hw"].origin == "physical"
    assert "air_data_hw" not in ROWS


def test_simulated_attack_effects_are_cps02_leaves() -> None:
    assert ROWS[CPS02_BUS_INJECTION].origin == "cyber"
    assert ROWS[CPS02_GNSS_SPOOF].origin == "cyber"
    assert ROWS[CPS02_INTEGRITY_MISS].probability == pytest.approx(0.1)
