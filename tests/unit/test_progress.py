import pytest

from landuse_filter.domain.progress import FAILED_RATE_ALERT, alerts, eta


def test_eta_uses_the_aggregate_live_rate():
    assert eta(7200, [1.0, 1.0]).hours == 1.0
    assert eta(0, []).hours == 0.0
    assert eta(10, []).hours is None  # nothing running: no estimate


@pytest.mark.parametrize(("completed", "failed", "n"), [(0, 0, 0), (100, 5, 0), (100, 6, 1)])
def test_failed_rate_alert(completed, failed, n):
    assert len(alerts(completed, failed)) == n
    assert FAILED_RATE_ALERT == 0.05
