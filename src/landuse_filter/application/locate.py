"""Where the texts of an input file are: ``(text sha, H3 cell)`` pairs for geographic planning."""

from landuse_filter.application.datasets import SPECS
from landuse_filter.application.geo import CellOf
from landuse_filter.application.locators import Fetch, PlanningLocator

__all__ = ["Fetch", "locator"]


def locator(dataset: str, fetch: Fetch, cell_of: CellOf) -> PlanningLocator:
    """``(dataset, path, local file) -> (sha, cell)`` for the datasets that have coordinates.

    ``fetch`` downloads sibling files of the same input repo (the wiki polygon tables).
    """
    factory = SPECS[dataset].planning_locator
    if factory is None:
        raise ValueError(f"{dataset}: no coordinates to order by")
    return factory(fetch, cell_of)
