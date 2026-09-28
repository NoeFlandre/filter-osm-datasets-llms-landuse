import asyncio

from landuse_filter.application.calibrate import Point, best, sweep


class Engine:
    """Throughput saturates at 32 in-flight requests."""

    def __init__(self):
        self.inflight = 0

    async def generate(self, ids):
        self.inflight += 1
        await asyncio.sleep(0.001 * max(1, self.inflight / 32))
        self.inflight -= 1
        return {"meta_info": {"completion_tokens": 10}}


def test_sweep_measures_every_window_on_all_prompts():
    points = sweep(Engine(), [[i] for i in range(128)], [1, 8, 32])
    assert [p.window for p in points] == [1, 8, 32]
    assert all(p.sentences == 128 and p.generated_tokens == 1280 for p in points)
    assert points[-1].sentences_per_second > points[0].sentences_per_second


def test_best_prefers_the_smallest_window_near_the_top():
    points = [Point(16, 100, 10, 0), Point(64, 197, 10, 0), Point(128, 200, 10, 0)]
    assert best(points).window == 64
    assert Point(1, 0, 0, 0).sentences_per_second == 0.0


class LoopBoundEngine(Engine):
    """Like SGLang: only works in the event loop it was first used in."""

    def __init__(self):
        super().__init__()
        self.loop = None

    async def generate(self, ids):
        loop = asyncio.get_running_loop()
        self.loop = self.loop or loop
        if loop is not self.loop:
            raise RuntimeError("engine bound to another event loop")
        return await super().generate(ids)


def test_sweep_runs_every_level_in_one_event_loop_and_reports_each():
    seen = []
    points = sweep(LoopBoundEngine(), [[i] for i in range(64)], [4, 8, 16], on_point=seen.append)
    assert [p.window for p in points] == [4, 8, 16] == [p.window for p in seen]
