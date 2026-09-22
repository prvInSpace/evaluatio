from dataclasses import dataclass
import gc
import math
import statistics
import time
from typing import Callable, List


@dataclass
class BenchResult:
    name: str
    repeats: int
    loops: int
    times: List[float]
    total_calls: int

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times)

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.times)

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def worst(self) -> float:
        return max(self.times)

    @property
    def ops_per_sec(self) -> float:
        # operations measured as number of function calls
        total_calls = self.total_calls
        total_time = sum(self.times)
        return total_calls / total_time if total_time > 0 else math.inf

    def pretty(self) -> str:
        return (
            f"Function: {self.name}\n"
            f"  Repeats × Loops: {self.repeats} × {self.loops} (calls = {self.total_calls})\n"
            f"  Mean:   {self.mean:.6f} s\n"
            f"  Std:    {self.stdev:.6f} s\n"
            f"  Best:   {self.best:.6f} s\n"
            f"  Worst:  {self.worst:.6f} s\n"
            f"  Thruput: {self.ops_per_sec:,.6f} calls/s\n"
        )


def benchmark(
    func: Callable,
    repeats: int = 1000,
    loops: int = 1,
    warmup: int = 10,
    disable_gc: bool = True,
) -> BenchResult:
    """Benchmark a function.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    repeats : int
        Number of repeated timing measurements (outer loop).
    loops : int
        Number of function calls per measurement (inner loop), batched to reduce overhead.
    warmup : int
        Number of warm-up iterations not recorded.
    disable_gc : bool
        Temporarily disable GC during timing to reduce jitter.
    """
    print(f"Benchmarking {getattr(func, "__name__", str(func))}")
    times: List[float] = []
    total_calls = repeats * loops

    # Warm-up
    for _ in range(max(0, warmup)):
        for _ in range(loops):
            func()

    # Timing
    try:
        if disable_gc:
            gc_old = gc.isenabled()
            gc.disable()
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(loops):
                func()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    finally:
        if disable_gc and "gc_old" in locals() and gc_old:
            gc.enable()

    return BenchResult(
        name=getattr(func, "__name__", str(func)),
        repeats=repeats,
        loops=loops,
        times=times,
        total_calls=total_calls,
    )


def format_comparison(results: List[BenchResult]) -> str:
    winner = min(results, key=lambda r: r.mean)
    lines = [
        "\n=== Results ===",
        *[r.pretty() for r in results],
        f"{'-' * 16}",
        f"Winner (lower mean time): {winner.name}",
        *[
            f"Speedup (mean ratio): {r.mean / winner.mean if winner.mean > 0 else math.inf:,.2f}×  ( {winner.name} vs {r.name} )"
            for r in results
            if r is not winner
        ],
    ]
    return "\n".join(lines)
