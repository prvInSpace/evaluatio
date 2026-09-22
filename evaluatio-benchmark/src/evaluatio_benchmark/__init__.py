import argparse
import importlib
import inspect
from typing import Callable

from evaluatio_benchmark.benchmark import BenchResult, format_comparison


def main() -> None:

    parser = argparse.ArgumentParser("evaluatio_benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", action="store_true") # Might change to path

    args = parser.parse_args()

    print("Hello from evaluatio-benchmark!")
    print(args)
    results = run_benchmark_config(args.config)
    print(format_comparison(results))


def run_benchmark_config(config: str) -> list[BenchResult]:
    if not config.startswith("evaluatio_benchmark"):
        config = "evaluatio_benchmark." + config     
    function = _load_function(config)
    return function()


def _load_function(config: str) -> Callable[[], list[BenchResult]]:
    *module_name, func_name = config.split(".")
    module = importlib.import_module(".".join(module_name))
    function = getattr(module, func_name)
    assert callable(function)
    signature = inspect.signature(function)
    assert str(signature.return_annotation) == "typing.List[evaluatio_benchmark.benchmark.BenchResult]"
    assert len(signature.parameters) == 0
    return function # type: ignore
