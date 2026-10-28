
import math


def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.
    """
    methods = ["python", "cython", "numpy"]

    pipeline_totals = {method: 0.0 for method in methods}
    operation_results = []
    crossover_sizes = {}

    for op in operations:
        name = op["name"]
        times = {}

        # 1. Calculate runtime in microseconds for each method.
        for method in methods:
            overhead = op[f"{method}_overhead_ns"]
            per_elem = op[f"{method}_per_elem_ns"]

            times[method] = (
                overhead + n * per_elem
            ) / 1000.0

            # Keep unrounded values for pipeline totals.
            pipeline_totals[method] += times[method]

        # 2. Calculate speedups relative to Python.
        python_time = times["python"]

        cython_speedup = (
            python_time / times["cython"]
            if times["cython"] != 0
            else (float("inf") if python_time > 0 else 1.0)
        )

        numpy_speedup = (
            python_time / times["numpy"]
            if times["numpy"] != 0
            else (float("inf") if python_time > 0 else 1.0)
        )

        # 3. Select the fastest method.
        # Alphabetical tie-break: cython < numpy < python.
        best = min(methods, key=lambda method: (times[method], method))

        # 4. Calculate crossover size.
        cython_pe = op["cython_per_elem_ns"]
        numpy_pe = op["numpy_per_elem_ns"]
        cython_oh = op["cython_overhead_ns"]
        numpy_oh = op["numpy_overhead_ns"]

        if cython_pe <= numpy_pe:
            crossover = -1
        else:
            ratio = (numpy_oh - cython_oh) / (
                cython_pe - numpy_pe
            )

            crossover = 0 if ratio <= 0 else math.ceil(ratio)

        crossover_sizes[name] = crossover

        # 5. Store rounded per-operation results.
        operation_results.append({
            "name": name,
            "python_us": round(times["python"], 2),
            "cython_us": round(times["cython"], 2),
            "numpy_us": round(times["numpy"], 2),
            "cython_speedup": round(cython_speedup, 2),
            "numpy_speedup": round(numpy_speedup, 2),
            "best": best,
        })

    # 6. Select the pipeline recommendation using unrounded totals.
    recommended = min(
        methods,
        key=lambda method: (pipeline_totals[method], method)
    )

    # 7. Return the final result.
    return {
        "operations": operation_results,
        "pipeline_python_us": round(pipeline_totals["python"], 2),
        "pipeline_cython_us": round(pipeline_totals["cython"], 2),
        "pipeline_numpy_us": round(pipeline_totals["numpy"], 2),
        "recommended": recommended,
        "crossover_sizes": crossover_sizes,
    }
