from factory_ai import config as C
from factory_ai.layout_optimizer import Problem, cost_manhattan, cost_path, is_feasible, optimize, violations
from factory_ai.process_flow import build_flows


def benchmark():
    layout = {
        "Cutting": (2, 12, 4, 3), "Drilling": (9, 12, 3, 3),
        "Assembly": (17, 12, 5, 3), "Inspection": (20, 5, 3, 2.5),
        "Packaging": (11, 5, 4, 3), "Storage": (2, 3, 4, 4),
    }
    fixed = {"Receiving": (0, 17, 1.5, 3), "Dispatch": (28.5, 1, 1.5, 3)}
    problem = Problem(30, 20, dict(C.NOMINAL_SIZE), build_flows(), fixed,
                      [(14, 9, 1.5, 1.5)], clearance=1.0)
    return problem, layout


def test_benchmark_layout_is_feasible_and_costs_are_positive():
    p, layout = benchmark()
    assert is_feasible(p, layout)
    assert not violations(p, layout)
    assert cost_manhattan(p, layout) > 0
    assert cost_path(p, layout) > 0


def test_short_optimizer_run_returns_feasible_layout():
    p, layout = benchmark()
    result = optimize(p, layout, n_seeds=1, iters=100, refine_iters=10, base_seed=7)
    assert result["final_feasible"]
    assert is_feasible(p, result["final_layout"])
    assert set(result["final_layout"]) == set(layout)
    assert result["final_cost_path"] > 0
