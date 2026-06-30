from __future__ import annotations

import numpy as np

from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance


def make_toy_instance() -> CrossDockInstance:
    flow = np.array(
        [
            [[10.0], [5.0], [2.0], [0.0]],
            [[3.0], [4.0], [0.0], [8.0]],
        ]
    )
    travel_time = np.array(
        [
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 5.0],
            [20.0, 5.0, 0.0],
        ]
    )
    trucks = ["C1", "C2", "O1", "O2"]
    return CrossDockInstance(
        compound_trucks=["C1", "C2"],
        outbound_trucks=["O1", "O2"],
        destinations=["D1", "D2", "D3", "D4"],
        doors=[1, 2, 3],
        product_types=["K1"],
        flow=flow,
        product_time=np.array([1.0]),
        travel_time=travel_time,
        enter_time={truck: 1.0 for truck in trucks},
        leave_time={truck: 1.0 for truck in trucks},
    )


def make_toy_solution() -> Solution:
    return Solution(
        compound_assignment={
            "C1": ("D1", 1),
            "C2": ("D2", 2),
        },
        outbound_assignment={
            "O1": ("D3", 3),
            "O2": ("D4", 3),
        },
        door_sequences={
            1: [],
            2: [],
            3: ["O1", "O2"],
        },
    )

