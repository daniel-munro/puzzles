from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

Color = str
Cell = Tuple[int, int]
Pattern = Tuple[Color, Color, Color, Color, Color]

POS_CENTER = 0
POS_NORTH = 1
POS_EAST = 2
POS_SOUTH = 3
POS_WEST = 4


@dataclass(frozen=True)
class Puzzle:
    size: int
    key: List[Pattern]
    givens: Dict[Cell, Color]

    @property
    def colors(self) -> Set[Color]:
        colors: Set[Color] = set()
        for pattern in self.key:
            colors.update(pattern)
        return colors

    @property
    def cells(self) -> List[Cell]:
        cells: List[Cell] = []
        r = self.size
        for y in range(-r, r + 1):
            for x in range(-r, r + 1):
                if abs(x) + abs(y) <= r:
                    cells.append((x, y))
        return cells

    def pluses(self) -> List[Tuple[Cell, Cell, Cell, Cell, Cell]]:
        cell_set = set(self.cells)
        pluses: List[Tuple[Cell, Cell, Cell, Cell, Cell]] = []
        for x, y in cell_set:
            center = (x, y)
            north = (x, y - 1)
            east = (x + 1, y)
            south = (x, y + 1)
            west = (x - 1, y)
            if (
                north in cell_set
                and east in cell_set
                and south in cell_set
                and west in cell_set
            ):
                pluses.append((center, north, east, south, west))
        return pluses


class Solver:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.cells = puzzle.cells
        self.patterns = self._expand_key(puzzle.key)
        self.colors = sorted({c for pattern in self.patterns for c in pattern})
        self.pluses = puzzle.pluses()
        self.cell_to_pluses = self._build_cell_to_pluses()

    def _expand_key(self, key: Iterable[Pattern]) -> List[Pattern]:
        expanded: Set[Pattern] = set()
        for pattern in key:
            variants = self._pattern_variants(pattern)
            expanded.update(variants)
        return sorted(expanded)

    def _pattern_variants(self, pattern: Pattern) -> Set[Pattern]:
        variants: Set[Pattern] = set()
        rotated = pattern
        for _ in range(4):
            variants.add(rotated)
            variants.add(self._flip_horizontal(rotated))
            rotated = self._rotate(rotated)
        return variants

    def _rotate(self, pattern: Pattern) -> Pattern:
        c, n, e, s, w = pattern
        return (c, w, n, e, s)

    def _flip_horizontal(self, pattern: Pattern) -> Pattern:
        c, n, e, s, w = pattern
        return (c, n, w, s, e)

    def _build_cell_to_pluses(self) -> Dict[Cell, List[Tuple[int, int]]]:
        cell_to_pluses: Dict[Cell, List[Tuple[int, int]]] = {c: [] for c in self.cells}
        for idx, plus in enumerate(self.pluses):
            for pos, cell in enumerate(plus):
                cell_to_pluses[cell].append((idx, pos))
        return cell_to_pluses

    def solve(self) -> Optional[Dict[Cell, Color]]:
        assignment: Dict[Cell, Color] = {}
        for cell, color in self.puzzle.givens.items():
            if cell not in self.cell_to_pluses:
                return None
            assignment[cell] = color
        for cell in assignment:
            if not self._pluses_consistent_for_cell(assignment, cell):
                return None
        return self._search(assignment)

    def solve_all(self) -> Tuple[int, Optional[Dict[Cell, Color]]]:
        assignment: Dict[Cell, Color] = {}
        for cell, color in self.puzzle.givens.items():
            if cell not in self.cell_to_pluses:
                return 0, None
            assignment[cell] = color
        for cell in assignment:
            if not self._pluses_consistent_for_cell(assignment, cell):
                return 0, None
        count, first = self._search_all(assignment, first_solution=None)
        return count, first

    def solve_all_solutions(
        self, max_solutions: Optional[int] = None
    ) -> Tuple[List[Dict[Cell, Color]], bool]:
        assignment: Dict[Cell, Color] = {}
        for cell, color in self.puzzle.givens.items():
            if cell not in self.cell_to_pluses:
                return [], True
            assignment[cell] = color
        for cell in assignment:
            if not self._pluses_consistent_for_cell(assignment, cell):
                return [], True
        solutions: List[Dict[Cell, Color]] = []
        complete = self._collect_solutions(assignment, solutions, max_solutions)
        return solutions, complete

    def _search_all(
        self,
        assignment: Dict[Cell, Color],
        first_solution: Optional[Dict[Cell, Color]],
    ) -> Tuple[int, Optional[Dict[Cell, Color]]]:
        if len(assignment) == len(self.cells):
            solution = dict(assignment)
            return 1, solution if first_solution is None else first_solution

        next_cell = self._select_unassigned_cell(assignment)
        if next_cell is None:
            return 0, first_solution

        allowed = self._allowed_colors_for_cell(assignment, next_cell)
        if not allowed:
            return 0, first_solution

        count = 0
        for color in allowed:
            assignment[next_cell] = color
            if self._pluses_consistent_for_cell(assignment, next_cell):
                sub_count, first_solution = self._search_all(assignment, first_solution)
                count += sub_count
            assignment.pop(next_cell)
        return count, first_solution

    def _collect_solutions(
        self,
        assignment: Dict[Cell, Color],
        solutions: List[Dict[Cell, Color]],
        max_solutions: Optional[int],
    ) -> bool:
        if max_solutions is not None and len(solutions) >= max_solutions:
            return False
        if len(assignment) == len(self.cells):
            solutions.append(dict(assignment))
            if max_solutions is not None and len(solutions) >= max_solutions:
                return False
            return True

        next_cell = self._select_unassigned_cell(assignment)
        if next_cell is None:
            return True

        allowed = self._allowed_colors_for_cell(assignment, next_cell)
        if not allowed:
            return True

        for color in allowed:
            assignment[next_cell] = color
            if self._pluses_consistent_for_cell(assignment, next_cell):
                if not self._collect_solutions(assignment, solutions, max_solutions):
                    assignment.pop(next_cell)
                    return False
            assignment.pop(next_cell)
        return True

    def _search(self, assignment: Dict[Cell, Color]) -> Optional[Dict[Cell, Color]]:
        if len(assignment) == len(self.cells):
            return dict(assignment)

        next_cell = self._select_unassigned_cell(assignment)
        if next_cell is None:
            return None

        allowed = self._allowed_colors_for_cell(assignment, next_cell)
        if not allowed:
            return None

        for color in allowed:
            assignment[next_cell] = color
            if self._pluses_consistent_for_cell(assignment, next_cell):
                result = self._search(assignment)
                if result is not None:
                    return result
            assignment.pop(next_cell)
        return None

    def _select_unassigned_cell(self, assignment: Dict[Cell, Color]) -> Optional[Cell]:
        best_cell: Optional[Cell] = None
        best_size = 1_000_000
        for cell in self.cells:
            if cell in assignment:
                continue
            allowed = self._allowed_colors_for_cell(assignment, cell)
            if not allowed:
                return cell
            if len(allowed) < best_size:
                best_size = len(allowed)
                best_cell = cell
                if best_size == 1:
                    break
        return best_cell

    def _allowed_colors_for_cell(
        self, assignment: Dict[Cell, Color], cell: Cell
    ) -> List[Color]:
        if cell in assignment:
            return [assignment[cell]]
        allowed: Set[Color] = set(self.colors)
        for plus_idx, pos in self.cell_to_pluses[cell]:
            plus = self.pluses[plus_idx]
            possible = self._possible_colors_for_position(assignment, plus, pos)
            if not possible:
                return []
            allowed &= possible
            if not allowed:
                return []
        return sorted(allowed)

    def _possible_colors_for_position(
        self,
        assignment: Dict[Cell, Color],
        plus: Tuple[Cell, Cell, Cell, Cell, Cell],
        pos: int,
    ) -> Set[Color]:
        possible: Set[Color] = set()
        for pattern in self.patterns:
            if self._pattern_matches(assignment, plus, pattern):
                possible.add(pattern[pos])
        return possible

    def _pattern_matches(
        self,
        assignment: Dict[Cell, Color],
        plus: Tuple[Cell, Cell, Cell, Cell, Cell],
        pattern: Pattern,
    ) -> bool:
        for pos, cell in enumerate(plus):
            if cell in assignment and assignment[cell] != pattern[pos]:
                return False
        return True

    def _pluses_consistent_for_cell(
        self, assignment: Dict[Cell, Color], cell: Cell
    ) -> bool:
        for plus_idx, _ in self.cell_to_pluses[cell]:
            plus = self.pluses[plus_idx]
            if not self._plus_has_matching_pattern(assignment, plus):
                return False
        return True

    def _plus_has_matching_pattern(
        self, assignment: Dict[Cell, Color], plus: Tuple[Cell, Cell, Cell, Cell, Cell]
    ) -> bool:
        for pattern in self.patterns:
            if self._pattern_matches(assignment, plus, pattern):
                return True
        return False


class Generator:
    def __init__(
        self,
        size: int,
        givens: Dict[Cell, Color],
        colors: Iterable[Color],
        key_size: int,
        disallowed_patterns: Optional[Iterable[Pattern]] = None,
        candidate_patterns: Optional[Iterable[Pattern]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.size = size
        self.givens = dict(givens)
        self.colors = list(colors)
        self.key_size = key_size
        self.rng = rng or random.Random()
        if candidate_patterns is None:
            candidate_patterns = self._all_patterns(self.colors)
        self.disallowed = self._canonical_set(disallowed_patterns or [])
        self.candidates = self._unique_patterns(candidate_patterns)
        if self.disallowed:
            self.candidates = [
                pattern
                for pattern in self.candidates
                if self._canonical_pattern(pattern) not in self.disallowed
            ]

    def find_unique_key(self, max_attempts: int = 1_000) -> Optional[List[Pattern]]:
        if self.key_size > len(self.candidates):
            return None
        for _ in range(max_attempts):
            key = self.rng.sample(self.candidates, self.key_size)
            puzzle = Puzzle(size=self.size, key=key, givens=self.givens)
            solver = Solver(puzzle)
            solutions, complete = solver.solve_all_solutions(max_solutions=2)
            if complete and len(solutions) == 1 and self._uses_all_colors(solutions[0]):
                return key
        return None

    def _all_patterns(self, colors: Iterable[Color]) -> List[Pattern]:
        colors_list = list(colors)
        patterns: List[Pattern] = []
        for c in colors_list:
            for n in colors_list:
                for e in colors_list:
                    for s in colors_list:
                        for w in colors_list:
                            patterns.append((c, n, e, s, w))
        return patterns

    def _unique_patterns(self, patterns: Iterable[Pattern]) -> List[Pattern]:
        seen: Dict[Pattern, Pattern] = {}
        for pattern in patterns:
            canonical = self._canonical_pattern(pattern)
            if canonical not in seen:
                seen[canonical] = pattern
        return sorted(seen.values())

    def _canonical_pattern(self, pattern: Pattern) -> Pattern:
        variants = self._pattern_variants(pattern)
        return min(variants)

    def _canonical_set(self, patterns: Iterable[Pattern]) -> Set[Pattern]:
        return {self._canonical_pattern(pattern) for pattern in patterns}

    def _uses_all_colors(self, solution: Dict[Cell, Color]) -> bool:
        present = set(solution.values())
        return all(color in present for color in self.colors)

    def _pattern_variants(self, pattern: Pattern) -> Set[Pattern]:
        variants: Set[Pattern] = set()
        rotated = pattern
        for _ in range(4):
            variants.add(rotated)
            variants.add(self._flip_horizontal(rotated))
            rotated = self._rotate(rotated)
        return variants

    def _rotate(self, pattern: Pattern) -> Pattern:
        c, n, e, s, w = pattern
        return (c, w, n, e, s)

    def _flip_horizontal(self, pattern: Pattern) -> Pattern:
        c, n, e, s, w = pattern
        return (c, n, w, s, e)


def _draw_solution_on_axes(
    ax,
    size: int,
    assignment: Dict[Cell, Color],
    color_map: Dict[Color, str],
) -> None:
    from matplotlib.patches import Rectangle

    r = size
    for y in range(-r, r + 1):
        for x in range(-r, r + 1):
            if abs(x) + abs(y) <= r:
                cell = (x, y)
                color = assignment.get(cell, None)
                face = color_map.get(color, "white")
                rect = Rectangle((x, -y), 1, 1, facecolor=face, edgecolor="black")
                ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.set_xlim(-r - 1, r + 2)
    ax.set_ylim(-r - 1, r + 2)
    ax.axis("off")


def _draw_pattern_tile(
    ax,
    pattern: Pattern,
    origin_x: float,
    origin_y: float,
    color_map: Dict[Color, str],
) -> None:
    from matplotlib.patches import Rectangle

    positions = {
        POS_CENTER: (1, 1),
        POS_NORTH: (1, 0),
        POS_EAST: (2, 1),
        POS_SOUTH: (1, 2),
        POS_WEST: (0, 1),
    }
    for pos, (dx, dy) in positions.items():
        color = color_map.get(pattern[pos], "white")
        rect = Rectangle(
            (origin_x + dx, origin_y + dy),
            1,
            1,
            facecolor=color,
            edgecolor="black",
        )
        ax.add_patch(rect)


def _draw_key_patterns_on_axes(
    ax,
    key: List[Pattern],
    color_map: Dict[Color, str],
    columns: int = 4,
) -> None:
    from itertools import groupby

    sorted_key = sorted(key, key=lambda pattern: pattern[POS_CENTER])
    grouped = [
        (center, list(patterns))
        for center, patterns in groupby(sorted_key, key=lambda p: p[POS_CENTER])
    ]

    tile = 3
    pad = 1
    label_height = 1.2
    columns = max(1, columns)

    total_height = 0.0
    max_width = 0.0
    layout = []
    for center, patterns in grouped:
        rows = math.ceil(len(patterns) / columns) if patterns else 0
        group_height = label_height + rows * (tile + pad)
        total_height += group_height + pad
        max_width = max(max_width, columns * (tile + pad) - pad)
        layout.append((center, patterns, rows))

    current_y = 0.0
    for center, patterns, rows in layout:
        ax.text(
            0,
            current_y + label_height * 0.7,
            f"Center {center}",
            ha="left",
            va="center",
            fontsize=10,
        )
        current_y += label_height
        for idx, pattern in enumerate(patterns):
            col = idx % columns
            row = idx // columns
            x = col * (tile + pad)
            y = current_y + row * (tile + pad)
            _draw_pattern_tile(ax, pattern, x, y, color_map)
        current_y += rows * (tile + pad) + pad

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, max_width + 0.5)
    ax.set_ylim(total_height, -0.5)
    ax.axis("off")


def display_puzzle(
    puzzle: Puzzle,
    color_map: Optional[Dict[Color, str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib is required for graphical display.")
        return

    if color_map is None:
        color_map = {}

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax_puzzle = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[0, 1])

    _draw_solution_on_axes(ax_puzzle, puzzle.size, puzzle.givens, color_map)
    ax_puzzle.set_title("Puzzle")
    _draw_key_patterns_on_axes(ax_key, puzzle.key, color_map)
    ax_key.set_title("Key Patterns")

    plt.tight_layout()
    plt.show()


def display_solution(
    size: int,
    assignment: Dict[Cell, Color],
    color_map: Optional[Dict[Color, str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib is required for graphical display.")
        return

    if color_map is None:
        color_map = {}

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_solution_on_axes(ax, size, assignment, color_map)
    plt.tight_layout()
    plt.show()


def display_solutions(
    size: int,
    solutions: List[Dict[Cell, Color]],
    color_map: Optional[Dict[Color, str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib is required for graphical display.")
        return

    if not solutions:
        print("No solutions to display.")
        return

    if color_map is None:
        color_map = {}

    fig, ax = plt.subplots(figsize=(6, 6))
    state = {"idx": 0}

    def draw_solution(index: int) -> None:
        ax.clear()
        _draw_solution_on_axes(ax, size, solutions[index], color_map)
        ax.set_title(f"Solution {index + 1} / {len(solutions)}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("right", "n", " "):
            state["idx"] = (state["idx"] + 1) % len(solutions)
            draw_solution(state["idx"])
        elif event.key in ("left", "p", "backspace"):
            state["idx"] = (state["idx"] - 1) % len(solutions)
            draw_solution(state["idx"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_solution(state["idx"])
    plt.tight_layout()
    plt.show()


def display_solution_grid(
    size: int,
    solutions: List[Dict[Cell, Color]],
    per_page: int = 6,
    cols: int = 3,
    color_map: Optional[Dict[Color, str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib is required for graphical display.")
        return

    if not solutions:
        print("No solutions to display.")
        return

    if color_map is None:
        color_map = {}

    per_page = max(1, per_page)
    cols = max(1, cols)
    rows = max(1, math.ceil(per_page / cols))
    total_pages = math.ceil(len(solutions) / per_page)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if isinstance(axes, list):
        axes_list = axes
    else:
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    state = {"page": 0}

    def draw_page(page: int) -> None:
        start = page * per_page
        end = min(len(solutions), start + per_page)
        for idx, ax in enumerate(axes_list):
            ax.clear()
            sol_idx = start + idx
            if sol_idx < end:
                _draw_solution_on_axes(ax, size, solutions[sol_idx], color_map)
                ax.set_title(f"{sol_idx + 1}")
            else:
                ax.axis("off")
        fig.suptitle(f"Solutions {start + 1}-{end} / {len(solutions)}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("right", "n", " "):
            state["page"] = (state["page"] + 1) % total_pages
            draw_page(state["page"])
        elif event.key in ("left", "p", "backspace"):
            state["page"] = (state["page"] - 1) % total_pages
            draw_page(state["page"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_page(state["page"])
    plt.tight_layout()
    plt.show()


def solve() -> None:
    # ## Original:
    # key = [
    #     ("O", "G", "G", "B", "G"),
    #     ("O", "G", "G", "O", "O"),
    #     ("O", "G", "O", "B", "O"),
    #     ("G", "O", "R", "B", "B"),
    #     ("G", "O", "G", "G", "O"),
    #     ("G", "O", "G", "R", "G"),
    #     ("G", "G", "R", "O", "R"),
    #     ("B", "G", "G", "B", "O"),
    #     ("B", "O", "B", "O", "B"),
    #     ("R", "G", "G", "G", "G"),
    # ]

    # givens = {
    #     (0, 0): "R",
    #     (-1, -1): "G",
    #     (-1, 1): "G",
    #     (1, -1): "G",
    #     (1, 1): "G",
    # }

    # ## Test 1 (one/two solutions):
    # key = [
    #     ("R", "G", "G", "G", "G"),
    #     ("G", "R", "R", "R", "R"),
    # ]

    # givens = {
    #     (0, 0): "R",
    # }

    ## Test 2 (many solutions):
    key = [
        ("R", "G", "G", "G", "G"),
        ("R", "G", "G", "G", "R"),
        ("R", "G", "G", "R", "R"),
        ("R", "G", "R", "G", "R"),
        ("R", "G", "R", "R", "R"),
        ("R", "R", "R", "R", "R"),
        ("G", "G", "G", "G", "G"),
        ("G", "G", "G", "G", "R"),
        ("G", "G", "G", "R", "R"),
        ("G", "G", "R", "G", "R"),
        ("G", "G", "R", "R", "R"),
        ("G", "R", "R", "R", "R"),
    ]

    givens = {}

    puzzle = Puzzle(size=2, key=key, givens=givens)

    solver = Solver(puzzle)
    max_solutions = 200
    solutions, complete = solver.solve_all_solutions(max_solutions=max_solutions)
    count = len(solutions)
    if count == 0:
        print("No solution found.")
        return
    view_mode = "grid"
    if count == 1:
        print("Unique solution found.")
    elif not complete:
        print(
            f"At least {count} solutions found (capped at {max_solutions}); "
            f"displaying in {view_mode} view."
        )
    else:
        print(f"{count} solutions found; displaying in {view_mode} view.")

    color_map = {
        "G": "#5aa469",
        "O": "#e4a84e",
        "B": "#6aaed6",
        "R": "#d0635e",
    }
    display_puzzle(puzzle, color_map=color_map)
    if count == 1:
        display_solution(puzzle.size, solutions[0], color_map=color_map)
    elif view_mode == "grid":
        display_solution_grid(
            puzzle.size, solutions, per_page=30, cols=6, color_map=color_map
        )
    else:
        display_solutions(puzzle.size, solutions, color_map=color_map)


def generate() -> None:
    ## Generator example:
    color_map = {
        "G": "#5aa469",
        "O": "#e4a84e",
        "B": "#6aaed6",
        "R": "#d0635e",
    }
    size = 4
    key_size = 10
    givens = {
        (0, 0): "R",
        (-1, -1): "G",
        (-1, 1): "G",
        (1, -1): "G",
        (1, 1): "G",
    }
    # disallowed = [
    #     ("R", "R", "R", "R", "R"),
    # ]
    gen = Generator(
        size=size,
        givens=givens,
        # disallowed_patterns=disallowed,
        colors=color_map.keys(),
        key_size=key_size,
    )
    max_attempts = 50_000
    key = gen.find_unique_key(max_attempts=max_attempts)
    if key is None:
        print(f"No unique key found after {max_attempts} attempts.")
        return
    print(key)
    puzzle = Puzzle(size=size, key=key, givens=givens)
    display_puzzle(puzzle, color_map=color_map)
    solver = Solver(puzzle)
    solutions, complete = solver.solve_all_solutions(max_solutions=2)
    assert complete and len(solutions) == 1
    display_solution(puzzle.size, solutions[0], color_map=color_map)


if __name__ == "__main__":
    # solve()
    generate()
