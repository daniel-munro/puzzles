"""Solve a logic grid puzzle.

Encode the puzzle in a YAML file, e.g.:

```yaml
categories:
  Person: [Alice, Bob, Carol]
  Pet: [Cat, Dog, Fish]
  Color: [Red, Blue, Green]

clues:
  - match: [Alice, Dog]
  - nomatch: [Bob, Red]

  - xor:
      - match: [Carol, Cat]
      - match: [Carol, Blue]

  - xor:
      - and:
          - match: [Alice, Fish]
          - match: [Green, Dog]
      - and:
          - match: [Alice, Dog]
          - match: [Green, Fish]

  - less: [nominations, film-noir, Bold Service]
  - diff: [nominations, Gabby Jones, romance, 3]
```
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import yaml
from constraint import AllDifferentConstraint, Problem


class PuzzleSpecError(ValueError):
    pass


@dataclass(frozen=True)
class Term:
    category: str
    value: str


@dataclass(frozen=True)
class ConstraintExpr:
    vars: Tuple[str, ...]
    eval_fn: Callable[[Dict[str, int]], bool]


def _dedupe_vars(variables: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: List[str] = []
    for var in variables:
        if var not in seen:
            ordered.append(var)
            seen.add(var)
    return tuple(ordered)


def _parse_term(
    text: Any,
    value_to_category: Dict[str, str],
) -> Term:
    value = str(text).strip()
    if value not in value_to_category:
        raise PuzzleSpecError(
            f"Unknown value '{value}'. Clues must use values only."
        )
    category = value_to_category[value]
    return Term(category=category, value=value)


def _expr_for_match(
    left: Term,
    right: Term,
    base_category: str,
    base_index: Dict[str, int],
) -> ConstraintExpr:
    left_var = None if left.category == base_category else f"{left.category}:{left.value}"
    right_var = None if right.category == base_category else f"{right.category}:{right.value}"
    vars_used = _dedupe_vars(v for v in (left_var, right_var) if v is not None)

    def eval_fn(assignment: Dict[str, int]) -> bool:
        left_value = (
            base_index[left.value]
            if left_var is None
            else assignment[left_var]
        )
        right_value = (
            base_index[right.value]
            if right_var is None
            else assignment[right_var]
        )
        return left_value == right_value

    return ConstraintExpr(vars_used, eval_fn)


def _expr_for_nomatch(
    left: Term,
    right: Term,
    base_category: str,
    base_index: Dict[str, int],
) -> ConstraintExpr:
    match_expr = _expr_for_match(left, right, base_category, base_index)

    def eval_fn(assignment: Dict[str, int]) -> bool:
        return not match_expr.eval_fn(assignment)

    return ConstraintExpr(match_expr.vars, eval_fn)


def _term_index(
    term: Term,
    base_category: str,
    base_index: Dict[str, int],
    assignment: Dict[str, int],
) -> int:
    if term.category == base_category:
        return base_index[term.value]
    return assignment[f"{term.category}:{term.value}"]


def _ordinal_index(
    term: Term,
    ordinal_category: str,
    base_category: str,
    base_index: Dict[str, int],
    category_index: Dict[str, Dict[str, int]],
    assignment: Dict[str, int],
) -> int:
    if ordinal_category == base_category:
        return _term_index(term, base_category, base_index, assignment)

    base_to_ordinal: Dict[int, int] = {}
    for value, ordinal_idx in category_index[ordinal_category].items():
        base_idx = assignment[f"{ordinal_category}:{value}"]
        base_to_ordinal[base_idx] = ordinal_idx

    term_base_idx = _term_index(term, base_category, base_index, assignment)
    return base_to_ordinal[term_base_idx]


def _expr_for_less_than(
    left: Term,
    right: Term,
    ordinal_category: str,
    base_category: str,
    base_index: Dict[str, int],
    category_index: Dict[str, Dict[str, int]],
) -> ConstraintExpr:
    left_var = None if left.category == base_category else f"{left.category}:{left.value}"
    right_var = (
        None if right.category == base_category else f"{right.category}:{right.value}"
    )
    ordinal_vars = (
        []
        if ordinal_category == base_category
        else [f"{ordinal_category}:{value}" for value in category_index[ordinal_category]]
    )
    vars_used = _dedupe_vars(
        v for v in (left_var, right_var, *ordinal_vars) if v is not None
    )

    def eval_fn(assignment: Dict[str, int]) -> bool:
        return _ordinal_index(
            left,
            ordinal_category,
            base_category,
            base_index,
            category_index,
            assignment,
        ) < _ordinal_index(
            right,
            ordinal_category,
            base_category,
            base_index,
            category_index,
            assignment,
        )

    return ConstraintExpr(vars_used, eval_fn)


def _expr_for_difference(
    left: Term,
    right: Term,
    delta: int,
    ordinal_category: str,
    base_category: str,
    base_index: Dict[str, int],
    category_index: Dict[str, Dict[str, int]],
) -> ConstraintExpr:
    left_var = None if left.category == base_category else f"{left.category}:{left.value}"
    right_var = (
        None if right.category == base_category else f"{right.category}:{right.value}"
    )
    ordinal_vars = (
        []
        if ordinal_category == base_category
        else [f"{ordinal_category}:{value}" for value in category_index[ordinal_category]]
    )
    vars_used = _dedupe_vars(
        v for v in (left_var, right_var, *ordinal_vars) if v is not None
    )

    def eval_fn(assignment: Dict[str, int]) -> bool:
        return (
            _ordinal_index(
                right,
                ordinal_category,
                base_category,
                base_index,
                category_index,
                assignment,
            )
            - _ordinal_index(
                left,
                ordinal_category,
                base_category,
                base_index,
                category_index,
                assignment,
            )
            == delta
        )

    return ConstraintExpr(vars_used, eval_fn)


def _parse_int(value: Any) -> int:
    if isinstance(value, bool):
        raise PuzzleSpecError(f"Invalid integer value '{value}'.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise PuzzleSpecError(f"Invalid integer value '{value}'.")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PuzzleSpecError(f"Invalid integer value '{value}'.") from exc


def _parse_expr(
    clue: Any,
    categories: Dict[str, List[str]],
    base_category: str,
    base_index: Dict[str, int],
    category_index: Dict[str, Dict[str, int]],
    value_to_category: Dict[str, str],
) -> ConstraintExpr:
    if not isinstance(clue, dict) or len(clue) != 1:
        raise PuzzleSpecError(f"Invalid clue format: {clue!r}")
    key, value = next(iter(clue.items()))

    if key in {"match", "nomatch"}:
        if not isinstance(value, list) or len(value) != 2:
            raise PuzzleSpecError(f"'{key}' expects a 2-item list: {clue!r}")
        left = _parse_term(value[0], value_to_category)
        right = _parse_term(value[1], value_to_category)
        if key == "match":
            return _expr_for_match(left, right, base_category, base_index)
        return _expr_for_nomatch(left, right, base_category, base_index)

    if key in {"less", "less_than"}:
        if not isinstance(value, list) or len(value) != 3:
            raise PuzzleSpecError(f"'{key}' expects [category, left, right]: {clue!r}")
        ordinal_category = str(value[0]).strip()
        if ordinal_category not in categories:
            raise PuzzleSpecError(
                f"Unknown category '{ordinal_category}' in '{key}' clue."
            )
        left = _parse_term(value[1], value_to_category)
        right = _parse_term(value[2], value_to_category)
        return _expr_for_less_than(
            left,
            right,
            ordinal_category,
            base_category,
            base_index,
            category_index,
        )

    if key in {"diff", "difference"}:
        if not isinstance(value, list) or len(value) != 4:
            raise PuzzleSpecError(
                f"'{key}' expects [category, left, right, delta]: {clue!r}"
            )
        ordinal_category = str(value[0]).strip()
        if ordinal_category not in categories:
            raise PuzzleSpecError(
                f"Unknown category '{ordinal_category}' in '{key}' clue."
            )
        left = _parse_term(value[1], value_to_category)
        right = _parse_term(value[2], value_to_category)
        delta = _parse_int(value[3])
        return _expr_for_difference(
            left,
            right,
            delta,
            ordinal_category,
            base_category,
            base_index,
            category_index,
        )

    if key in {"and", "xor"}:
        if not isinstance(value, list) or not value:
            raise PuzzleSpecError(f"'{key}' expects a non-empty list: {clue!r}")
        sub_exprs = [
            _parse_expr(
                item,
                categories,
                base_category,
                base_index,
                category_index,
                value_to_category,
            )
            for item in value
        ]
        vars_used = _dedupe_vars(var for expr in sub_exprs for var in expr.vars)

        def eval_fn(assignment: Dict[str, int]) -> bool:
            results = [expr.eval_fn(assignment) for expr in sub_exprs]
            if key == "and":
                return all(results)
            return sum(1 for result in results if result) == 1

        return ConstraintExpr(vars_used, eval_fn)

    raise PuzzleSpecError(f"Unknown clue operator '{key}'.")


def _validate_categories(categories: Dict[str, List[str]]) -> Dict[str, str]:
    if not categories:
        raise PuzzleSpecError("Puzzle must define at least one category.")
    sizes = {len(values) for values in categories.values()}
    if len(sizes) != 1:
        raise PuzzleSpecError("All categories must have the same number of values.")
    value_to_category: Dict[str, str] = {}
    for category, values in categories.items():
        for value in values:
            if value in value_to_category:
                raise PuzzleSpecError(
                    f"Value '{value}' appears in multiple categories."
                )
            value_to_category[value] = category
    return value_to_category


def _load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise PuzzleSpecError("Puzzle spec must be a mapping.")
    return spec


def _build_solution_rows(
    categories: Dict[str, List[str]],
    base_category: str,
    assignment: Dict[str, int],
) -> List[Dict[str, str]]:
    base_values = categories[base_category]
    rows: List[Dict[str, str]] = []
    index_to_values: Dict[str, List[Optional[str]]] = {}
    for category, values in categories.items():
        if category == base_category:
            index_to_values[category] = list(values)
            continue
        slots: List[Optional[str]] = [None] * len(values)
        for value in values:
            var = f"{category}:{value}"
            slots[assignment[var]] = value
        index_to_values[category] = slots

    for idx, base_value in enumerate(base_values):
        row: Dict[str, str] = {base_category: base_value}
        for category in categories:
            if category == base_category:
                continue
            row[category] = index_to_values[category][idx] or ""
        rows.append(row)
    return rows


class LogicGridSolver:
    def __init__(self, spec: Dict[str, Any]):
        categories = spec.get("categories")
        if not isinstance(categories, dict):
            raise PuzzleSpecError("Puzzle spec must include categories mapping.")
        self.categories = {}
        for name, values in categories.items():
            if not isinstance(values, list):
                raise PuzzleSpecError(
                    f"Category '{name}' must have a list of values."
                )
            self.categories[name] = [str(value) for value in values]
        self.value_to_category = _validate_categories(self.categories)
        self.clues = spec.get("clues", [])
        if self.clues is None:
            self.clues = []
        if not isinstance(self.clues, list):
            raise PuzzleSpecError("Clues must be a list.")
        self.base_category = next(iter(self.categories))
        self.base_index = {
            value: idx for idx, value in enumerate(self.categories[self.base_category])
        }
        self.category_index = {
            category: {value: idx for idx, value in enumerate(values)}
            for category, values in self.categories.items()
        }

    def solve(self) -> Optional[List[Dict[str, str]]]:
        problem = Problem()
        constant_false = False

        for category, values in self.categories.items():
            if category == self.base_category:
                continue
            var_names = []
            for value in values:
                var = f"{category}:{value}"
                var_names.append(var)
                problem.addVariable(var, list(range(len(values))))
            problem.addConstraint(AllDifferentConstraint(), var_names)

        for clue in self.clues:
            expr = _parse_expr(
                clue,
                self.categories,
                self.base_category,
                self.base_index,
                self.category_index,
                self.value_to_category,
            )
            if not expr.vars:
                if not expr.eval_fn({}):
                    constant_false = True
                    break
                continue

            def constraint_fn(*values, _vars=expr.vars, _eval=expr.eval_fn) -> bool:
                assignment = dict(zip(_vars, values))
                return _eval(assignment)

            problem.addConstraint(constraint_fn, expr.vars)

        if constant_false:
            return None

        assignment = problem.getSolution()
        if assignment is None:
            return None
        return _build_solution_rows(self.categories, self.base_category, assignment)


def solve_file(path: str) -> Optional[List[Dict[str, str]]]:
    spec = _load_spec(path)
    return LogicGridSolver(spec).solve()


def _format_solution(
    rows: List[Dict[str, str]], categories: Sequence[str]
) -> str:
    if not rows:
        return "No solution."
    col_widths = {
        category: max(len(category), *(len(row[category]) for row in rows))
        for category in categories
    }
    header = " | ".join(
        f"{category:<{col_widths[category]}}" for category in categories
    )
    divider = "-+-".join("-" * col_widths[category] for category in categories)
    lines = [header, divider]
    for row in rows:
        lines.append(
            " | ".join(
                f"{row[category]:<{col_widths[category]}}" for category in categories
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("Usage: python logic_grid.py <puzzle.yaml>")
        return 2
    solution = solve_file(argv[1])
    if solution is None:
        print("No solution.")
        return 1
    categories = list(solution[0].keys())
    print(_format_solution(solution, categories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
