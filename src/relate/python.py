"""Objective Python-code relation coordinates used by the original result."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

PYTHON_RELATION_NAMES = (
    "cyclomatic_complexity",
    "max_control_depth",
    "distinct_call_sites",
)


@dataclass(frozen=True, slots=True)
class PythonStructure:
    cyclomatic_complexity: float
    max_control_depth: float
    distinct_call_sites: float

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            (
                self.cyclomatic_complexity,
                self.max_control_depth,
                self.distinct_call_sites,
            ),
            dtype=np.float64,
        )


class _StructureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.complexity = 1
        self.depth = 0
        self.max_depth = 0
        self.calls: set[str] = set()
        self._root_seen = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._root_seen:
            return
        self._root_seen = True
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._root_seen:
            return
        self._root_seen = True
        for statement in node.body:
            self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        self._visit_if_chain(node, enter_depth=True)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self._visit_control(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self._visit_control(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self._visit_control(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_control(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_control(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_control(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += max(len(node.values) - 1, 0)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_Match(self, node: ast.Match) -> None:
        self.complexity += max(len(node.cases) - 1, 0)
        self._visit_control(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.add(_call_target(node.func))
        self.generic_visit(node)

    def _visit_if_chain(self, node: ast.If, *, enter_depth: bool) -> None:
        self.complexity += 1
        if enter_depth:
            self._enter_control()
        self.visit(node.test)
        for statement in node.body:
            self.visit(statement)
        if self._has_elif(node):
            self._visit_if_chain(node.orelse[0], enter_depth=False)
        else:
            for statement in node.orelse:
                self.visit(statement)
        if enter_depth:
            self.depth -= 1

    @staticmethod
    def _has_elif(node: ast.If) -> bool:
        return (
            len(node.orelse) == 1
            and isinstance(node.orelse[0], ast.If)
            and node.orelse[0].col_offset == node.col_offset
        )

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_nodes: tuple[ast.AST, ...],
    ) -> None:
        entered = 0
        for generator in generators:
            self.complexity += 1 + len(generator.ifs)
            self.visit(generator.iter)
            self._enter_control()
            entered += 1
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for result_node in result_nodes:
            self.visit(result_node)
        self.depth -= entered

    def _enter_control(self) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

    def _visit_control(self, node: ast.AST) -> None:
        self._enter_control()
        self.generic_visit(node)
        self.depth -= 1


def extract_python_structure(source: str) -> PythonStructure:
    """Extract the three structural coordinates from one top-level function."""
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        raise ValueError("source must contain exactly one top-level function")
    visitor = _StructureVisitor()
    visitor.visit(functions[0])
    return PythonStructure(
        cyclomatic_complexity=float(visitor.complexity),
        max_control_depth=float(visitor.max_depth),
        distinct_call_sites=float(len(visitor.calls)),
    )


def _call_target(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "<dynamic>"
