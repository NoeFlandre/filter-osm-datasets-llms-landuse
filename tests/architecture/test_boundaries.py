import ast
from pathlib import Path

DOMAIN = Path(__file__).parents[2] / "src" / "landuse_filter" / "domain"
ALLOWED = {
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "hashlib",
    "json",
    "math",
    "re",
    "typing",
    "numpy",
    "landuse_filter",
}


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


def test_domain_imports_only_pure_modules():
    for path in DOMAIN.glob("*.py"):
        bad = set(imports(path)) - ALLOWED
        assert not bad, f"{path.name} imports {bad}"


def test_domain_does_not_import_outer_layers():
    for path in DOMAIN.glob("*.py"):
        text = path.read_text()
        for layer in ("adapters", "application", "cli"):
            assert f"landuse_filter.{layer}" not in text, path.name
