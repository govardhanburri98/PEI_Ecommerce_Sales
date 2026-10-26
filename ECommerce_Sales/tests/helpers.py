"Small helpers so pytest can use the Databricks notebook files without a cluster."
import ast
import os
from pathlib import Path
from types import SimpleNamespace

# notebooks live in ../src by default; set NOTEBOOK_DIR to point at your own copy
SRC = Path(os.environ.get("NOTEBOOK_DIR", Path(__file__).resolve().parent.parent / "src"))
HEADER = "# Databricks notebook source"
SEPARATOR = "# COMMAND ----------"


def rows(df):
    """Collect a DataFrame as a list of dicts, easier to assert on."""
    return [r.asDict() for r in df.collect()]


def _cells(path):
    text = Path(path).read_text()
    assert text.startswith(HEADER), f"{path} is not a Databricks notebook-source file"
    return [c.strip() for c in text.replace(HEADER, "", 1).split(SEPARATOR) if c.strip()]


def _is_constant(node):
    # only plain UPPER_CASE constants that do not touch spark (paths, schemas, column lists)
    if not isinstance(node, ast.Assign):
        return False
    if not all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
        return False
    return "spark" not in {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}


def _run(path, ns):
    for cell in _cells(path):
        if cell.startswith("# MAGIC %run"):
            target = cell.split("%run", 1)[1].strip()
            _run((Path(path).parent / target).with_suffix(".py"), ns)
        elif cell.startswith("# MAGIC"):
            continue                      # %md / %sql / %pip cells are not python
        else:
            for node in ast.parse(cell).body:
                # run imports, functions and constants; skip cells that read/write real tables
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) \
                        or _is_constant(node):
                    code = compile(ast.Module([node], []), str(path), "exec")
                    exec(code, ns)


def load_notebook(name, spark=None):
    """Load the definitions of a notebook (functions + constants) into a namespace object."""
    ns = {"__name__": f"nb_{name}", "spark": spark}
    _run(SRC / f"{name}.py", ns)
    return SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


def sql_statements(name):
    """Return the SQL text of every %sql cell in a notebook (without the MAGIC prefix)."""
    out = []
    for cell in _cells(SRC / f"{name}.py"):
        if cell.startswith("# MAGIC %sql"):
            lines = [l.replace("# MAGIC", "", 1).strip() for l in cell.splitlines()[1:]]
            out.append("\n".join(lines).strip().rstrip(";"))
    return out
