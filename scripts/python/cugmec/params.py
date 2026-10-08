"""
Read and patch src/cuGMEC_param.h.

parse_param_h() returns every `const`/`constexpr` value and every `using` switch as Python values:
numbers are evaluated (they may refer to earlier constants), trueType/falseType become bools,
strings stay strings, enum names such as Maxwell stay strings, and std::array initializers are kept as text.
"""
import ast
import operator
import re

_DECL = re.compile(r"^[ \t]*(?:constexpr|const)\s+(?P<type>[\w:]+(?:<[^;=]*>)?)\s+(?P<name>\w+)\s*=\s*(?P<value>[^;]*);", re.M)
_USING = re.compile(r"^[ \t]*using\s+(?P<name>\w+)\s*=\s*(?P<value>trueType|falseType|float|double)\s*;", re.M)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(expr, env):
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and isinstance(env.get(node.id), (int, float)):
            return env[node.id]
        raise ValueError
    return ev(ast.parse(expr, mode="eval").body)


def parse_param_h(text):
    """{name: value} for every constant and switch in cuGMEC_param.h (later definitions win)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    found = [(m.start(), m.group("name"), m.group("type"), m.group("value").strip()) for m in _DECL.finditer(text)]
    found += [(m.start(), m.group("name"), "using", m.group("value")) for m in _USING.finditer(text)]
    env = {}
    for _, name, ctype, value in sorted(found):
        if ctype == "using":
            env[name] = {"trueType": True, "falseType": False}.get(value, value)
        elif value.startswith('"'):
            env[name] = ast.literal_eval(value)
        elif "array" in ctype:
            env[name] = value
        else:
            try:
                v = _eval(value, env)
                env[name] = int(v) if ctype == "int" else float(v)
            except (ValueError, SyntaxError, ZeroDivisionError):
                env[name] = value  # enum value or something we do not evaluate
    return env


def _literal(value, ctype):
    if isinstance(value, bool):
        return "trueType" if value else "falseType"
    if isinstance(value, str):
        return value if ctype not in ("std::string", "string") else '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if ctype == "int":
        if float(value) != int(value):
            raise ValueError(f"{value} is not an integer")
        return str(int(value))
    return repr(float(value))  # shortest string that round-trips


def patch_param_h(text, values):
    """Return param.h text with each name in `values` set to its new value. Every name must be defined exactly once."""
    for name, value in values.items():
        if isinstance(value, bool):
            pattern = re.compile(rf"^([ \t]*using\s+{name}\s*=\s*)(?:trueType|falseType)(\s*;)", re.M)
            text, n = pattern.subn(lambda m: m.group(1) + _literal(value, "using") + m.group(2), text)
        else:
            pattern = re.compile(rf"^([ \t]*(?:constexpr|const)\s+(?P<type>[\w:]+)\s+{name}\s*=\s*)[^;]*(;)", re.M)
            text, n = pattern.subn(lambda m: m.group(1) + _literal(value, m.group("type")) + m.group(3), text)
        if n != 1:
            raise ValueError(f"expected one definition of {name} in cuGMEC_param.h, found {n}")
    return text


def check_grid(p):
    """Grid sizes that do not divide the launch geometry in src/cuGMEC_setup.h evenly. Returns a list of problems."""
    nx, ny, nz, ppc = p["gridNx"], p["gridNy"], p["gridNz"], p["ppcNums"]
    ranks = p["hostNums"] * p["devNums"]
    dev_ny = ny / ranks
    bz = nz if nz in (8, 16, 32) else 16
    m2py = 2 if nz == 32 else 4
    markers = nx * ny * nz / ranks * ppc
    rules = [
        (ny % ranks == 0, "gridNy must be divisible by hostNums*devNums"),
        (nx % 128 == 0, "gridNx must be a multiple of 128 (GhostNMGridDimx = 2*gridNx/256)"),
        (dev_ny.is_integer() and dev_ny * nx % 256 == 0, "devNy*gridNx must be a multiple of 256 (LocalNMGridDimx)"),
        (not dev_ny.is_integer() or dev_ny < 2 or dev_ny % 2 == 0, "devNy must be even or 1 (MRK4BlockDimy = 2)"),
        (nz % bz == 0, f"gridNz must be a multiple of {bz} (block x-size)"),
        (ny % m2py == 0, f"gridNy must be a multiple of {m2py} (M2PGridDimy)"),
        ((ny + 4) * nx * nz % 256 == 0, "(gridNy+4)*gridNx*gridNz must be a multiple of 256 (MergeGridDimx)"),
        (markers.is_integer() and markers % 8192 == 0, "markers per GPU must be a multiple of 8192 (PICGridDimx)"),
    ]
    return [text for ok, text in rules if not ok]
