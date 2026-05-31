#!/usr/bin/env python3
"""
MeasureBreakes – Eager Demo
Proactive, cost‑aware static analysis for TorchDynamo graph breaks.
"""

import ast
import torch
import torch.nn as nn
import pandas as pd

# ---------- 1. Static break detector (AST) ----------
class BreakDetector(ast.NodeVisitor):
    def __init__(self):
        self.breaks = []   # (break_type, line)

    def visit_Call(self, node):
        # .item() call
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'item':
            self.breaks.append(('item()', node.lineno))
        # print()
        elif isinstance(node.func, ast.Name) and node.func.id == 'print':
            self.breaks.append(('print()', node.lineno))
        # numpy call (np.anything)
        elif (isinstance(node.func, ast.Attribute) and
              isinstance(node.func.value, ast.Name) and
              node.func.value.id == 'np'):
            self.breaks.append(('numpy', node.lineno))
        # torch.nonzero()
        elif (isinstance(node.func, ast.Attribute) and
              node.func.attr == 'nonzero'):
            self.breaks.append(('dynamic shape (nonzero)', node.lineno))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # Mask indexing like x[x > 0] -> dynamic shape
        slice_str = ast.unparse(node.slice)
        if '>' in slice_str or '<' in slice_str or '==' in slice_str:
            self.breaks.append(('dynamic shape (mask)', node.lineno))
        self.generic_visit(node)

def static_analyze(func):
    try:
        tree = ast.parse(func)
        d = BreakDetector()
        d.visit(tree)
        return d.breaks
    except SyntaxError:
        return []

# ---------- 2. Model definitions (as source strings) ----------
models = {
    "CleanModel": """def forward(self, x):
    return x.relu() + 1""",
    "ModelWithItem": """def forward(self, x):
    if x.sum().item() > 0:
        return x + 1
    return x - 1""",
    "ModelWithPrint": """def forward(self, x):
    print("Debug:", x.shape)
    return x.relu()""",
    "ModelWithNumpy": """def forward(self, x):
    y = np.array(x.cpu().detach().numpy())
    return torch.tensor(y, device=x.device)""",
    "DataDependentShape": """def forward(self, x):
    mask = x > 0
    return x[mask]""",                     # dynamic shape
    "MultiBreak": """def forward(self, x):
    print("fwd")
    if x.sum().item() > 0:
        y = np.array(x.cpu().detach().numpy())
        return torch.tensor(y, device=x.device)
    return x.relu()"""
}

# Also define actual Module classes for runtime execution
class CleanModel(nn.Module):
    def forward(self, x): return x.relu() + 1
class ModelWithItem(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0: return x + 1
        return x - 1
class ModelWithPrint(nn.Module):
    def forward(self, x):
        print("Debug:", x.shape)
        return x.relu()
class ModelWithNumpy(nn.Module):
    def forward(self, x):
        import numpy as np
        y = np.array(x.cpu().detach().numpy())
        return torch.tensor(y, device=x.device)
class DataDependentShape(nn.Module):
    def forward(self, x):
        mask = x > 0
        return x[mask]
class MultiBreak(nn.Module):
    def forward(self, x):
        print("fwd")
        if x.sum().item() > 0:
            import numpy as np
            y = np.array(x.cpu().detach().numpy())
            return torch.tensor(y, device=x.device)
        return x.relu()

model_instances = {
    "CleanModel": CleanModel(),
    "ModelWithItem": ModelWithItem(),
    "ModelWithPrint": ModelWithPrint(),
    "ModelWithNumpy": ModelWithNumpy(),
    "DataDependentShape": DataDependentShape(),
    "MultiBreak": MultiBreak(),
}

# ---------- 3. Runtime break collection ----------
def runtime_break_count(model, example_input):
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(model, backend="eager")(example_input)
        return len(explanation.break_reasons)
    except Exception:
        return -1   # compilation error counts as severe break

# ---------- 4. Cost model (type -> penalty) ----------
BREAK_COSTS = {
    'numpy': 12,
    'item()': 6,
    'dynamic shape (mask)': 5,
    'dynamic shape (nonzero)': 5,
    'print()': 3,
}
def estimate_cost(static_breaks):
    return sum(BREAK_COSTS.get(bt, 2) for bt, _ in static_breaks)

# ---------- 5. Main pipeline ----------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 10).to(device)

    results = []
    for name, src in models.items():
        static = static_analyze(src)
        static_labels = [f"{bt} (L{ln})" for bt, ln in static]
        runtime = runtime_break_count(model_instances[name], x)
        cost = estimate_cost(static)

        results.append({
            "Model": name,
            "Static Breaks": ", ".join(static_labels) if static_labels else "none",
            "Runtime Break Count": runtime,
            "Est. Cost (ms)": cost,
        })

    # Rank by cost descending
    df = pd.DataFrame(results)
    df = df.sort_values("Est. Cost (ms)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df)+1))

    print("\n=== MeasureBreakes Demo ===\n")
    print(df.to_string(index=False))
    df.to_csv("measurebreaks_eager.csv", index=False)
    print("\nResults saved to measurebreaks_eager.csv")

if __name__ == "__main__":
    main()
