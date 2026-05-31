#!/usr/bin/env python3
"""
MeasureBreaks – Simple Demo (Inductor Backend)
Proactive, cost‑aware static analysis for TorchDynamo graph breaks.
Uses inductor backend to get real runtime graph break counts.
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
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'item':
            self.breaks.append(('item()', node.lineno))
        elif isinstance(node.func, ast.Name) and node.func.id == 'print':
            self.breaks.append(('print()', node.lineno))
        elif (isinstance(node.func, ast.Attribute) and
              isinstance(node.func.value, ast.Name) and
              node.func.value.id == 'np'):
            self.breaks.append(('numpy', node.lineno))
        elif (isinstance(node.func, ast.Attribute) and
              node.func.attr == 'nonzero'):
            self.breaks.append(('dynamic shape (nonzero)', node.lineno))
        self.generic_visit(node)

    def visit_Subscript(self, node):
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

# ---------- 2. Model definitions (source strings) ----------
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
    return x[mask]""",
    "MultiBreak": """def forward(self, x):
    print("fwd")
    if x.sum().item() > 0:
        y = np.array(x.cpu().detach().numpy())
        return torch.tensor(y, device=x.device)
    return x.relu()"""
}

# Actual Module classes for runtime
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

# ---------- 3. Runtime break collection using inductor backend ----------
def runtime_break_count(model, example_input):
    torch._dynamo.reset()
    try:
        # Use default backend (inductor) for real compilation
        explanation = torch._dynamo.explain(model)(example_input)
        return len(explanation.break_reasons)
    except Exception as e:
        # Compilation error indicates a severe break (uncompilable)
        return f"error: {type(e).__name__}"

# ---------- 4. Cost model ----------
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
    print(f"Using device: {device}")
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

    df = pd.DataFrame(results)
    df = df.sort_values("Est. Cost (ms)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df)+1))

    print("\n=== MeasureBreaks Demo (Inductor Backend) ===\n")
    print(df.to_string(index=False))
    df.to_csv("measurebreaks_inductor.csv", index=False)
    print("\nResults saved to measurebreaks_inductor.csv")

if __name__ == "__main__":
    main()
