
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from codeletsA.train_teacher02p import YOnlyTeacher, TransformerTeacher

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Parameters from the training scripts
    m_in = 1228
    m_out = 2457
    hidden_dim = 512
    d_model = 128
    nhead = 4
    depth = 2

    # MLP (YOnlyTeacher)
    mlp_model = YOnlyTeacher(m_in=m_in, m_out=m_out, hidden=hidden_dim)
    mlp_params = count_parameters(mlp_model)
    print(f"MLP (YOnlyTeacher) parameters: {mlp_params:,}")

    # Transformer (TransformerTeacher)
    transformer_model = TransformerTeacher(m_in=m_in, m_out=m_out, d_model=d_model, nhead=nhead, depth=depth)
    transformer_params = count_parameters(transformer_model)
    print(f"Transformer (TransformerTeacher) parameters: {transformer_params:,}")
