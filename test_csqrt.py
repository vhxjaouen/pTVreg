import torch
x = torch.linspace(0, 2, 100)
l1 = torch.abs(x)
smooth_l1 = torch.sqrt(x**2 + 5e-3)
print(f"Zero grad for L1: {0}, for smoothed L1: {smooth_l1[1] - smooth_l1[0]}")
