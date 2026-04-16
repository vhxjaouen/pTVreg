import re

with open('pTVreg/registration.py', 'r') as f:
    text = f.read()

# 1. Update init to process lambda_reg
init_old = """        self.grid_spacing = grid_spacing
        self.lambda_reg = lambda_reg
        self.n_levels = n_levels"""

init_new = """        self.grid_spacing = grid_spacing
        self.n_levels = n_levels
        
        # Process lambda_reg array / scalar
        if isinstance(lambda_reg, (list, tuple)):
            if len(lambda_reg) == 1:
                base = lambda_reg[0]
                self.lambda_reg = [base * (2 ** (self.n_levels - 1 - i)) for i in range(self.n_levels)]
            elif len(lambda_reg) != self.n_levels:
                raise ValueError(f"lambda_reg list must have {self.n_levels} elements.")
            else:
                self.lambda_reg = list(lambda_reg)
        else:
            base = float(lambda_reg)
            # Default scaling: stiffer at coarse levels, looser at fine levels
            self.lambda_reg = [base * (2 ** (self.n_levels - 1 - i)) for i in range(self.n_levels)]"""
text = text.replace(init_old, init_new)

# 2. Update compute_loss definition
loss_old = "    def compute_loss(self, warped, fixed, flow, csqrt=5e-3):"
loss_new = "    def compute_loss(self, warped, fixed, flow, level_idx, csqrt=5e-3):"
text = text.replace(loss_old, loss_new)

# 3. Update return in compute_loss
ret_old = "        return metric_loss + self.lambda_reg * tv_loss, metric_loss, tv_loss"
ret_new = """        current_lambda = self.lambda_reg[level_idx]
        return metric_loss + current_lambda * tv_loss, metric_loss, tv_loss"""
text = text.replace(ret_old, ret_new)

# 4. Pass level_idx in optimize closure
opt1_old = "                loss, m_loss, tv = self.compute_loss(warped, fixed_curr, flow)"
opt1_new = "                loss, m_loss, tv = self.compute_loss(warped, fixed_curr, flow, i)"
text = text.replace(opt1_old, opt1_new)

opt2_old = "                _, m_loss, true_tv = self.compute_loss(warped, fixed_curr, flow)"
opt2_new = "                _, m_loss, true_tv = self.compute_loss(warped, fixed_curr, flow, i)"
text = text.replace(opt2_old, opt2_new)

with open('pTVreg/registration.py', 'w') as f:
    f.write(text)
