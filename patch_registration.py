import re

with open('pTVreg/registration.py', 'r') as f:
    content = f.read()

# Regex to find everything from "# 2. ADMM" down to "logging the level\n            with torch.no_grad():"
pattern = r"# 2\. ADMM and.*?(?=\# Manually run one more forward pass strictly)"
new_code = """# 2. Setup Optimizer matching COPD_final.m
            # Using standard LBFGS with max_iter = n_iters to let the quasi-Newton solver
            # build up a strong Hessian approximation without ADMM interruptions.
            optimizer = optim.LBFGS([self.knots], lr=lr, max_iter=n_iters, line_search_fn='strong_wolfe')
            
            # 3. Optimization Loop
            def closure():
                optimizer.zero_grad()
                warped, flow = self.forward(i)
                fixed_curr = self.fixed_pyramid[i] if self.fixed_pyramid else None
                loss, m_loss, tv = self.compute_loss(warped, fixed_curr, flow)
                loss.backward()
                return loss

            # LBFGS runs internally for multiple iterations
            for epoch in range(1):
                optimizer.step(closure)
            
            """

replaced = re.sub(pattern, new_code, content, flags=re.DOTALL)
with open('pTVreg/registration.py', 'w') as f:
    f.write(replaced)

