import re
with open("pTVreg/cli.py", "r") as f:
    text = f.read()

text = text.replace(
    'moving_data = (moving_data - np.min(moving_data)) / (np.ptp(moving_data) + 1e-8)',
    'moving_min = np.min(moving_data)\n    moving_ptp = np.ptp(moving_data) + 1e-8\n    moving_data = (moving_data - moving_min) / moving_ptp'
)
text = text.replace(
    'warped_np = warped.detach().cpu().numpy().squeeze()',
    'warped_np = warped.detach().cpu().numpy().squeeze()\n        warped_np = (warped_np * moving_ptp) + moving_min'
)

with open("pTVreg/cli.py", "w") as f:
    f.write(text)
