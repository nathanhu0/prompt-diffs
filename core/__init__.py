"""Side-effect: apply process-wide torch backend tweaks that any GPU-using script
in this repo should inherit. Runs once per process at the first `import core` or
`from core.X import Y`. Keep this file's responsibility narrow: ONE-PLACE
configuration of torch backends, no other module logic.
"""
import torch

# Disable cuDNN's SDPA fused-attention backend. Hypothesized load-bearing fix for
# SALVE soft-prompt bf16 NaN at step ~250 on sphinx9 (H100, sm_90). cuDNN 9.19
# has documented bf16 backward dQ NaN regressions on Hopper with variable-length
# attention masks (pytorch#166211, cudnn-frontend#160). On Ampere (A100/A6000)
# the cuDNN backend isn't selected by SDPA as a candidate anyway, so this line
# is a strict no-op there. Set here (in the `core` package init) so that ANY
# `from core.* import …` in any runner script in this repo inherits the fix
# without each runner having to repeat the call. UNVALIDATED — needs one
# H100 SALVE rerun to confirm (job 15998102).
torch.backends.cuda.enable_cudnn_sdp(False)
