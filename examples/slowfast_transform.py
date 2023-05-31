"""

Transform for SlowFast

"""
import torch
import nvidia.dali.plugin.pytorch as dalitorch
import nvidia.dali.fn as fn

alpha = 4

def pack_pathway(frames: torch.Tensor):
    fast_pathway = frames

    # Perform temporal sampling from the fast pathway.
    index = torch.linspace(
            0, frames.shape[1] - 1, frames.shape[1] // alpha
        ).long()
    index = index.to(device=fast_pathway.device)
    slow_pathway = torch.index_select(
        frames,
        1,
        index,
    )
    return [slow_pathway, fast_pathway]

# # To use torch_python_function, pass exec_async=False and exec_pipelined=False
# # to the DALI pipeline.
# # This doesn't actually work, since the passed-in function must return a tensor,
# # not a list of tensors.
# def dali_pack_pathway(frames):
#     return dalitorch.fn.torch_python_function(frames, function=pack_pathway, num_outputs=1)

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
      return pack_pathway(frames)
