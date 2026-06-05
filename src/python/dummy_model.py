# scripts/dummy_model.py
import sys

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def forward(self, x):
        # x: [B,8,8,119]
        B = x.shape[0]
        # uniform policy
        policy = torch.ones(B, 4672, dtype=torch.float32, device=x.device) / 4672
        # zero value
        value  = torch.zeros(B, 1, dtype=torch.float32, device=x.device)
        return policy, value
    def save(self, path):
        torch.save(self.state_dict(), path)
        #torch.jit.save(torch.jit.script(self), path)
if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "dummy_model.pt"
    model = DummyModel()
    scripted = torch.jit.script(model)
    scripted.save(output_path)
    print(f"Saved dummy model to {output_path}")
