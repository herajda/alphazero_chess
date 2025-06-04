# scripts/dummy_model.py
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def forward(self, x):
        # x: [B,8,8,119]
        B = x.shape[0]
        # uniform policy
        policy = torch.ones(B, 4672, dtype=torch.float32) / 4672
        # zero value
        value  = torch.zeros(B, 1,      dtype=torch.float32)
        return policy, value
    def save(self, path):
        torch.save(self.state_dict(), path)
        #torch.jit.save(torch.jit.script(self), path)
if __name__ == "__main__":
    model = DummyModel()
    scripted = torch.jit.script(model)
    scripted.save("dummy_model.pt")
    print("→ Saved dummy_model.pt")