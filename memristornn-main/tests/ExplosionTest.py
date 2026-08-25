import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

dtype = torch.float
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    print("Running on CPU")

num_steps = 100
batch_size=64

class MIF(nn.Linear):
    def __init__(
        self,
        in_features, 
        out_features, 
        layer_index,
        use_explosion_proof,
        lr=0.01,
        batch_size=64,
        bias=False,
        R_on=1000,
        R_off=100000,
        v_on=110,
        v_off=5,
        tau=100,
        E1=0,
        E2=50,
        C=100 * 10**(-6),
        k_th=0.6 * 25,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.layer_index = layer_index
        self.batch_size = batch_size
        self.threshold = 1
        self.R_on = R_on
        self.R_off = R_off
        self.v_on = v_on
        self.v_off = v_off
        self.tau = tau
        self.E1 = E1
        self.E2 = E2
        self.C = C
        self.k_th = k_th
        self.use_explosion_proof = use_explosion_proof

        self.x1_pos = torch.ones((self.batch_size, self.out_features), device=device, dtype=dtype) * 0.0238
        self.x2_pos = torch.ones((self.batch_size, self.out_features), device=device, dtype=dtype) * 0.0238
        self.G1_pos = self.x1_pos / self.R_on + (1-self.x1_pos)/self.R_off
        self.G2_pos = self.x2_pos / self.R_on + (1-self.x2_pos)/self.R_off
        self.v_pos = torch.zeros((self.batch_size, self.out_features), device=device, dtype=dtype)
        self.x1_neg = torch.ones((self.batch_size, self.out_features), device=device, dtype=dtype) * 0.0238
        self.x2_neg = torch.ones((self.batch_size, self.out_features), device=device, dtype=dtype) * 0.0238
        self.G1_neg = self.x1_neg / self.R_on + (1-self.x1_neg)/self.R_off
        self.G2_neg = self.x2_neg / self.R_on + (1-self.x2_neg)/self.R_off
        self.v_neg = torch.zeros((self.batch_size, self.out_features), device=device, dtype=dtype)

        self.opt = Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=1e-5)
        print("MIF initialized")

    def mem_step(self, _input, w, b, x1, x2, G1, G2, v):
        _input_mm = F.linear(_input, w, b)

        v = (v + (_input_mm + G1 * self.E1 + G2 * self.E2) /
             self.C) / (1 + ((G1 + G2) / self.C))
        # v.clamp_(0.0, 1000.0)  # INFO More explosion proofing
        inv_tau = 1.0 / self.tau
        x1 = inv_tau * (  # v[t] or v[t+1] both fine
            (1 - x1) * torch.sigmoid(((v-self.E1)-self.v_on)/self.k_th) -
            x1 * torch.sigmoid((self.v_off-(v-self.E1))/self.k_th)) + x1
        x2 = inv_tau * (  # v[t] or v[t+1] both fine
            (1 - x2) * torch.sigmoid(((v-self.E2)-self.v_on)/self.k_th) -
            x2 * torch.sigmoid((self.v_off-(v-self.E2))/self.k_th)) + x2
        # INFO prevent explosion from floating-point rounding
        # x1.clamp_(0.0, 1.0)
        # x2.clamp_(0.0, 1.0)
    
        G1 = (x1/self.R_on + (1 - x1)/self.R_off)
        G2 = (x2/self.R_on + (1 - x2)/self.R_off)

        return x1, x2, G1, G2, v

    def mem_step_exp_prone(self, _input, w, b, x1, x2, G1, G2, v):
        _input_mm = F.linear(_input, w, b)
        
        v = (_input_mm-G1*(v-self.E1)-G2*(v-self.E2))/self.C + v
        x1 = 1/self.tau*((1-x1)/(1+torch.exp((self.v_on-(v-self.E1))/self.k_th)) - x1/(
            # v[t] or v[t+1] both fine
            1+torch.exp(((v-self.E1)-self.v_off)/self.k_th))) + x1
        x2 = 1/self.tau*((1-x2)/(1+torch.exp((self.v_on-(v-self.E2))/self.k_th)) - x2/(
            # v[t] or v[t+1] both fine
            1+torch.exp(((v-self.E2)-self.v_off)/self.k_th))) + x2
        G1 = x1/self.R_on + (1-x1)/self.R_off
        G2 = x2/self.R_on + (1-x2)/self.R_off

        return x1, x2, G1, G2, v
    
    def forward(self, _input_pos, _input_neg):
        v_pos_history = torch.empty(num_steps, self.batch_size, self.out_features, device=device)
        v_neg_history = torch.empty(num_steps, self.batch_size, self.out_features, device=device)

        activity_pos = torch.zeros(self.batch_size, device=device)
        activity_neg = torch.zeros(self.batch_size, device=device)

        # INFO Do last loop but now with history
        x1_pos_next = self.x1_pos.detach()
        x2_pos_next = self.x2_pos.detach()
        G1_pos_next = self.G1_pos.detach()
        G2_pos_next = self.G2_pos.detach()
        v_pos_next = self.v_pos.detach()
        x1_neg_next = self.x1_neg.detach()
        x2_neg_next = self.x2_neg.detach()
        G1_neg_next = self.G1_neg.detach()
        G2_neg_next = self.G2_neg.detach()
        v_neg_next = self.v_neg.detach()

        for step in range(_input_pos.size(0)): # WARN Hardcoded kinda maybe its fine
            _input_in = torch.cat([_input_pos[step], _input_neg[step]], dim=0)
            x1_in = torch.cat([x1_pos_next, x1_neg_next], dim=0)
            x2_in = torch.cat([x2_pos_next, x2_neg_next], dim=0)
            G1_in = torch.cat([G1_pos_next, G1_neg_next], dim=0)
            G2_in = torch.cat([G2_pos_next, G2_neg_next], dim=0)
            v_in = torch.cat([v_pos_next, v_neg_next], dim=0)

            if self.use_explosion_proof:
                x1_all, x2_all, G1_all, G2_all, v_all = self.mem_step(
                    _input_in, self.weight, self.bias, x1_in, x2_in, G1_in, G2_in, v_in)
            else:
                x1_all, x2_all, G1_all, G2_all, v_all = self.mem_step_exp_prone(
                    _input_in, self.weight, self.bias, x1_in, x2_in, G1_in, G2_in, v_in)
            
            v_pos_next, v_neg_next = torch.chunk(v_all, 2, dim=0)
            x1_pos_next, x1_neg_next = torch.chunk(x1_all, 2, dim=0)
            x2_pos_next, x2_neg_next = torch.chunk(x2_all, 2, dim=0)
            G1_pos_next, G1_neg_next = torch.chunk(G1_all, 2, dim=0)
            G2_pos_next, G2_neg_next = torch.chunk(G2_all, 2, dim=0)

            v_pos_history[step] = v_pos_next.detach() # t, batch, dim
            v_neg_history[step] = v_neg_next.detach()

            activity_pos += v_pos_next.pow(2).mean(1)  # batch
            activity_neg += v_neg_next.pow(2).mean(1)
            
            x1_pos_next, x2_pos_next, G1_pos_next, G2_pos_next, v_pos_next = x1_pos_next.detach(), x2_pos_next.detach(), G1_pos_next.detach(), G2_pos_next.detach(), v_pos_next.detach()
            x1_neg_next, x2_neg_next, G1_neg_next, G2_neg_next, v_neg_next = x1_neg_next.detach(), x2_neg_next.detach(), G1_neg_next.detach(), G2_neg_next.detach(), v_neg_next.detach()

        eps = 1e-8
        v_pos_norm = v_pos_history / (v_pos_history.norm(p=2, dim=2, keepdim=True) + eps)
        v_neg_norm = v_neg_history / (v_neg_history.norm(p=2, dim=2, keepdim=True) + eps)

        g_pos = activity_pos/_input_pos.size(0)
        g_neg = activity_neg/_input_neg.size(0)

        return (v_pos_norm.detach(), v_neg_norm.detach(), g_pos, g_neg)


mif_explosion_proof = MIF(128, 128, 0, True).to(device)
mif_explosion_prone = MIF(128, 128, 0, False).to(device)
with torch.no_grad():
    mif_explosion_proof.weight.fill_(1.0)

input_pos = torch.ones((num_steps, batch_size, 128), device=device) * 1
input_neg = torch.zeros((num_steps, batch_size, 128), device=device) 

print("\nmif_explosion_proof")
v_pos, v_neg, g_pos, g_neg = mif_explosion_proof.forward(input_pos, input_neg)
print(f"v_pos has NaN: {torch.any(torch.isnan(v_pos)).item()}")
print(f"v_neg has NaN: {torch.any(torch.isnan(v_neg)).item()}")
print(f"g_pos has NaN: {torch.any(torch.isnan(g_pos)).item()}")
print(f"g_neg has NaN: {torch.any(torch.isnan(g_neg)).item()}")

print("\nmif_explosion_prone")
v_pos, v_neg, g_pos, g_neg = mif_explosion_prone.forward(input_pos, input_neg)
print(f"v_pos has NaN: {torch.any(torch.isnan(v_pos)).item()}")
print(f"v_neg has NaN: {torch.any(torch.isnan(v_neg)).item()}")
print(f"g_pos has NaN: {torch.any(torch.isnan(g_pos)).item()}")
print(f"g_neg has NaN: {torch.any(torch.isnan(g_neg)).item()}")
