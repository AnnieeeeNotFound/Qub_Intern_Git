# %%
#! nvidia-smi

# !pip install ipywidgets
# !pip install triton-windows

# %%
import itertools
from datetime import datetime
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import shutil #added to detect cl.exe

# %%

try:
    PATH = Path(__file__).parent
except NameError:
    PATH = Path.cwd()

# INFO Training Parameters
batch_size = 64
data_path = './data'
layer_data_path = './layer_data'
layer_log_path = './layer_log'
scale1 = 128
scale2 = 8192 # INFO Increased from 4096 for more accurate scaling
num_steps = 100
dis_steps = 20
dims = [784, 512, 512]
num_classes = 10
logger = logging.getLogger(__name__)

# INFO ensure paths exist
Path(data_path).mkdir(parents=True, exist_ok=True)
Path(layer_data_path).mkdir(parents=True, exist_ok=True)
Path(layer_log_path).mkdir(parents=True, exist_ok=True)

dtype = torch.float
# INFO Check if GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    logger.info("Running on CPU")

train_loader, test_loader = None, None

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True # INFO Allow TensorFloat32 on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high') # INFO Improves performance for linear layers
torch.manual_seed(77)

# %%

def overlay_y_on_x(x, y):
    # INFO Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0  # INFO zero out the first 10 pixels of input data
    # INFO the y-label 'activates' the corresponding pixel
    x_[range(x.shape[0]), y] = x.max()
    return x_

# %%
class MIF(nn.Linear):
    def __init__(
        self,
        in_features, 
        out_features, 
        layer_index,
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

        # INFO Pre-allocate tensors
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
        # my patch 8.21: INFO torch.compile's Inductor backend needs MSVC (cl.exe) on Windows;
        # fall back to eager execution when no C++ compiler is available
        if shutil.which("cl") is not None:
            self.compiled_mem_step = torch.compile(self.mem_step, fullgraph=True)
        else:
            self.compiled_mem_step = self.mem_step
    
    def mem_step(self, _input, w, b, x1, x2, G1, G2, v):
        _input_mm = F.linear(_input, w, b) #synaptic current flowing through the 512 neurons

        v = (v + (_input_mm + G1 * self.E1 + G2 * self.E2) /
             self.C) / (1 + ((G1 + G2) / self.C))
        v.clamp_(0.0, 1000.0)  # INFO More explosion proofing
        inv_tau = 1.0 / self.tau
        x1 = inv_tau * (  # v[t] or v[t+1] both fine
            (1 - x1) * torch.sigmoid(((v-self.E1)-self.v_on)/self.k_th) -
            x1 * torch.sigmoid((self.v_off-(v-self.E1))/self.k_th)) + x1
        x2 = inv_tau * (  # v[t] or v[t+1] both fine
            (1 - x2) * torch.sigmoid(((v-self.E2)-self.v_on)/self.k_th) -
            x2 * torch.sigmoid((self.v_off-(v-self.E2))/self.k_th)) + x2
        # INFO prevent explosion from floating-point rounding
        x1.clamp_(0.0, 1.0)
        x2.clamp_(0.0, 1.0)
    
        G1 = (x1/self.R_on + (1 - x1)/self.R_off)
        G2 = (x2/self.R_on + (1 - x2)/self.R_off)

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

        for step in range(_input_pos.size(0)): # WARN Assumes first dim is time
            _input_in = torch.cat([_input_pos[step], _input_neg[step]], dim=0)
            x1_in = torch.cat([x1_pos_next, x1_neg_next], dim=0)
            x2_in = torch.cat([x2_pos_next, x2_neg_next], dim=0)
            G1_in = torch.cat([G1_pos_next, G1_neg_next], dim=0)
            G2_in = torch.cat([G2_pos_next, G2_neg_next], dim=0)
            v_in = torch.cat([v_pos_next, v_neg_next], dim=0)

            x1_all, x2_all, G1_all, G2_all, v_all = self.compiled_mem_step(
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

    def train_layer(self, _input_pos, _input_neg, prev_g_pos, prev_g_neg, is_first_layer):
        v_pos_norm, v_neg_norm, g_pos, g_neg = self.forward(_input_pos, _input_neg)
        batches_loss = F.softplus(self.threshold - (g_pos - g_neg))
        prev_loss = F.softplus(self.threshold - (prev_g_pos - prev_g_neg))
        # loss = ((batches_loss + prev_loss)/2).mean() + F.sigmoid((g_pos + g_neg)/self.threshold).mean()
        # + (lambda_peer * peer_loss)
        
        loss = batches_loss.mean() if is_first_layer else ((batches_loss + prev_loss)/2).mean()
        self.opt.zero_grad()
        loss.backward()  # INFO local backward-pass
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()  # INFO update weights

        # with torch.no_grad():
        #     neuron_activity = self.weight.grad.abs().mean(dim=1)
        #     activity_mask = neuron_activity / (neuron_activity.max() + 1e-8)

        #     # Dead neurons get a 'jitter' to kick them out of the 0.0 rut
        #     # Broadcast to [out_features, in_features]
        #     noise_scale = (1.0 - activity_mask).unsqueeze(1)
        #     jitter = torch.randn_like(self.weight) * noise_scale * 0.01
        #     self.weight.add_(jitter)
        return (v_pos_norm, v_neg_norm, g_pos.detach(), g_neg.detach(), loss.detach())

    def predict(self, _input):
        with torch.no_grad():
            input_batch_size = _input.size(1)
            v_history = torch.empty(num_steps, input_batch_size, self.out_features, device=device)
            activity = torch.zeros(input_batch_size, device=device)
            x1_pos_next = self.x1_pos.detach()
            x2_pos_next = self.x2_pos.detach()
            G1_pos_next = self.G1_pos.detach()
            G2_pos_next = self.G2_pos.detach()
            v_pos_next = self.v_pos.detach()
            # INFO Single forward pass
            for step in range(_input.size(0)):
                x1_pos_next, x2_pos_next, G1_pos_next, G2_pos_next, v_pos_next = self.compiled_mem_step(
                    _input[step], self.weight, self.bias, x1_pos_next, x2_pos_next, G1_pos_next, G2_pos_next, v_pos_next)
                v_history[step] = v_pos_next
                activity += v_pos_next.pow(2).mean(1)
            eps = 1e-8
            v_norm = v_history / (v_history.norm(p=2, dim=2, keepdim=True) + eps)
            # v_norm = v_history / (v_history.norm(p=1, dim=2, keepdim=True) + eps)
            goodness = activity/_input.size(0)
            return v_norm, goodness

    # INFO Reset tensors
    def reset(self):
        with torch.no_grad():
            self.x1_pos.detach_().fill_(0.0238)
            self.x2_pos.detach_().fill_(0.0238)
            self.x1_neg.detach_().fill_(0.0238)
            self.x2_neg.detach_().fill_(0.0238)
            
            self.G1_pos.detach_().copy_(self.x1_pos / self.R_on + (1 - self.x1_pos) / self.R_off)
            self.G2_pos.detach_().copy_(self.x2_pos / self.R_on + (1 - self.x2_pos) / self.R_off)
            self.G1_neg.detach_().copy_(self.x1_neg / self.R_on + (1 - self.x1_neg) / self.R_off)
            self.G2_neg.detach_().copy_(self.x2_neg / self.R_on + (1 - self.x2_neg) / self.R_off)
            
            self.v_pos.detach_().zero_()
            self.v_neg.detach_().zero_()

# %%

class Alpha(nn.Module):
    def __init__(self, feature_size, batch_size, num_steps, device, tau_alpha=4.0):
        super(Alpha, self).__init__()
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device

        self.tau_alpha = tau_alpha
        self.feature_size = feature_size

    def forward_alpha(self, _input, a, I):
        a = -a/self.tau_alpha + _input
        I = (a-I)/self.tau_alpha + I
        return a, I
    
    def forward(self, _input_x_pos, _input_x_neg, dis_steps):
        inp0_pos = torch.zeros((1, self.feature_size), device=device)
        inp0_neg = torch.zeros((1, self.feature_size), device=device)
        a0_pos, I0_pos = self.init_Alpha(batch_size, self.feature_size)
        a0_neg, I0_neg = self.init_Alpha(batch_size, self.feature_size)
        
        I0_pos_history = torch.empty(num_steps, self.batch_size, self.feature_size, device=device)
        I0_neg_history = torch.empty(num_steps, self.batch_size, self.feature_size, device=device)
        for step in range(num_steps):
            if step % dis_steps == 0:
                a0_pos, I0_pos = self.forward_alpha(_input_x_pos, a0_pos, I0_pos)
                a0_neg, I0_neg = self.forward_alpha(_input_x_neg, a0_neg, I0_neg)
            else:
                a0_neg, I0_neg = self.forward_alpha(inp0_pos, a0_neg, I0_neg)
                a0_pos, I0_pos = self.forward_alpha(inp0_neg, a0_pos, I0_pos)
            I0_pos_history[step] = I0_pos
            I0_neg_history[step] = I0_neg

        return I0_pos_history, I0_neg_history

    def init_Alpha(self, batch_size, *args):
        I = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        a = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        return a, I

# %%
class Net(nn.Module):
    def __init__(self, in_features, batch_size):
        super().__init__()

        self.layers = nn.ModuleList() #create empty list of MIF
        self.alpha0 = Alpha(in_features, batch_size, num_steps, device) #esentially calling Alpha.forward()
    
    def train_next_layer(self, x_pos, x_neg):
        _input_pos_h, _input_neg_h = self.alpha0(x_pos, x_neg, dis_steps=dis_steps)
        _input_pos_h, _input_neg_h = _input_pos_h / scale1, _input_neg_h / scale1
        g_pos, g_neg = torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)

        for i in range(len(self.layers)-1):
            layer: MIF = self.layers[i]
            # INFO Forward only first (passing)
            v_pos_history, v_neg_history, g_pos, g_neg = layer.forward(_input_pos_h, _input_neg_h)
            _input_pos_h, _input_neg_h = v_pos_history / scale1, v_neg_history / scale1  # INFO keep average v between layers consistent
            layer.reset()  #reset the neurons

        # INFO Train last layer
        last_layer: MIF = self.layers[-1]
        v_pos_history, v_neg_history, g_pos, g_neg, loss = last_layer.train_layer(
            _input_pos_h, _input_neg_h, g_pos, g_neg, len(self.layers) <= 1) #update weights
        last_layer.reset()
        return loss.item()

    def predict(self, x):
        goodness_per_label = []
        x_batch_size = x.size(0)

        x_expanded = x.repeat_interleave(10, dim=0) # [Batch*10, 784]
        labels = torch.arange(10).tile(x_batch_size)  # [0,1...9, 0,1...9]
        h = overlay_y_on_x(x_expanded, labels)

        # INFO Loop through all labels
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
        
            _input_pos_h, _ = self.alpha0(h, h, dis_steps=dis_steps)
            _input_pos_h = _input_pos_h / scale1

            for layer in self.layers:  # INFO inner loop over the layers
                h, g = layer.predict(_input_pos_h) #forward ,make output new input 
                _input_pos_h = h / scale1
                layer.reset()
                goodness.append(g)
            goodness_per_label.append(sum(goodness))
            # goodness_per_label.append(goodness[-1])

        goodness_per_label = torch.stack(goodness_per_label, dim=1)
        return goodness_per_label.argmax(1)
    
    def load_layers(self, filepaths, batch_size):
        self.layers = nn.ModuleList() #empty list of layers
        for i, filepath in enumerate(filepaths):
            state_dict = torch.load(filepath, map_location=device) #weight matrix
            sizes = state_dict['weight'].shape
            layer = MIF(sizes[1], sizes[0], i, 0.01, batch_size=batch_size).to(device)
            layer.load_state_dict(state_dict) #load weights into the layers
            self.layers.append(layer)

# %%
# INFO Trains a net and saves it, set same layer to True if training the last layer, and False to train a new layer
def train_and_save(in_dims, out_dims, name, dataset: str = "mnist", lr=0.002, batch_size=64, epochs=20, checkpoint_intervals=10, net_paths: list = [], same_layer = False):
    net: Net
    min_epoch = 0


    #configure net for later training
       
    if net_paths == []: #first layer
        net = Net(in_dims, batch_size).to(device) #  create empty net
        net.layers = nn.ModuleList() # create empty layer

    else: # load previous checkpoint
        net = load_net(net_paths, batch_size)
        if same_layer: #resume training on last layer
            match = re.search(r'-e(\d+)', net_paths[-1]) # graph epoch number from filename
            if match: min_epoch = int(match.group(1)) # which epoch are we on?
        
    if not same_layer: # add new layer
        layer = MIF(in_dims, out_dims, 0, lr, batch_size).to(device) # create new untrained layer
        net.layers.append(layer) # stack it 

    global train_loader, test_loader
    train_loader, test_loader = get_dataset(dataset, batch_size)

    # naming and logging
    for i in range(0, epochs, checkpoint_intervals): 
        curr_epochs = min_epoch+i+checkpoint_intervals
        full_name = f'l{len(net.layers)}-{name}_{dataset}-b{batch_size}-lr{lr}-e{curr_epochs}-i{in_dims}-o{out_dims}'
        
        log_file = Path(f'{layer_log_path}/{full_name}.log')
        if log_file.exists():
            log_file.unlink()
            print(f"Existing log deleted: {log_file}")
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        logging.basicConfig(
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                file_handler,
                console_handler,
            ],
            force=True,
            level=logging.DEBUG,)
        logger.info(f'Training {full_name} on epoch {curr_epochs} for {epochs} epochs')
         # the TRAINING insdie the logging loop
        losses = train_net(net, checkpoint_intervals)

        save_path = f"{layer_data_path}/{full_name}.pth"
        logger.info(f'Saving layer to {save_path}')
        try:
            torch.save(net.state_dict(), save_path) # dump net weight into the file
        except:
            logger.error(f'Failed to save layer to {save_path}')

    return losses


def load_net(filepaths: list, batch_size):
    checkpoint = torch.load(layer_data_path + '/' + filepaths[0], map_location=device)
    layer_count = len([k for k in checkpoint.keys() if 'weight' in k])

    _, in_dims = checkpoint[f'layers.{0}.weight'].shape
    net = Net(in_dims, batch_size).to(device)
    net.layers = nn.ModuleList()

    for i in range(layer_count):
        out_dims, in_dims = checkpoint[f'layers.{i}.weight'].shape
        net.layers.append(MIF(in_dims, out_dims, i, 0.01, batch_size=batch_size).to(device))
    net.load_state_dict(checkpoint)
    logger.info(f"Loaded net(s) {filepaths[0]}")

    for i in range(1, len(filepaths)):
        checkpoint = torch.load(layer_data_path + '/' + filepaths[i], map_location=device)
        out_dims, in_dims = checkpoint[f'weight'].shape
        layer = MIF(in_dims, out_dims, i, 0.01, batch_size=batch_size).to(device)
        layer.load_state_dict(checkpoint)
        net.layers.append(layer)
        logger.info(f"Loaded layer {filepaths[i]}")
    return net


def train_net(net: Net, epochs=20):
    eval_step = 234 # 234 batches between accuracy acheck
    losses = [] # lsit of loss of each batch

    if train_loader == None or test_loader == None: # training data loaded?
        logger.error("Loader empty")
        return
    
    for epoch in range(epochs):
        test = itertools.cycle(test_loader) # test data
        pbar = tqdm(enumerate(iter(train_loader)), total=len(train_loader)) # progress bar

        for batch_idx, (x, y) in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) #move to GPU

            # INFO generate positive samples (i.e., correct labels overlayed on input data)
            x_pos = overlay_y_on_x(x, y)

            # INFO generate negative samples (i.e., incorrect labels overlayed on input data)
            shift = torch.randint(1, num_classes, (x.size(0),), device=device)
            rnd = (y + shift) % num_classes
            x_neg = overlay_y_on_x(x, rnd)

            batch_loss = net.train_next_layer(x_pos, x_neg)

            # add description
            desc = f'Batch{batch_idx+1}|Loss:{batch_loss:.6f}|Epoch:{epoch+1}/{epochs}'
            pbar.set_description(desc)
            logger.info(desc)
            losses.append(batch_loss)

            if batch_idx % eval_step == eval_step - 1:
                x_te, y_te = next(test)
                x_te, y_te = x_te.to(device), y_te.to(device)
                acc = 100 * net.predict(x_te).eq(y_te).float().mean().item()
                eval_step_loss = sum(losses[-eval_step:])/eval_step
                logger.info(f'\nTest accuracy:{acc}%|Avg loss:{eval_step_loss}\n')
    return losses

# INFO Prepares dataset based on name
def get_dataset(name: str, batch_size):
    def flatten_tensor(x):  # INFO needs to be a named function
        return torch.flatten(x)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Lambda(flatten_tensor)])

    match name:
        case "mnist":
            train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True)
            test_set = datasets.MNIST(data_path, train=False,download=True, transform=transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             drop_last=True, pin_memory=True)
        case "fmnist":
            train_set = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True)
            test_set = datasets.FashionMNIST(data_path, train=False,download=True, transform=transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             drop_last=True, pin_memory=True)
        case "kmnist":
            train_set = datasets.KMNIST(data_path, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True)
            test_set = datasets.KMNIST(data_path, train=False,download=True, transform=transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             drop_last=True, pin_memory=True)
        case "cifar10":
            train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True)
            test_set = datasets.CIFAR10(data_path, train=False,download=True, transform=transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             drop_last=True, pin_memory=True)

    return train_loader, test_loader

# INFO Load net and predict based on dataset
def predict_net(filepaths: list, dataset: str):
    global batch_size
    batch_size = 1000

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(f'{layer_log_path}/'+'{:%Y-%m-%d}.log'.format(datetime.now()))
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            file_handler,
            console_handler,
        ],
        force=True,
        level=logging.DEBUG,)

    test_loader = get_dataset(dataset, batch_size)[1]
    correct = 0
    total = 0

    net = load_net(filepaths, batch_size)

    logger.info(f"Testing {filepaths} on {dataset} dataset")
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        predictions = net.predict(x)
        correct += predictions.eq(y).sum().item()
        total += y.size(0)
        logger.debug(f"Correct:{correct}")
        logger.debug(f"Total:{total}")
        logger.info(f'Test accuracy: {100 * correct / total:.2f}%')

# %%
if __name__ == "__main__":
    # losses = train_and_save(dims[0], dims[1], 'k', "mnist", 0.001, 
    #                         epochs=100, checkpoint_intervals=20)
    # losses = train_and_save(dims[1], dims[2], 'k', "mnist", 0.01, 
    #                         epochs=100, checkpoint_intervals=20, net_paths=['l1-k_mnist-b64-lr0.001-e20-i784-o512.pth'])
    # logger.info("Comments: k stands for 100 epochs test")
    pass

# %%
predict_net(['l2-k_mnist-b64-lr0.01-e40-i512-o512.pth'], "mnist") # , 'l2_h6-b64-lr0.01-e40+-i512-o512.pth'
# Best MNIST (20 epochs): layer 1 97.47, + layer 2 97.58
# NEW Best MNIST (40 epochs): 97.58, + layer 2 98.21


