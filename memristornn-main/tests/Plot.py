import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    PATH = Path(__file__).parent.parent
except NameError:
    PATH = Path.cwd()

def stitch_and_plot_logs(file_pattern):
    all_data = []
    files = sorted(glob.glob(file_pattern))
    print(files)
    
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines: continue
            
            start_match = re.search(r'on epoch (\d+)', lines[0])
            if not start_match: continue
            start_epoch = int(start_match.group(1))
            
            for line in lines:
                match = re.search(r'Loss:([\d\.\-e]+)\|Epoch:(\d+)/(\d+)', line)
                if match:
                    loss = float(match.group(1))
                    rel_epoch = int(match.group(2))
                    total_in_file = int(match.group(3))
                    
                    abs_epoch = start_epoch + rel_epoch - 1 - total_in_file
                    all_data.append({'epoch': abs_epoch, 'loss': loss})
    
    if not all_data:
        print("No data found")
        return

    df = pd.DataFrame(all_data)
    epoch_stats = df.groupby('epoch')['loss'].mean().reset_index()
    
    # INFO Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_stats['epoch'], epoch_stats['loss'], marker='.', linestyle='-')
    plt.title('Training Loss')
    plt.xlabel('Global Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return epoch_stats



file_filter_l1 = "l1-k_mnist-b64-lr0.001-e*-i784-o512" + ".log"
file_filter_l2 = "l2-k_mnist-b64-lr0.01-e*-i512-o512" + ".log"
df = stitch_and_plot_logs(str(PATH/"layer_log"/file_filter_l1))
