import time
import torch
import numpy as np
from thop import profile

try:
    from network.network import Network_Resnet, Network_ConvNext
except ImportError:
    pass

def benchmark(model, device, input_shape=(1, 3, 224, 224)):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    with torch.no_grad():
        dummy_output = model(dummy_input)
        embedding_dim = dummy_output.shape[1]  

    dummy_gallery = torch.randn(1000, embedding_dim).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    iterations = 100
    timings = []

    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()

            probe_features = model(dummy_input)

            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)
    
    avg_time = np.mean(timings)
    

    match_timings = []
    match_iterations = 1000

    with torch.no_grad():
        for _ in range(match_iterations):
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()

            scores = torch.matmul(probe_features, dummy_gallery.T)

            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            match_timings.append(end - start)

    avg_match_time = np.mean(match_timings)
    

    print("=" * 30)
    print(f"Embedding Time: {avg_time * 1000:.2f} ms/image")
    print(f"Matching Time: {avg_match_time * 1000:.2f}ms/match")  
    print(f"Model: {model.__class__.__name__}")
    print(f"Total latency: {(avg_time + avg_match_time) * 1000:.2f} ms/image")
    print(f"Max Throughput: {1 / (avg_time + avg_match_time):.2f} FPS")
    print("=" * 30)

def measure_flops(model):
    input_image = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    macs, params = profile(model, inputs=(input_image,), verbose=False)

    print("=" * 30)
    print(f"FLOPS (MACs x2): {macs * 2 / 1e9:.2f} GFLOPS")
    print(f"Parameters: {params / 1e6:.2f} Million")
    print("=" * 30)