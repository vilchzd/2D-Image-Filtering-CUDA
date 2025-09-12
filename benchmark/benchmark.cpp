import subprocess
import statistics

image = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\love.png"
grid_sizes = [8, 16, 32, 48]
runs = 15

print("grid,cpu_avg,gpu_avg")
for g in grid_sizes:
    cpu_times, gpu_times = [], []
    for _ in range(runs):
        result = subprocess.run(
            ["main.exe", image, str(g)],
            capture_output=True, text=True
        )
        grid, cpu, gpu = map(int, result.stdout.strip().split(","))
        cpu_times.append(cpu)
        gpu_times.append(gpu)
    print(f"{g},{statistics.mean(cpu_times):.2f},{statistics.mean(gpu_times):.2f}")
