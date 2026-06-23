import matplotlib.pyplot as plt
import re
import numpy as np

# turn data.txt into the data
filename = 'DataFiles/testdata.txt'

generations = []
av_fitness = []

with open(filename, 'r') as file:
    for line in file:
        if 'Gen' in line and 'avg_fitness' in line:
            gen_match = re.search(r'Gen\s+(\d+)/\d+', line)
            if gen_match:
                gen = int(gen_match.group(1))
                generations.append(gen)

        fitness_match = re.search(r'avg_fitness:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if fitness_match:
            fitness = float(fitness_match.group(1))
            av_fitness.append(fitness)

# Convert to numpy arrays
generations = np.array(generations)
av_fitness = np.array(av_fitness)

# Linear regression with numpy.polyfit
slope, intercept = np.polyfit(generations, av_fitness, 1)

# Predicted values for the trend line
y_pred = slope * generations + intercept

# Calculate R-squared
ss_res = np.sum((av_fitness - y_pred) ** 2)   # residual sum of squares
ss_tot = np.sum((av_fitness - np.mean(av_fitness)) ** 2)   # total sum of squares
r_squared = 1 - (ss_res / ss_tot)

print("Slope (avg increase per generation):", slope)
print("Intercept:", intercept)
print("R-squared:", r_squared)

# Plot with regression line
plt.figure(figsize=(10, 5))
plt.plot(generations, av_fitness, marker='o', linestyle='-', color='purple', label="Observed fitness")
plt.plot(generations, y_pred, 'r--', label=f"Trend line (slope={slope:.3f}, R²={r_squared:.3f})")
plt.title("Average Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.grid(True)
plt.legend()
plt.show()
