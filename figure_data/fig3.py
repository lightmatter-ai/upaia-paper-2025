import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, laplace, logistic

# Set the style and context for seaborn plots
sns.set_context('paper', font_scale=1.5)

# Define a color palette
palette = sns.color_palette('deep')

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# -----------------------------
# Plot 1: Vector Transfer Function
# -----------------------------

# Load the calibration data
calib_data_path = "vector_sweeps.npz"
calib_data = np.load(calib_data_path)['arr_0']

# Reshape the calibration data
num_rows, num_cols, num_points = calib_data.shape
total_curves = num_rows * num_cols
observed_codes = calib_data.reshape(total_curves, num_points) / 4

# Generate expected ADC codes
expected_codes = np.tile(np.arange(-1024, 1024), (total_curves, 1))

# Flatten the data for plotting
df_vector = pd.DataFrame({
    'Expected ADC Code': expected_codes.flatten(),
    'Observed ADC Code': observed_codes.flatten(),
})

# Plot the mean observed ADC code with standard deviation shading
sns.lineplot(
    x='Expected ADC Code',
    y='Observed ADC Code',
    data=df_vector,
    ax=axes[0],
    errorbar='sd',
    color=palette[0],
    estimator='mean'
)

# Plot two random example curves
random_indices = np.random.choice(total_curves, 2, replace=False)
for idx in random_indices:
    axes[0].plot(
        expected_codes[idx],
        observed_codes[idx],
        linestyle='--',
        alpha=0.5
    )

# Set labels for the first subplot
axes[0].set_xlabel("Expected ADC Code")
axes[0].set_ylabel("Observed ADC Code")

# -----------------------------
# Plot 2: Weight Transfer Functions
# -----------------------------

# Load weight sweeps data
weight_sweeps_path = 'weight_sweeps.npy'
weight_sweeps = np.load(weight_sweeps_path)

# Reshape weight sweeps data
num_weight_curves, _, num_weight_points = weight_sweeps.shape
total_weight_curves = num_weight_curves * weight_sweeps.shape[1]
flattened_weights = weight_sweeps.reshape(-1)
weight_codes = np.tile(np.arange(num_weight_points), total_weight_curves)

# Create DataFrame for weight transfer functions
df_weight = pd.DataFrame({
    'Weight Code': weight_codes,
    'Observed ADC Code': flattened_weights,
})

# Plot the mean observed ADC code with standard deviation shading
sns.lineplot(
    x='Weight Code',
    y='Observed ADC Code',
    data=df_weight,
    ax=axes[1],
    errorbar='sd',
    color=palette[1],
    estimator='mean'
)

# Plot two random example curves
random_indices_weight = np.random.choice(total_weight_curves, 2, replace=False)
for idx in random_indices_weight:
    axes[1].plot(
        np.arange(num_weight_points),
        weight_sweeps.reshape(total_weight_curves, num_weight_points)[idx],
        marker='o',
        markersize=2,
        linestyle='',
        alpha=0.5
    )

# Set labels for the second subplot
axes[1].set_xlabel('Weight Code')
axes[1].set_ylabel('Observed ADC Code')

# -----------------------------
# Plot 3: Computation Error Distribution with Fits
# -----------------------------

# Load error distribution data
error_data_path = "error_dists.npz"
error_data = np.load(error_data_path)
observed_errors = error_data['observed']
expected_errors = error_data['expected']

# Calculate the differences
differences = expected_errors - observed_errors
combined_diff = differences.flatten()

# Fit Gaussian distribution
mu, std = norm.fit(combined_diff)

# Fit Laplace distribution
laplace_loc, laplace_scale = laplace.fit(combined_diff)

# Fit Logistic distribution
logistic_loc, logistic_scale = logistic.fit(combined_diff)

# Compute histogram data
counts, bin_edges = np.histogram(combined_diff, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the histogram
axes[2].semilogy(bin_centers, counts, color=palette[2], label='Data')

# Plot Gaussian fit
x_values = np.linspace(combined_diff.min(), combined_diff.max(), 100)
gaussian_fit = norm.pdf(x_values, mu, std)
axes[2].semilogy(x_values, gaussian_fit, 'k--', linewidth=2, label='Gaussian Fit')

# Plot Logistic fit
logistic_fit = logistic.pdf(x_values, logistic_loc, logistic_scale)
axes[2].plot(x_values, logistic_fit, color='b', linewidth=2, linestyle='-', label='Logistic Fit')

# Set labels and limits for the third subplot
axes[2].set_xlabel('Difference (Expected - Observed)')
axes[2].set_ylabel('Density')
axes[2].set_ylim([1e-4, 1])

# Add legend to the third subplot
axes[2].legend()

# -----------------------------
# Finalize and Save the Figure
# -----------------------------

plt.tight_layout()
output_image = 'fig3.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')

# Open the saved figure (macOS specific)
if os.name == 'posix':
    os.system(f'open {output_image}')
elif os.name == 'nt':
    os.startfile(output_image)
else:
    print(f"Please open {output_image} manually.")