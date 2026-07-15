from train_model import generate_data
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate data for exactly 1 scene so we can visualize it clearly
# We will use 15,000 samples to make the cloud density look solid and detailed
X, Y, L, C, C_xyOUT = generate_data(
    samples=250000, 
    scenes=1, 
    m=15, 
    n=15, 
    beta=1.8, 
    L_min=10, 
    L_max=10, 
    Cth_min=0.45, 
    Cth_max=0.45, 
    phase_angle=0, 
    Abase=0.4
)

# Convert outputs to numpy arrays for easy indexing/masking
X = np.array(X)
Y = np.array(Y)
C_xyOUT = np.array(C_xyOUT)
L_val = L[0]            # Extract the L value used for this scene
C_thresh = C[0]         # Extract the threshold used

# 2. Separate clear sky (density = 0) from active clouds (density > 0)
# This lets us style the blue background separately from the white cloud fluff
cloud_mask = C_xyOUT > 0

# 3. Initialize the plot
plt.figure(figsize=(8, 8), facecolor='#1E293B')  # Dark blue-slate frame
ax = plt.axes()
ax.set_facecolor('#0F172A')  # Darker space background

# Plot the clear sky background points (thin blue vapor layer)
ax.scatter(
    X[~cloud_mask], Y[~cloud_mask], 
    c='#1E293B', s=4, alpha=0.15, edgecolors='none'
)

# Plot the active clouds (using a smooth gray-to-white gradient based on density)
cloud_scatter = ax.scatter(
    X[cloud_mask], Y[cloud_mask], 
    c=C_xyOUT[cloud_mask], 
    cmap='gray',  # We'll define a quick linear gradient or use 'Blues_r' / 'gray'
    s=8, 
    alpha=0.8, 
    edgecolors='none'
)

# Use 'cool' or 'gray' as a fallback colormap for the density mapping
# Let's map it to 'Blues_r' or 'bone' for a chilly, realistic sky color palette
cloud_scatter.set_cmap('bone')

# 4. Styling and Labels
ax.set_title(f"Generated Cloud Field (L = {L_val:.1f} km, Threshold = {C_thresh:.2f})", 
             color='white', fontsize=14, pad=15)
ax.set_xlabel("X Coordinate (km)", color='white', labelpad=10)
ax.set_ylabel("Y Coordinate (km)", color='white', labelpad=10)

# Match ticks to the dark style
ax.tick_params(colors='white')
ax.set_xlim(0, L_val)
ax.set_ylim(0, L_val)
ax.set_aspect('equal')  # Ensure the physical grid scale is perfectly square

# Add a colorbar to show the density scale
cbar = plt.colorbar(cloud_scatter, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Cloud Density', color='white', labelpad=10)
cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

# Display the plot
plt.show()