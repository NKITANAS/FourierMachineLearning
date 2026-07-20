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
# This lets us style the sky background separately from the cloud fluff
cloud_mask = C_xyOUT > 0

# 3. Initialize the plot, matching the 'Blues_r' palette used in test_stuff/plots.py
# and qml/plots.py. Unlike those two (which raster the whole grid with imshow), this
# one is a scatter plot, so cloud density 1.0 renders as white -- invisible on a plain
# white page. The axes background is set to the same colormap's darkest blue (its
# value-0 color) so the "clear sky" points blend seamlessly into it and the white
# cloud points actually stand out, while everything still comes from one palette.
cloud_cmap = plt.get_cmap('Blues_r')
sky_color = cloud_cmap(0.0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor(sky_color)

# Plot the clear sky background points
ax.scatter(
    X[~cloud_mask], Y[~cloud_mask],
    c=[sky_color], s=4, alpha=0.15, edgecolors='none'
)

# Plot the active clouds. 'Blues_r' (reversed Blues) maps low density to dark blue
# and high density to white, so white = dense cloud and blue = clear sky --
# consistent with the cloud/sky color convention used in the other plots.py files.
cloud_scatter = ax.scatter(
    X[cloud_mask], Y[cloud_mask],
    c=C_xyOUT[cloud_mask],
    cmap=cloud_cmap,
    vmin=0, vmax=1,
    s=8,
    alpha=0.8,
    edgecolors='none'
)

# 4. Styling and Labels
ax.set_title(f"Generated Cloud Field (L = {L_val:.1f} km, Threshold = {C_thresh:.2f})",
             fontsize=14, pad=15)
ax.set_xlabel("X Coordinate (km)", labelpad=10)
ax.set_ylabel("Y Coordinate (km)", labelpad=10)

ax.set_xlim(0, L_val)
ax.set_ylim(0, L_val)
ax.set_aspect('equal')  # Ensure the physical grid scale is perfectly square

# Add a colorbar to show the density scale
cbar = fig.colorbar(cloud_scatter, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Cloud Density', labelpad=10)

# Display the plot
plt.show()
