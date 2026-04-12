import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from src.centroiding import get_centroider

image_path = "data/images/Z36_0001.fit"
output_path = "data/images/Z36_0001_centroids.png"

image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)

# Subtract local background so only sharp star spikes remain
background = uniform_filter(image, size=64)
image = np.clip(image - background, 0, None)
print(f"min={image.min():.3f}  max={image.max():.3f}  mean={image.mean():.3f}")

centroider = get_centroider("cog", {"threshold": 0.3, "min_area": 9})
centroids = centroider.extract(image)
print(f"Found {len(centroids)} stars")

fig, ax = plt.subplots(figsize=(10, 10))
vmin, vmax = np.percentile(image, [1, 99])
ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
for cx, cy in centroids:
    ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=1.5)
ax.set_title(f"{image_path}  —  {len(centroids)} centroids found")
ax.axis("off")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved {output_path}")
plt.show()
