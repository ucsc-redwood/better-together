import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the devices and applications
device1 = "3A021JEHN02756"
device2 = "9b034f1b"
applications = ["cifar-dense", "cifar-sparse", "tree"]

# Create mapping of file paths to titles
image_paths = {}
for device in [device1, device2]:
    for app in applications:
        path = f"data/bm_logs/{device}/{app}/vk/{device}_{app}_vulkan_ratio.png"
        # Convert app name to display title
        title = app.replace("cifar-", "CIFAR-").replace("-", " ").title()
        image_paths[path] = title

# Create a figure with 2x3 subplots
fig = plt.figure(figsize=(18, 10))

# Plot images for first device (top row)
for idx, (path, title) in enumerate(list(image_paths.items())[:3]):
    plt.subplot(2, 3, idx + 1)
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.title(f"{title} on {device1}")
    plt.axis("off")

# Plot images for second device (bottom row)
for idx, (path, title) in enumerate(list(image_paths.items())[3:]):
    plt.subplot(2, 3, idx + 4)
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.title(f"{title} on {device2}")
    plt.axis("off")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig("device_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
