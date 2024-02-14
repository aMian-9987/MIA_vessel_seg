
"""
The self-implement connected component analysis
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import label, distance_transform_edt

# Load NIfTI image and convert to array
img = nib.load('monoE_single/WSL.nii')
data = img.get_fdata()

# Label connected components in the image
labeled_array, num_features = label(data)

# Compute the size of each connected component
component_sizes = {i: np.sum(labeled_array == i) for i in range(1, num_features + 1)}

# Filter out small components based on a size threshold
filtered_components = {label: size for label, size in component_sizes.items() if size >= 2}

# Initialize an array to store the final connected components
final_data = np.zeros_like(data)

# Add only the components that meet the size threshold
for label, size in filtered_components.items():
    final_data[labeled_array == label] = label
print("cleared")

# Print the size of each filtered component (optional, can be commented out)
for component_label, size in filtered_components.items():
    print(f"Component {component_label}: {size} voxels")

# Identify and add the largest components to the final array
num_largest_components = 2
largest_components = sorted(filtered_components, key=filtered_components.get, reverse=True)[:num_largest_components]

# Print the size of the largest components
for i, comp_label in enumerate(largest_components, 1):
    print(f"Component {i} with label {comp_label} has {filtered_components[comp_label]} voxels")

# Set a size threshold for connecting smaller components to the nearest larger one
size_threshold = 300

# Re-initialize the final array to store the largest components
final_data = np.zeros_like(data)

# Add the largest components to the final array
for label in largest_components:
    final_data[labeled_array == label] = label

# Connect smaller components to their nearest larger component if they're below the threshold
for label in range(1, num_features + 1):
    if label in largest_components or component_sizes[label] < size_threshold:
        continue  # Skip if it's a largest component or below the threshold

    # Create a mask for the current small component
    component_mask = (labeled_array == label)

    # Calculate distance to the nearest larger component and connect
    distances_to_large_components = np.full(component_mask.shape, np.inf)
    for large_label in largest_components:
        large_component_mask = (final_data == large_label)
        edt_result = distance_transform_edt(~large_component_mask)
        distances_to_large_components = np.minimum(distances_to_large_components, edt_result)

    # Find the nearest larger component and connect the current small component to it
    min_distance = np.min(distances_to_large_components[component_mask])
    if min_distance < np.inf:
        nearest_point_index = np.argwhere(distances_to_large_components == min_distance)[0]
        nearest_large_component_label = final_data[tuple(nearest_point_index)]
        final_data[component_mask] = nearest_large_component_label

# Save the final result to a new NIfTI file
new_img = nib.Nifti1Image(final_data, img.affine)
nib.save(new_img, 'newnew_WSL.nii')