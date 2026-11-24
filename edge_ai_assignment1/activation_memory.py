# Activation memory calculation for the selected 64-64 model

layer_sizes = [5, 64, 64, 1]

# FP32 uses 4 bytes per value
total_activation_values = sum(layer_sizes)
activation_bytes = total_activation_values * 4

largest_layer = max(layer_sizes)
largest_layer_bytes = largest_layer * 4

print("Selected model: 5 -> 64 -> 64 -> 1")
print("Activation values across all layers:", total_activation_values)
print("Conservative total activation memory:", activation_bytes, "bytes")
print("Largest single layer activation:", largest_layer_bytes, "bytes")
print("Largest layer size:", largest_layer, "float32 values")

