# ESP32-S3 inference-time estimate
# Model architecture: 5 -> 64 -> 64 -> 1

MACS = 4480
CPU_CLOCK_HZ = 240_000_000

# Assumed CPU cycles required per MAC
cycle_assumptions = [10, 25, 50]

print("ESP32-S3 inference-time estimate")
print("CPU clock: 240 MHz")
print("Model MACs:", MACS)
print()

for cycles in cycle_assumptions:
    total_cycles = MACS * cycles
    time_ms = total_cycles / CPU_CLOCK_HZ * 1000

    print(
        f"{cycles} cycles/MAC: "
        f"{total_cycles:,} cycles, "
        f"{time_ms:.3f} ms"
    )

print()
print("Planning estimate: approximately 1-5 ms per inference.")
print("This is an estimate, not a measurement on physical hardware.")
