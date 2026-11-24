import csv

widths = [32, 128, 256, 512, 2048]

SRAM_BYTES = 300 * 1024
PSRAM_BYTES = 8 * 1024 * 1024

print(f"{'Width':>8} {'Params':>12} {'FP32 bytes':>12} {'INT8 bytes':>12} {'MACs':>12} {'SRAM FP32':>12} {'SRAM INT8':>12} {'PSRAM FP32':>12} {'PSRAM INT8':>12}")

rows = []

for w in widths:
    # Network: 5 -> w -> w -> 1
    params = (5 * w + w) + (w * w + w) + (w + 1)
    fp32 = params * 4
    int8 = params
    macs = 5 * w + w * w + w

    sram_fp32 = fp32 <= SRAM_BYTES
    sram_int8 = int8 <= SRAM_BYTES
    psram_fp32 = fp32 <= PSRAM_BYTES
    psram_int8 = int8 <= PSRAM_BYTES

    row = {
        "width": w,
        "params": params,
        "fp32_bytes": fp32,
        "int8_bytes": int8,
        "macs": macs,
        "fits_300kb_sram_fp32": sram_fp32,
        "fits_300kb_sram_int8": sram_int8,
        "fits_8mb_psram_fp32": psram_fp32,
        "fits_8mb_psram_int8": psram_int8,
    }
    rows.append(row)

    print(
        f"{w:8} {params:12,} {fp32:12,} {int8:12,} {macs:12,} "
        f"{str(sram_fp32):>12} {str(sram_int8):>12} "
        f"{str(psram_fp32):>12} {str(psram_int8):>12}"
    )

with open("mission4_memory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved mission4_memory.csv")

