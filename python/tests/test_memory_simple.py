# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx

'''
# Check current setting
sysctl iogpu.wired_limit_mb

sudo sysctl iogpu.wired_limit_mb=<size in MiB>

# Reset to default (0 = automatic allocation)
sudo sysctl iogpu.wired_limit_mb=0

# OR just reboot to reset everything
sudo reboot
'''

if __name__ == "__main__":
    max_size = mx.metal.device_info()["max_recommended_working_set_size"]

    print(f"maximum recommended working set size: {max_size / 1024 / 1024 / 1024:.2f} GiB")

