"""Start the newton open xr profile."""

from dvrk_simulator_base.launch import open_xr_launch


def generate_launch_description():
    return open_xr_launch("newton")
