"""Allocated CPU parity, training-support diagnostic, and optional XPU smoke."""
from power_fm_foresight._config import load_config
from experiments.parity import evaluate
from experiments.calibration_pilot import main as calibration_pilot


def main():
    evaluate(load_config())
    calibration_pilot()


if __name__ == "__main__":
    main()
