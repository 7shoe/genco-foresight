"""Allocated CPU parity, training-support diagnostic, and optional XPU smoke."""
from common import load_config
from scripts.parity import evaluate
from scripts.calibration_pilot import main as calibration_pilot


def main():
    evaluate(load_config())
    calibration_pilot()


if __name__ == "__main__":
    main()
