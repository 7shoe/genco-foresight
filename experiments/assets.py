"""Original asset preparation command."""
import argparse
from power_fm_foresight._config import load_config
from power_fm_foresight._data.assets import *

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3.yaml", help="Currently, v3 is the default")
    parser.add_argument("--download", action="store_true", help="Explicitly allow missing-asset downloads")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference-partitions", nargs="*", type=int,
                        help="Download specified 200-scenario official OPFData partitions")
    args = parser.parse_args()
    config = load_config(args.config)
    print("Checkpoint assets:", checkpoint_assets(config, args.download), flush=True)
    if args.profile:
        print("Profile:", eia_profile(config, args.download), flush=True)
    if args.reference_partitions is not None:
        print("Reference data:", converted_partitions(args.reference_partitions,
              Path(config["paths"]["assets_root"])/("official_"+config["task"]["canonical_case"].removeprefix("pglib_opf_")), args.download, config=config), flush=True)


if __name__ == "__main__":
    main()
