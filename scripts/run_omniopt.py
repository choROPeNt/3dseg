import argparse
import yaml
import os
from torch3dseg.utils.config import load_config_direct

def parse_arguments():
    parser = argparse.ArgumentParser(description="Update YAML configuration with argparse values.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the base YAML configuration file")
    parser.add_argument("--f_maps", type=int, help="Initial number of feature maps")
    parser.add_argument("--num_groups", type=int, help="Number of groups in GroupNorm")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, help="Number of workers for data loading")
    
    return vars(parser.parse_args())

def get_unique_filename(base_filename="omniopt_run01_01.yml"):
    if not os.path.exists(base_filename):
        return base_filename
    
    base_name, ext = os.path.splitext(base_filename)
    i = 1
    while os.path.exists(f"{base_name}_{i}{ext}"):
        i += 1
    return f"{base_name}_{i}{ext}"

def load_and_update_yaml(config_path, args):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if "model" in config:
        if args["f_maps"] is not None:
            config["model"]["f_maps"] = args["f_maps"]
        if args["num_groups"] is not None:
            config["model"]["num_groups"] = args["num_groups"]
    
    if "optimizer" in config and args["learning_rate"] is not None:
        config["optimizer"]["learning_rate"] = args["learning_rate"]
    
    if "loaders" in config:
        if args["batch_size"] is not None:
            config["loaders"]["batch_size"] = args["batch_size"]
        if args["num_workers"] is not None:
            config["loaders"]["num_workers"] = args["num_workers"]
    
    return config

def write_to_yaml(config, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Updated configuration saved to {filepath}")

def main():
    args = parse_arguments()
    config_path = args.pop("config")
    filepath = get_unique_filename()
    updated_config = load_and_update_yaml(config_path, args)
    write_to_yaml(updated_config, filepath)
    config = load_config_direct(filepath)
    print(config)
    ## Training

if __name__ == "__main__":
    main()
