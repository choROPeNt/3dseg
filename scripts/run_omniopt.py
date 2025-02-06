import argparse
import yaml
import os
from torch3dseg.utils.config import load_config_direct
from torch3dseg.utils import utils

logger = utils.get_logger('ConfigPreProcessing')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Update YAML configuration with argparse values.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the base YAML configuration file")
    parser.add_argument("--dir",type=str, default="configs", help="directory to store the  YAML configuration files")
    parser.add_argument("--checkpoint_dir",type=str, default="checkpoints", help="directory to store the  YAML configuration files")
    ## Hyperparameters
    parser.add_argument("--f_maps", type=int, help="Initial number of feature maps")
    parser.add_argument("--num_groups", type=int, help="Number of groups in GroupNorm")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")

    
    return vars(parser.parse_args())

def get_unique_filename(config_path, save_dir):
    base_name = os.path.splitext(os.path.basename(config_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    
    i = 0
    while os.path.exists(os.path.join(save_dir, f"{base_name}_{i:03d}.yml")):
        i += 1
    
    return os.path.join(save_dir, f"{base_name}_{i:03d}.yml")

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

    if "trainer" in config:
        if args["checkpoint_dir"] is not None:
            config["trainer"]["checkpoint_dir"] = args["checkpoint_dir"]
    
    return config

def write_to_yaml(config, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    logger.info(f"Updated configuration saved to {filepath}")


##########################
## Main Programm
##########################
def main():
    #######################
    ## Pre-Processing
    #######################
    # Get Arguments
    args = parse_arguments()
    config_path = args.pop("config")
    save_dir = args.pop("dir")

    # Create Optimization Name
    filepath = get_unique_filename(config_path, save_dir)
    
    # Update the Checkpoint dir
    args["checkpoint_dir"] = os.path.join(args["checkpoint_dir"],os.path.splitext(os.path.basename(filepath))[0])
    logger.info(f"Set checkpoint dir to {args['checkpoint_dir']}")
    # Update base yml config file
    updated_config = load_and_update_yaml(config_path, args)
    write_to_yaml(updated_config, filepath)

    #######################
    ## Training
    ####################### 
    
    config = load_config_direct(filepath)
    




if __name__ == "__main__":
    main()
