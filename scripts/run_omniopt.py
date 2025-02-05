import argparse
import yaml
import os
from torch3dseg.utils.config import load_config_direct


## Definition of hyperparameter
def parse_arguments():
    parser = argparse.ArgumentParser(description="Wrapper for Train-skript for OmniOpt")
    parser.add_argument("--f_maps", type=int, default=32, help="Initial number of feature maps")
    parser.add_argument("--num_groups", type=int, default=8, help="Number of groups in GroupNorm")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for optimizer")
    
    return vars(parser.parse_args())

def get_unique_filename(base_filename="output.yml"):
    
    if not os.path.exists(base_filename):
        return base_filename
    
    base_name, ext = os.path.splitext(base_filename)
    
    i = 1
    while os.path.exists(f"{base_name}_{i}{ext}"):
        i += 1
    return f"{base_name}_{i}{ext}"

### Magic
def write_to_yaml(args, filepath):
    config = {
        "device": "cuda:0",
        "model": {
            "name": "UNet3D",
            "in_channels": 1,
            "out_channels": 4,
            "layer_order": "gcr",
            "f_maps": args["f_maps"],
            "num_groups": args["num_groups"],
            "final_sigmoid": False,
            "is_segmentation": True
        },
        "loss": {
            "name": "BCEDiceLoss",
            "ignore_index": None,
            "skip_last_target": False
        },
        "optimizer": {
            "learning_rate": args["learning_rate"],
            "weight_decay": 0.00001
        },
        "eval_metric": {
            "name": "MeanIoU",
            "threshold": 0.4,
            "use_last_target": True
        },
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": 0.2,
            "patience": 20
        },
        "trainer": {
            "eval_score_higher_is_better": True,
            "checkpoint_dir": "./checkpoints/NEAPEL_run_005",
            "resume": None,
            "pre_trained": None,
            "validate_after_iters": 100,
            "log_after_iters": 100,
            "max_num_epochs": 50,
            "max_num_iterations": 100000
        }
    }
    
    with open(filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Arguments saved to {filepath}")



def main():
    args = parse_arguments()

    filepath = get_unique_filename(base_filename="omniopt_run.yml")
    write_to_yaml(args, filepath=filepath)
    config = load_config_direct(filepath)
    print(config)
    ## Training
    



if __name__ == "__main__":
    main()
