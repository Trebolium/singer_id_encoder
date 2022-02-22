from utils import print_args
from solver import SingerIdentityEncoder
from pathlib import Path
import argparse, pdb

def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes a set of arguments to run the singer identity encoder model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # path specifications    
    parser.add_argument("-rid", "--run_id", type=str, default='testRuns', help= "Name of destination model directory and associated files.\
        If --new_run_id specified,this becomes the name of the model directory from which ckpt is extracted for pretrained weights")
    parser.add_argument("-nrid", "--new_run_id", type=str, default=None, help= \
        "If not None, this becomes the name of the new destination model directory and associated files, trained using ckpt from model specified in -run_id.")
    parser.add_argument("-fd", "--feature_dir", type=Path, default="./example_dataset", help= \
        "Path to directory of to feature dataset, which must contain train, val directories and feat_params.yaml file")
    parser.add_argument("-md", "--models_dir", type=Path, default="./", help=\
        "Define the parent directory for all model directories")

    #schedulers
    parser.add_argument("-te", "--tb_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots for in tensorboard.")
    parser.add_argument("-se", "--save_every", type=int, default=1000, help= \
        "Number of steps between updates of the model on the disk. Overwritten at every save")
    parser.add_argument("-ti", "--train_iters", type=int, default=200, help= "Number of training steps to take before passing back to validation steps")
    parser.add_argument("-vi", "--val_iters", type=int, default=10, help= "Number of validation steps to take before passing back to training steps")
    parser.add_argument("-p", "--patience", type=int, default=25, help= "Determines how long EarlyStopping waits before ceasing training")
    parser.add_argument("-stp", "--stop_at_step", type=int, default=100000, help= "Upper limit for number of steps before ceasing training")

    #framework setup
    parser.add_argument("-lri", "--learning_rate_init", type=int, default=1e-4, help= "Choose which cuda driver to use.")
    parser.add_argument("-spb", "--speakers_per_batch", type=int, default=8, help= "Choose which cuda driver to use.")
    parser.add_argument("-ups", "--utterances_per_speaker", type=int, default=10, help= "Choose which cuda driver to use.")
    parser.add_argument("-wc", "--which_cuda", type=int, default=0, help= "Choose which cuda driver to use.")
    parser.add_argument("-ul", "--use_loss", type=str, default='ge2e', help= "Choose mode for determining loss value")

    #model setup
    parser.add_argument("-hs", "--model_hidden_size", type=int, default=256, help= "Number of dimensions in hidden layer.")
    parser.add_argument("-es", "--model_embedding_size", type=int, default=256, help= "Model embedding size.")
    parser.add_argument("-nl", "--num_layers", type=int, default=3, help= "Number of LSTM stacks in model.")
    parser.add_argument("-nt", "--num_timesteps", type=int, default=307, help= "Number of timesteps used in feature example fed to network")

    parser.add_argument("-n", "--notes", type=str, default='', help= "Add these notes which will be saved to a config text file that gets saved in your saved directory")
    config = parser.parse_args()
    
    # Process arguments
    config.models_dir.mkdir(exist_ok=True)
    config.string_sum = str(config)
    print_args(config, parser)

    encoder = SingerIdentityEncoder(config)

    encoder.train()
    