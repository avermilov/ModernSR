import argparse


def get_train_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config", type=str,
                           help="Config file for training Super Resolution models.")

    return argparser


def get_inference_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str,
                           help="Config file for training Super Resolution models.")
    argparser.add_argument("--in_dir", type=str,
                           help="Input inference image directory.")
    argparser.add_argument("--out_dir", type=str,
                           help="Inference results directory.")
    argparser.add_argument("--workers", type=int, default=0,
                           help="Number of workers for loading inference dataset.")
    argparser.add_argument("--batch_size", type=int, default=1,
                           help="Batch size of inference data loader.")
    argparser.add_argument("--scale", type=int,
                           help="Super Resolution scale of the model.")
    argparser.add_argument("--checkpoint_path", type=str,
                           help="Path to checkpoint to be used for inference.")
    argparser.add_argument("--model", type=str,
                           help="Model type to be used for inference.")

    return argparser
