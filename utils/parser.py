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


def get_split_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("src_dir", type=str,
                           help="Noise or kernels  to be moved")
    argparser.add_argument("train_share", type=float,
                           help="Percentage of files to be used for training")

    return argparser


def get_extract_noise_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src_dir", type=str, default=None,
                           help="Image directory from which content is to be used.")
    argparser.add_argument("--dest_dir", type=str, default=None,
                           help="Directory in which to save the results.")
    argparser.add_argument("--noise_level", type=int, default=3,
                           help="How strong extracted noise should be.")
    argparser.add_argument("--window_size", type=int, default=7,
                           help="Size of window to be used for computing average.")
    argparser.add_argument("--blur_kernel_size", type=int, default=0,
                           help="Size of blur kernel to be used.")
    argparser.add_argument("--operation", type=str, default="noise",
                           help="Operation to be done: extract noise or denoise images.")

    return argparser


def get_concat_videos_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("vid1", type=str, default=None,
                           help="First video.")
    argparser.add_argument("vid2", type=str, default=None,
                           help="Second video")
    argparser.add_argument("--stack", type=str, default="v", choices=["h", "v"],
                           help="Stack videos vertically or horizontally.")

    return argparser


def get_make_film_dataset_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("src_path", type=str, default=None,
                           help="Path to film video to be used.")
    argparser.add_argument("dest_dir", type=str, default=None,
                           help="Directory in which results are to be saved.")
    argparser.add_argument("frequency", type=int, default=None,
                           help="How often (in frames) a frame should be saved.")
    argparser.add_argument("--prefix", type=str, default="img",
                           help="Image prefix.")

    return argparser


def get_visualize_kernels_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("src_dir", type=str,
                           help="Source kernels directory to be used.")
    argparser.add_argument("dest_dir", type=str,
                           help="Directory in which resulting images are to be put.")

    return argparser
