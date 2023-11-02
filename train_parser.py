import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_config", type=str, default="./configs/vqa2021.yml")
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--output_base_dir", type=str, default="./outputs/")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--debug", action="store_true")

    return parser

def generate_pretrain_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_config", type=str, default="./configs/pretrain_data/pmcp.yml")
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--output_base_dir", type=str, default="./pretrain_outputs/")
    parser.add_argument("--debug", action="store_true")

    return parser
