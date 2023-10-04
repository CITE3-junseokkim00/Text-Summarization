import argparse
from train import KoT5ConditionalGeneration
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--hparams', default=None, type=str)
parser.add_argument("--model_binary", default='./checkpoint/last.ckpt', type=str)
parser.add_argument("--output_dir", default='koT5_summary', type=str)
args = parser.parse_args()

info = KoT5ConditionalGeneration.load_from_checkpoint(args.model_binary)
info.model.save_pretrained(args.output_dir)