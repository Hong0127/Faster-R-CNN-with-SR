import argparse
from scripts.train import train
from scripts.test import test

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN with Super Resolution for SOD4SB")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode to run: train or test")

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()

if __name__ == '__main__':
    main()