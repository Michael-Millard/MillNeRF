"""
Main entry point for MillNeRF training and rendering.
"""

import argparse
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.training.train import main as train_main
from src.rendering.render import main as render_main
from src.data.prepare import main as prepare_main


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(description='MillNeRF - Neural Radiance Fields Implementation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train a NeRF model')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Path to config file')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint')
    
    # Rendering subcommand
    render_parser = subparsers.add_parser('render', help='Render novel views')
    render_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to model checkpoint')
    render_parser.add_argument('--config', type=str, default='configs/default.yaml',
                              help='Path to config file')
    render_parser.add_argument('--mode', type=str, default='novel',
                              choices=['novel', 'train', 'test'],
                              help='Rendering mode')
    render_parser.add_argument('--output_dir', type=str, default='build/renders',
                              help='Output directory for rendered images')
    
    # Data preparation subcommand
    prep_parser = subparsers.add_parser('prepare', help='Prepare training data')
    prep_parser.add_argument('--images_dir', type=str, required=True,
                            help='Directory containing input images')
    prep_parser.add_argument('--output_dir', type=str, default='data',
                            help='Output directory for processed data')
    prep_parser.add_argument('--resize_factor', type=int, default=4,
                            help='Factor to resize images by')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_main(args)
    elif args.command == 'render':
        render_main(args)
    elif args.command == 'prepare':
        prepare_main(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()