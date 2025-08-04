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
from src.utils.image_conversion import convert_heic_to_format
from src.utils.colmap_utils import run_colmap_sfm, colmap_to_nerf_format
from src.utils.colmap_debug import suggest_colmap_fixes
from src.utils.gif_generator import create_gif, create_gif_from_renders, batch_create_gifs

def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(description='MillNeRF - Neural Radiance Fields Implementation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Image conversion subcommand
    convert_parser = subparsers.add_parser('convert', help='Convert HEIC images to JPEG/PNG')
    convert_parser.add_argument('--input_dir', type=str, required=True,
                               help='Directory containing HEIC images')
    convert_parser.add_argument('--output_dir', type=str, default=None,
                               help='Output directory (default: input_dir + "_converted")')
    convert_parser.add_argument('--format', type=str, default='JPEG',
                               choices=['JPEG', 'PNG'], help='Output format')
    convert_parser.add_argument('--quality', type=int, default=95,
                               help='JPEG quality (1-100)')
    
    # COLMAP subcommand
    colmap_parser = subparsers.add_parser('colmap', help='Run COLMAP Structure-from-Motion')
    colmap_parser.add_argument('--images_dir', type=str, required=True,
                              help='Directory containing images')
    colmap_parser.add_argument('--output_dir', type=str, default='colmap_output',
                              help='Output directory for COLMAP results')
    colmap_parser.add_argument('--camera_model', type=str, default='OPENCV',
                              choices=['PINHOLE', 'SIMPLE_PINHOLE', 'OPENCV', 'RADIAL'],
                              help='Camera model for COLMAP')

    # COLMAP debug subcommand
    debug_parser = subparsers.add_parser('debug-colmap', help='Debug COLMAP issues')
    debug_parser.add_argument('--database_path', type=str, required=True,
                             help='Path to COLMAP database.db')
    debug_parser.add_argument('--images_dir', type=str,
                             help='Images directory for suggestions')
    
    # Data preparation subcommand
    prep_parser = subparsers.add_parser('prepare', help='Prepare training data')
    prep_parser.add_argument('--images_dir', type=str, required=True,
                            help='Directory containing input images')
    prep_parser.add_argument('--output_dir', type=str, default='data',
                            help='Output directory for processed data')
    prep_parser.add_argument('--resize_factor', type=int, default=4,
                            help='Factor to resize images by')
    
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

    # GIF generation subcommand
    gif_parser = subparsers.add_parser('gif', help='Create animated GIF from image sequence')
    gif_parser.add_argument('input_dir', help='Directory containing sequential images')
    gif_parser.add_argument('--output', '-o', help='Output GIF path (default: auto-generated)')
    gif_parser.add_argument('--fps', type=float, default=10, 
                           help='Frames per second (default: 10)')
    gif_parser.add_argument('--max-size', type=int, default=800,
                           help='Maximum image dimension in pixels (default: 800)')
    gif_parser.add_argument('--loop', type=int, default=0,
                           help='Number of loops (0 = infinite, default: 0)')
    gif_parser.add_argument('--quality', type=int, default=85,
                           help='Optimization quality 1-100 (default: 85)')
    gif_parser.add_argument('--batch', action='store_true',
                           help='Process all subdirectories in input_dir')

    args = parser.parse_args()

    if args.command == 'convert':
        convert_heic_to_format(args.input_dir, args.output_dir, args.format, args.quality)
    elif args.command == 'colmap':
        # Run COLMAP
        sparse_dir = run_colmap_sfm(args.images_dir, args.output_dir, args.camera_model)
        
        if sparse_dir is None:
            print(f"\n🔧 Running COLMAP diagnostics...")
            database_path = os.path.join(args.output_dir, 'database.db')
            suggest_colmap_fixes(args.images_dir, database_path)
        else:
            # Convert to NeRF format
            transforms_file = os.path.join(args.output_dir, 'transforms.json')
            colmap_to_nerf_format(sparse_dir, transforms_file)
            
            print(f"\n✅ COLMAP processing complete!")
            print(f"Camera poses saved to: {transforms_file}")
    elif args.command == 'debug-colmap':
        suggest_colmap_fixes(args.images_dir, args.database_path)
    elif args.command == 'prepare':
        prepare_main(args)
    elif args.command == 'train':
        train_main(args)
    elif args.command == 'render':
        render_main(args)
    elif args.command == 'gif':
        # Handle GIF creation
        duration = int(1000 / args.fps)  # Convert fps to milliseconds
        
        if args.batch:
            success = batch_create_gifs(
                args.input_dir,
                duration=duration,
                loop=args.loop,
                max_size=args.max_size,
                quality=args.quality
            )
        else:
            if args.output is None:
                # Auto-generate output name
                from pathlib import Path
                input_path = Path(args.input_dir)
                args.output = str(input_path.parent / f"{input_path.name}_animation.gif")
            
            success = create_gif(
                args.input_dir,
                args.output,
                duration=duration,
                loop=args.loop,
                max_size=args.max_size,
                quality=args.quality
            )
            
            if not success:
                exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()