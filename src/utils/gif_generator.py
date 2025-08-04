"""
GIF generation utilities for creating animated sequences from rendered images.
"""

import os
import glob
import argparse
from pathlib import Path
from PIL import Image
import re
import numpy as np


def natural_sort_key(text):
    """
    Natural sorting key to handle numeric sequences properly.
    Converts 'image_1.png', 'image_10.png', 'image_2.png' 
    to proper order: 1, 2, 10 instead of 1, 10, 2
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]


def find_image_files(directory, extensions=('.png', '.jpg', '.jpeg')):
    """Find all image files in directory and sort them naturally."""
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase
        pattern = os.path.join(directory, f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern, recursive=False))
    
    # Sort naturally to handle numeric sequences
    image_files = sorted(image_files, key=natural_sort_key)
    
    return image_files


def resize_image_if_needed(image, max_size=800):
    """Resize image if it's too large, maintaining aspect ratio."""
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def create_gif(image_directory, output_path, duration=100, loop=0, 
               max_size=800, optimize=True, quality=85):
    """
    Create an animated GIF from a directory of sequential images.
    
    Args:
        image_directory (str): Path to directory containing sequential images
        output_path (str): Path for output GIF file
        duration (int): Duration between frames in milliseconds (default: 100ms = 10fps)
        loop (int): Number of loops (0 = infinite loop, default: 0)
        max_size (int): Maximum dimension for resizing large images (default: 800px)
        optimize (bool): Optimize GIF file size (default: True)
        quality (int): JPEG quality for optimization (1-100, default: 85)
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎬 Creating GIF from images in: {image_directory}")
    
    # Find all image files
    image_files = find_image_files(image_directory)
    
    if not image_files:
        print(f"❌ No image files found in {image_directory}")
        print("   Supported formats: .png, .jpg, .jpeg (case insensitive)")
        return False
    
    print(f"📸 Found {len(image_files)} images")
    
    # Load and process images
    images = []
    for i, img_path in enumerate(image_files):
        try:
            print(f"   Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            img = resize_image_if_needed(img, max_size)
            
            images.append(img)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load {img_path}: {e}")
            continue
    
    if not images:
        print("❌ No valid images could be loaded")
        return False
    
    print(f"✅ Successfully loaded {len(images)} images")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create GIF
    print(f"🎥 Creating GIF: {output_path}")
    print(f"   Frame duration: {duration}ms ({1000/duration:.1f} fps)")
    print(f"   Loop count: {'infinite' if loop == 0 else loop}")
    print(f"   Image size: {images[0].size}")
    
    try:
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=optimize,
            quality=quality
        )
        
        # Get file size
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✅ GIF created successfully!")
        print(f"   Output: {output_path}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Total frames: {len(images)}")
        print(f"   Total duration: {len(images) * duration / 1000:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        return False


def create_gif_from_renders(renders_dir, output_name=None, **kwargs):
    """
    Create GIF from NeRF renders directory.
    Convenience function for common NeRF use case.
    
    Args:
        renders_dir (str): Path to renders directory (e.g., 'build/renders')
        output_name (str): Output filename (default: auto-generated)
        **kwargs: Additional arguments passed to create_gif()
    """
    renders_path = Path(renders_dir)
    
    if not renders_path.exists():
        print(f"❌ Renders directory not found: {renders_dir}")
        return False
    
    # Auto-generate output name if not provided
    if output_name is None:
        output_name = f"nerf_animation_{renders_path.name}.gif"
    
    # Ensure .gif extension
    if not output_name.lower().endswith('.gif'):
        output_name += '.gif'
    
    # Create output in the same parent directory as renders
    output_path = renders_path.parent / output_name
    
    return create_gif(str(renders_path), str(output_path), **kwargs)


def batch_create_gifs(base_directory, pattern="*", **kwargs):
    """
    Create GIFs from multiple subdirectories.
    Useful for processing multiple NeRF experiments.
    
    Args:
        base_directory (str): Base directory containing subdirectories of images
        pattern (str): Pattern to match subdirectories (default: "*")
        **kwargs: Additional arguments passed to create_gif()
    """
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_directory}")
        return
    
    # Find matching subdirectories
    subdirs = []
    for item in base_path.glob(pattern):
        if item.is_dir():
            # Check if it contains images
            if find_image_files(str(item)):
                subdirs.append(item)
    
    if not subdirs:
        print(f"❌ No image directories found matching pattern: {pattern}")
        return
    
    print(f"🎬 Creating GIFs for {len(subdirs)} directories")
    
    successful = 0
    for subdir in subdirs:
        output_name = f"{subdir.name}_animation.gif"
        output_path = subdir.parent / output_name
        
        print(f"\n--- Processing: {subdir.name} ---")
        if create_gif(str(subdir), str(output_path), **kwargs):
            successful += 1
    
    print(f"\n✅ Successfully created {successful}/{len(subdirs)} GIFs")


def main():
    """Command line interface for GIF generation."""
    parser = argparse.ArgumentParser(description='Create animated GIFs from sequential images')
    parser.add_argument('input_dir', help='Directory containing sequential images')
    parser.add_argument('--output', '-o', help='Output GIF path (default: auto-generated)')
    parser.add_argument('--fps', type=float, default=10, 
                       help='Frames per second (default: 10)')
    parser.add_argument('--duration', type=int, 
                       help='Frame duration in milliseconds (overrides fps)')
    parser.add_argument('--max-size', type=int, default=800,
                       help='Maximum image dimension in pixels (default: 800)')
    parser.add_argument('--loop', type=int, default=0,
                       help='Number of loops (0 = infinite, default: 0)')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable GIF optimization')
    parser.add_argument('--quality', type=int, default=85,
                       help='Optimization quality 1-100 (default: 85)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all subdirectories in input_dir')
    parser.add_argument('--pattern', default='*',
                       help='Pattern for batch processing (default: "*")')
    
    args = parser.parse_args()
    
    # Calculate duration from fps if not explicitly set
    if args.duration is None:
        duration = int(1000 / args.fps)  # Convert fps to milliseconds
    else:
        duration = args.duration
    
    if args.batch:
        # Batch processing mode
        batch_create_gifs(
            args.input_dir,
            pattern=args.pattern,
            duration=duration,
            loop=args.loop,
            max_size=args.max_size,
            optimize=not args.no_optimize,
            quality=args.quality
        )
    else:
        # Single directory mode
        if args.output is None:
            # Auto-generate output name
            input_path = Path(args.input_dir)
            args.output = str(input_path.parent / f"{input_path.name}_animation.gif")
        
        success = create_gif(
            args.input_dir,
            args.output,
            duration=duration,
            loop=args.loop,
            max_size=args.max_size,
            optimize=not args.no_optimize,
            quality=args.quality
        )
        
        if not success:
            exit(1)


if __name__ == '__main__':
    main()
