"""
Image format conversion utilities.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

# Register HEIF opener with Pillow
register_heif_opener()


def convert_heic_to_format(input_dir, output_dir=None, output_format='JPEG', quality=95):
    """
    Convert HEIC images to specified format.
    
    Args:
        input_dir (str): Directory containing HEIC images
        output_dir (str): Output directory (if None, uses input_dir + '_converted')
        output_format (str): Output format ('JPEG', 'PNG', etc.)
        quality (int): JPEG quality (1-100, only applies to JPEG)
    
    Returns:
        list: List of converted image paths
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = str(input_path) + '_converted'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all HEIC files (case insensitive)
    heic_files = []
    for ext in ['*.heic', '*.HEIC', '*.heif', '*.HEIF']:
        heic_files.extend(input_path.glob(ext))
    
    if not heic_files:
        print(f"No HEIC files found in {input_dir}")
        return []
    
    print(f"Found {len(heic_files)} HEIC files")
    print(f"Converting to {output_format} format...")
    
    converted_files = []
    
    for heic_file in tqdm(heic_files, desc="Converting images"):
        try:
            # Open HEIC image
            image = Image.open(heic_file)
            
            # Convert to RGB if necessary (for JPEG)
            if output_format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Generate output filename
            if output_format.upper() == 'JPEG':
                output_file = output_path / f"{heic_file.stem}.jpg"
            elif output_format.upper() == 'PNG':
                output_file = output_path / f"{heic_file.stem}.png"
            else:
                output_file = output_path / f"{heic_file.stem}.{output_format.lower()}"
            
            # Save converted image
            save_kwargs = {}
            if output_format.upper() == 'JPEG':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            
            image.save(output_file, format=output_format, **save_kwargs)
            converted_files.append(str(output_file))
            
        except Exception as e:
            print(f"Error converting {heic_file}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_files)} images to {output_dir}")
    return converted_files


def convert_directory_formats(input_dir, output_dir=None, target_format='JPEG', 
                            source_formats=None, quality=95):
    """
    Convert images from various formats to target format.
    
    Args:
        input_dir (str): Directory containing images
        output_dir (str): Output directory
        target_format (str): Target format ('JPEG', 'PNG', etc.)
        source_formats (list): List of source formats to convert from
        quality (int): JPEG quality (1-100)
    """
    if source_formats is None:
        source_formats = ['HEIC', 'HEIF', 'PNG', 'BMP', 'TIFF', 'WEBP']
    
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = str(input_path) + f'_converted_to_{target_format.lower()}'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Register HEIF opener
    register_heif_opener()
    
    # Find all images to convert
    image_files = []
    for fmt in source_formats:
        for ext in [f'*.{fmt.lower()}', f'*.{fmt.upper()}']:
            image_files.extend(input_path.glob(ext))
    
    if not image_files:
        print(f"No images found with formats {source_formats} in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images to convert")
    print(f"Converting to {target_format} format...")
    
    converted_files = []
    
    for img_file in tqdm(image_files, desc="Converting images"):
        try:
            # Open image
            image = Image.open(img_file)
            
            # Convert to RGB for JPEG
            if target_format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Generate output filename
            if target_format.upper() == 'JPEG':
                output_file = output_path / f"{img_file.stem}.jpg"
            else:
                ext = target_format.lower()
                output_file = output_path / f"{img_file.stem}.{ext}"
            
            # Skip if already in target format and same location
            if img_file.suffix.lower() == f'.{target_format.lower()}' and input_path == output_path:
                continue
            
            # Save converted image
            save_kwargs = {}
            if target_format.upper() == 'JPEG':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            
            image.save(output_file, format=target_format, **save_kwargs)
            converted_files.append(str(output_file))
            
        except Exception as e:
            print(f"Error converting {img_file}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_files)} images to {output_dir}")
    return converted_files


def main():
    """Command line interface for image conversion."""
    parser = argparse.ArgumentParser(description='Convert HEIC and other image formats')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: input_dir + "_converted")')
    parser.add_argument('--format', type=str, default='JPEG', 
                       choices=['JPEG', 'PNG', 'WEBP'],
                       help='Output format')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality (1-100)')
    parser.add_argument('--source_formats', nargs='+', 
                       default=['HEIC', 'HEIF'],
                       help='Source formats to convert from')
    
    args = parser.parse_args()
    
    # Convert images
    if 'HEIC' in args.source_formats or 'HEIF' in args.source_formats:
        converted_files = convert_heic_to_format(
            args.input_dir, 
            args.output_dir, 
            args.format, 
            args.quality
        )
    else:
        converted_files = convert_directory_formats(
            args.input_dir,
            args.output_dir,
            args.format,
            args.source_formats,
            args.quality
        )
    
    print(f"\nConversion complete! {len(converted_files)} images converted.")


if __name__ == '__main__':
    main()