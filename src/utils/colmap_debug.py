"""
COLMAP debugging and diagnosis utilities.
"""

import sqlite3
import numpy as np
from pathlib import Path


def analyze_colmap_database(database_path):
    """Analyze COLMAP database to diagnose issues."""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    print("=== COLMAP Database Analysis ===")
    
    # Check number of images
    cursor.execute("SELECT COUNT(*) FROM images")
    num_images = cursor.fetchone()[0]
    print(f"Number of images: {num_images}")
    
    # Check database schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Database tables: {tables}")
    
    # Check keypoints table structure
    if 'keypoints' in tables:
        cursor.execute("PRAGMA table_info(keypoints)")
        keypoint_columns = [row[1] for row in cursor.fetchall()]
        print(f"Keypoints table columns: {keypoint_columns}")
        
        # Count keypoints per image (COLMAP stores keypoints as BLOB)
        cursor.execute("""
            SELECT images.name, LENGTH(keypoints.data) / 6 as num_keypoints
            FROM images 
            LEFT JOIN keypoints ON images.image_id = keypoints.image_id
            ORDER BY num_keypoints DESC
        """)
        
        keypoint_stats = cursor.fetchall()
        if keypoint_stats:
            print(f"\nKeypoints per image:")
            for name, count in keypoint_stats[:10]:  # Show top 10
                print(f"  {name}: {count if count else 0} keypoints")
            
            counts = [count if count else 0 for _, count in keypoint_stats]
            if counts and max(counts) > 0:
                print(f"\nKeypoint statistics:")
                print(f"  Mean: {np.mean(counts):.1f}")
                print(f"  Min: {min(counts)}")
                print(f"  Max: {max(counts)}")
            else:
                print("\n❌ No keypoints found in any images!")
        else:
            print("\n❌ No keypoint data found!")
    else:
        print("❌ No keypoints table found!")
    
    # Check matches
    if 'matches' in tables:
        cursor.execute("SELECT COUNT(*) FROM matches")
        num_matches = cursor.fetchone()[0]
        print(f"\nNumber of image pairs with matches: {num_matches}")
        
        if num_matches > 0:
            cursor.execute("""
                SELECT pair_id, LENGTH(data) / 8 as num_pair_matches
                FROM matches 
                ORDER BY LENGTH(data) DESC 
                LIMIT 10
            """)
            match_data = cursor.fetchall()
            
            print(f"\nTop matches by count:")
            for pair_id, num_pair_matches in match_data:
                print(f"  Pair {pair_id}: {num_pair_matches} matches")
        else:
            print("❌ No matches found between any image pairs!")
    else:
        print("❌ No matches table found!")
    
    # Check two_view_geometries for verified matches
    if 'two_view_geometries' in tables:
        cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
        num_geometries = cursor.fetchone()[0]
        print(f"\nNumber of verified geometric relationships: {num_geometries}")
        
        if num_geometries > 0:
            # Check what columns are available in two_view_geometries
            cursor.execute("PRAGMA table_info(two_view_geometries)")
            geom_columns = [row[1] for row in cursor.fetchall()]
            print(f"Two-view geometry columns: {geom_columns}")
            
            cursor.execute("""
                SELECT pair_id, config 
                FROM two_view_geometries 
                LIMIT 10
            """)
            geom_data = cursor.fetchall()
            
            print(f"\nGeometric relationships found:")
            for pair_id, config in geom_data[:5]:
                print(f"  Pair {pair_id}: config {config}")
    
    conn.close()
    
    # Provide diagnosis
    print(f"\n=== Diagnosis ===")
    if num_images < 3:
        print("❌ Too few images. Need at least 3 images for reconstruction.")
        return False
    
    if 'keypoints' in tables and counts and max(counts) < 100:
        print("⚠️  Some images have very few keypoints. Check image quality.")
        print("   - Images may be blurry, low contrast, or lack distinctive features")
        return False
    
    if 'matches' not in tables or num_matches == 0:
        print("❌ No matches found between images. Possible issues:")
        print("   - Images don't overlap enough (need 60%+ overlap)")
        print("   - Images are too different (lighting, angle, scale)")
        print("   - Images lack distinctive features")
        return False
    
    if 'matches' in tables and num_matches < num_images // 2:
        print("⚠️  Very few image pairs matched. Check image sequence:")
        print("   - Ensure consecutive images have significant overlap")
        return False
    
    if 'two_view_geometries' in tables and num_geometries == 0:
        print("❌ No geometric relationships verified. This means:")
        print("   - Matches exist but don't form valid 3D geometry")
        print("   - Images may be taken from too similar viewpoints")
        print("   - Camera motion may be purely rotational (no translation)")
        return False
    
    print("✅ Database looks healthy.")
    return True


def suggest_colmap_fixes(images_dir, database_path):
    """Suggest fixes for COLMAP issues."""
    analyze_colmap_database(database_path)
    
    print(f"\n=== Suggested Fixes ===")
    print("1. **Reduce image resolution further:**")
    print(f"   python main.py prepare --images_dir {images_dir} --resize_factor 8")
    
    print("\n2. **Try different camera model:**")
    print(f"   python main.py colmap --images_dir {images_dir} --camera_model PINHOLE")
    
    print("\n3. **Check image overlap manually:**")
    print("   - Ensure consecutive images share 60%+ of the scene")
    print("   - Look for distinctive features (corners, textures)")
    print("   - Avoid blurry or low-contrast images")
    
    print("\n4. **Use subset of best images:**")
    print("   - Select 20-30 images with good overlap")
    print("   - Remove blurry or poorly lit images")
    
    print("\n5. **Try sequential matching instead of exhaustive:**")
    print("   - Good for video-like sequences")


def main():
    """Debug COLMAP database."""
    import argparse
    parser = argparse.ArgumentParser(description='Debug COLMAP database')
    parser.add_argument('database_path', help='Path to COLMAP database.db')
    parser.add_argument('--images_dir', help='Images directory for suggestions')
    args = parser.parse_args()
    
    analyze_colmap_database(args.database_path)
    
    if args.images_dir:
        suggest_colmap_fixes(args.images_dir, args.database_path)


if __name__ == '__main__':
    main()