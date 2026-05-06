import os
import shutil

def organize_cvae_files(base_dir):
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        print(f"Processing directory: {subdir}")
        
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f)) and f.endswith('.png')]
        
        for filename in files:
            parts = filename.split('_')
            if len(parts) >= 2:
                sample_id = parts[1]
                sample_folder = os.path.join(subdir_path, sample_id)
                
                if not os.path.exists(sample_folder):
                    os.makedirs(sample_folder)
                
                src_path = os.path.join(subdir_path, filename)
                dst_path = os.path.join(sample_folder, filename)
                
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                else:
                    print(f"Warning: {dst_path} already exists. Skipping.")
            else:
                print(f"Skipping file with unusual format: {filename}")

if __name__ == "__main__":
    base_generated_cvae = "/home/dalbom/dev/13_codes_corrosion_diffusion/generated_CVAE"
    organize_cvae_files(base_generated_cvae)
