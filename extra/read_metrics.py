import os
import csv
from natsort import natsorted
import argparse

root_dir = ""
expname = ""

def store_csv(args):

    scene_values = {}

    for scene_name in natsorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene_name)
        # scene_path = os.path.join(scene_path, expname)

        if os.path.isdir(scene_path):
            # Create a dictionary to store metric values for the current scene
            metric_values = {"PSNR": None, "SSIM": None, "LPIPS": None}
            
            # Read the mean.txt file for the current scene
            if args.masked:
                mean_file_path = os.path.join(scene_path, "imgs_test_all/masked_mean.txt")
            else:
                mean_file_path = os.path.join(scene_path, "imgs_test_all/mean.txt")
            if os.path.isfile(mean_file_path):
                with open(mean_file_path, "r") as mean_file:
                    lines = mean_file.readlines()
                    lines = [line.strip() for line in lines]
                    for i, (key, value) in enumerate(metric_values.items()):
                        metric_values[key] = float(lines[i])
            # Add the metric values for the current scene to the scene_values dictionary
            scene_values[scene_name] = metric_values

    # scene_values = dict(sorted(scene_values.items()))

    # Create a CSV file to store the results
    if args.masked:
        output_csv_path = f"masked_{args.expname}.csv"
    else:
        output_csv_path = f"{args.expname}.csv"
    output_csv_path = os.path.join(root_dir, output_csv_path)
    
   

    # Write the CSV file
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header row
        writer.writerow(["Metric"] + list(scene_values.keys()))
        
        # Write metric values for each scene
        for metric_name in ["LPIPS", "SSIM", "PSNR"]:
            metric_data = [scene_values[scene_name][metric_name] for scene_name in scene_values.keys()]
            if args.weight is not None:
                weighted_metric_data = [x * y for x, y in zip(metric_data, args.weight)]
                avg = sum(weighted_metric_data) / sum(args.weight)
            else:
                avg = sum(metric_data) / len(metric_data)
            metric_data.append(avg)
            metric_data = [round(x, 3) for x in metric_data]
            writer.writerow([metric_name] + metric_data)
            

    print(f"CSV file '{output_csv_path}' has been created.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str)
    parser.add_argument("--masked", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--weight", type=int, default=None, nargs='+') 
    args = parser.parse_args()
    expname = args.expname
    root_dir = f"{args.log_dir}"
    store_csv(args)
    