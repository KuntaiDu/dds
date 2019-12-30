
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import subprocess
import cv2

import yaml

with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())


kernel_size = dds_env['kernel_size']
num_sqrt = dds_env['num_sqrt']

def compute_regions_size_tight(results, video_name, gt_images_path, hres, hqp, start_fid, end_fid):

    # import pdb; pdb.set_trace()

    gt_images_path = Path(gt_images_path)
    save_path = Path(video_name + f"-second-phase-size")
    save_path.mkdir(exist_ok = True)

    fid_set = {r.fid for r in results.regions}

    # increase coding efficiency by tightly laying out all images
    for fid in range(start_fid, end_fid):


        image = cv2.imread(str(gt_images_path / ('%010d.png' % fid)))
        image = cv2.resize(image, (round(image.shape[1] * hres), round(image.shape[0] * hres)))
        images = []


        for r in results.regions:
            if r.fid != fid:
                continue

            width = image.shape[1]
            height = image.shape[0]
            x0 = round(r.x * width)
            y0 = round(r.y * height)
            x0 = min(x0, width - kernel_size)
            y0 = min(y0, height - kernel_size)
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            images.append(image[y0: y0 + kernel_size, x0: x0 + kernel_size, :])

        assert len(images) == num_sqrt * num_sqrt

        # print(len(images))
        #if (fid == 51):
        #    import pdb; pdb.set_trace()

        concat_image = cv2.hconcat([cv2.vconcat([images[j * num_sqrt + i] for i in range(num_sqrt)]) for j in range(num_sqrt)])

        cv2.imwrite(str(save_path / ('%010d.png' % fid)), concat_image)

    # since the images are small, we can afford a computation expensive compression
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", f"{save_path / '%010d.png'}",
        "-start_number", f"{start_fid}",
        "-vcodec", "libx264",
        "-preset", "veryslow",
        "-tune", "animation",
        "-qp", f"{hqp}",
        "-vframes", f"{end_fid - start_fid}",
        "-me_method", "full",
        "-me_range", f"{kernel_size * num_sqrt}",
        f"{save_path/'temp.mp4'}"
    ])

    print((save_path/'temp.mp4').stat().st_size)

    return (save_path/'temp.mp4').stat().st_size