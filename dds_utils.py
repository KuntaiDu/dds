import math
import re
import os
import csv
import shutil
import subprocess
import numpy as np
import cv2 as cv


class ServerConfig:
    def __init__(self, low_res, high_res, low_qp, high_qp, bsize,
                 h_thres, l_thres, max_obj_size, min_obj_size,
                 tracker_length, boundary, intersection_threshold,
                 tracking_threshold, suppression_threshold, simulation):
        self.low_resolution = low_res
        self.high_resolution = high_res
        self.low_qp = low_qp
        self.high_qp = high_qp
        self.batch_size = bsize
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.min_object_size = min_obj_size
        self.tracker_length = tracker_length
        self.boundary = boundary
        self.intersection_threshold = intersection_threshold
        self.simulation = simulation
        self.tracking_threshold = tracking_threshold
        self.suppression_threshold = suppression_threshold


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution,
                 origin="generic"):
        self.fid = fid
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.label = label
        self.resolution = resolution
        self.origin = origin

    def __str__(self):
        string_rep = (f"{self.fid}, {self.x:0.3f}, {self.y:0.3f}, "
                      f"{self.w:0.3f}, {self.h:0.3f}, {self.conf:0.3f}, "
                      f"{self.label}, {self.origin}")
        return string_rep

    def is_same(self, region_to_check, threshold=0.5):
        # If the fids or labels are different
        # then not the same
        if (self.fid != region_to_check.fid or
                ((self.label != "-1" and region_to_check.label != "-1") and
                 (self.label != region_to_check.label))):
            return False

        # If the intersection to union area
        # ratio is greater than the threshold
        # then the regions are the same
        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False

    def copy(self):
        return Region(self.fid, self.x, self.y, self.w, self.h, self.conf,
                      self.label, self.resolution, self.origin)


class Results:
    def __init__(self):
        self.regions = []

    def __len__(self):
        return len(self.regions)

    def results_high_len(self, threshold):
        count = 0
        for r in self.regions:
            if r.conf > threshold:
                count += 1
        return count

    def is_dup(self, result_to_add, threshold=0.5):
        # return the regions with IOU greater than threshold
        # and maximum confidence
        max_conf = -1
        max_conf_result = None
        for existing_result in self.regions:
            if existing_result.is_same(result_to_add, threshold):
                if existing_result.conf > max_conf:
                    max_conf = existing_result.conf
                    max_conf_result = existing_result
        return max_conf_result

    def combine_results(self, additional_results, threshold=0.5):
        for result_to_add in additional_results.regions:
            self.add_single_result(result_to_add, threshold)

    def add_single_result(self, region_to_add, threshold=0.5):
        if threshold == 1:
            self.append(region_to_add)
            return
        dup_region = self.is_dup(region_to_add, threshold)
        if (not dup_region or
                ("tracking" in region_to_add.origin and
                 "tracking" in dup_region.origin)):
            self.regions.append(region_to_add)
        else:
            final_object = None
            if dup_region.origin == region_to_add.origin:
                final_object = max([region_to_add, dup_region],
                                   key=lambda r: r.conf)
            elif ("low" in dup_region.origin and
                  "high" in region_to_add.origin):
                final_object = region_to_add
            elif ("high" in dup_region.origin and
                  "low" in region_to_add.origin):
                final_object = dup_region
            dup_region.x = final_object.x
            dup_region.y = final_object.y
            dup_region.w = final_object.w
            dup_region.h = final_object.h
            dup_region.conf = final_object.conf
            dup_region.origin = final_object.origin

    def suppress(self, threshold=0.5):
        new_regions_list = []
        while len(self.regions) > 0:
            max_conf_obj = max(self.regions, key=lambda e: e.conf)
            new_regions_list.append(max_conf_obj)
            self.regions.remove(max_conf_obj)
            objs_to_remove = []
            for r in self.regions:
                if r.fid != max_conf_obj.fid:
                    continue
                if calc_iou(r, max_conf_obj) > threshold:
                    objs_to_remove.append(r)
            for r in objs_to_remove:
                self.regions.remove(r)
        self.regions = new_regions_list
        self.regions.sort(key=lambda e: e.fid)

    def append(self, region_to_add):
        self.regions.append(region_to_add)

    def remove(self, region_to_remove):
        self.regions.remove(region_to_remove)

    def fill_gaps(self, number_of_frames):
        if len(self.regions) == 0:
            return
        results_to_add = Results()
        max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        for i in range(number_of_frames):
            if i not in fids_in_results:
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     0.1, "no obj",
                                                     max_resolution))
        self.combine_results(results_to_add)
        self.regions.sort(key=lambda r: r.fid)

    def write_results_txt(self, fname):
        results_file = open(fname, "w")
        for region in self.regions:
            # prepare the string to write
            str_to_write = (f"{region.fid},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf},"
                            f"{region.resolution},{region.origin}\n")
            results_file.write(str_to_write)
        results_file.close()

    def write_results_csv(self, fname):
        results_files = open(fname, "w")
        csv_writer = csv.writer(results_files)
        for region in self.regions:
            row = [region.fid, region.x, region.y,
                   region.w, region.h,
                   region.label, region.conf,
                   region.resolution, region.origin]
            csv_writer.writerow(row)
        results_files.close()

    def write(self, fname):
        if re.match(r"\w+[.]csv\Z", fname):
            self.write_results_csv(fname)
        else:
            self.write_results_txt(fname)


def read_results_txt_dict(fname):
    """Return a dictionary with fid mapped to
       and array that contains all SingleResult objects
       from that particular frame"""
    results_dict = {}

    with open(fname, "r") as f:
        lines = f.readlines()
        f.close()

    for line in lines:
        line = line.split(",")
        fid = int(line[0])
        x, y, w, h = [float(e) for e in line[1:5]]
        conf = float(line[6])
        label = line[5]
        resolution = float(line[7])
        origin = "generic"
        if len(line) > 8:
            origin = line[8].strip()
        single_result = Region(fid, x, y, w, h, conf, label,
                               resolution, origin.rstrip())

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(single_result)

    return results_dict


def read_results_csv_dict(fname):
    """Return a dictionary with fid mapped to an array
    that contains all Regions objects"""
    results_dict = {}

    rows = []
    with open(fname) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        fid = int(row[0])
        x, y, w, h = [float(e) for e in row[1:5]]
        conf = float(row[6])
        label = row[5]
        resolution = float(row[7])
        origin = float(row[8])

        region = Region(fid, x, y, w, h, conf, label, resolution, origin)

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(region)

    return results_dict


def read_results_dict(fname):
    # TODO: Need to implement a CSV function
    if re.match(r"\w+[.]csv\Z", fname):
        return read_results_csv_dict(fname)
    else:
        return read_results_txt_dict(fname)


def calc_intersection_area(a, b):
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.h, b.y + b.h)
    ri = min(a.x + a.w, b.x + b.w)

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a.w)
    h = max(0, a.h)

    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area


def get_interval_area(width, all_yes):
    area = 0
    for y1, y2 in all_yes:
        area += (y2 - y1) * width
    return area


def insert_range_y(all_yes, y1, y2):
    ranges_length = len(all_yes)
    idx = 0
    while idx < ranges_length:
        if not (y1 > all_yes[idx][1] or all_yes[idx][0] > y2):
            # Overlapping
            y1 = min(y1, all_yes[idx][0])
            y2 = max(y2, all_yes[idx][1])
            del all_yes[idx]
            ranges_length = len(all_yes)
        else:
            idx += 1

    all_yes.append((y1, y2))


def get_y_ranges(regions, j, x1, x2):
    all_yes = []
    while j < len(regions):
        if (x1 < (regions[j].x + regions[j].w) and
                x2 > regions[j].x):
            y1 = regions[j].y
            y2 = regions[j].y + regions[j].h
            insert_range_y(all_yes, y1, y2)
        j += 1
    return all_yes


def compute_area_of_frame(regions):
    regions.sort(key=lambda r: r.x + r.w)

    all_xes = []
    for r in regions:
        all_xes.append(r.x)
        all_xes.append(r.x + r.w)
    all_xes.sort()

    area = 0
    j = 0
    for i in range(len(all_xes) - 1):
        x1 = all_xes[i]
        x2 = all_xes[i + 1]

        if x1 < x2:
            while (regions[j].x + regions[j].w) < x1:
                j += 1
            all_yes = get_y_ranges(regions, j, x1, x2)
            area += get_interval_area(x2 - x1, all_yes)

    return area


def compute_area_of_regions(results):
    if len(results.regions) == 0:
        return 0

    min_frame = min([r.fid for r in results.regions])
    max_frame = max([r.fid for r in results.regions])

    total_area = 0
    for fid in range(min_frame, max_frame + 1):
        regions_for_frame = [r for r in results.regions if r.fid == fid]
        total_area += compute_area_of_frame(regions_for_frame)

    return total_area


def compress_and_get_size(images_path, start_id, end_id, qp,
                          enforce_iframes=False, resolution=None):
    number_of_frames = end_id - start_id
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    if resolution and enforce_iframes:
        scale = f"scale=trunc(iw*{resolution}/2)*2:trunc(ih*{resolution}/2)*2"
        if not qp:
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%010d.png",
                                              "-vcodec", "libx264", "-g", "15",
                                              "-keyint_min", "15",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
        else:
            # import pdb; pdb.set_trace()
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%010d.png",
                                              "-vcodec", "libx264", "-g", "15",
                                              "-keyint_min", "15",
                                              "-qp", f"{qp}",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
    else:
        encoding_result = subprocess.run(["ffmpeg", "-y",
                                          "-start_number", str(start_id),
                                          "-i", f"{images_path}/%010d.png",
                                          "-loglevel", "error",
                                          "-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p", "-crf", "23",
                                          encoded_vid_path],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)

    size = 0
    if encoding_result.returncode != 0:
        # Encoding failed
        print("ENCODING FAILED")
        print(encoding_result.stdout)
        print(encoding_result.stderr)
        exit()
    else:
        size = os.path.getsize(encoded_vid_path)

    return size


def extract_images_from_video(images_path, req_regions):
    if not os.path.isdir(images_path):
        return

    for fname in os.listdir(images_path):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    extacted_images_path = os.path.join(images_path, "%010d.png")
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      # "-vcodec", "mjpeg",
                                      "-pix_fmt", "yuvj420p",
                                      "-g", "8", "-q:v", "2",
                                      "-vsync", "0", "-start_number", "0",
                                      extacted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()

    fnames = sorted(
        [os.path.join(images_path, name)
         for name in os.listdir(images_path) if "png" in name])
    fids = sorted(list(set([r.fid for r in req_regions.regions])))
    fids_mapping = zip(fids, fnames)
    for fname in fnames:
        # Rename temporarily
        os.rename(fname, f"{fname}_temp")

    for fid, fname in fids_mapping:
        os.rename(os.path.join(f"{fname}_temp"),
                  os.path.join(images_path, f"{str(fid).zfill(10)}.png"))

def squeeze_regions(results, vid_name, images_direc, resolution=None):
    cached_image = None
    cropped_images = {}

    for region in results.regions:
        if not (cached_image and
                cached_image[0] == region.fid):
            image_path = os.path.join(images_direc,
                                      f"{str(region.fid).zfill(10)}.png")
            cached_image = (region.fid, cv.imread(image_path))

        # Just move the complete image
        if region.x == 0 and region.y == 0 and region.w == 1 and region.h == 1:
            cropped_images[region.fid] = cached_image[1]
            continue

        width = cached_image[1].shape[1]
        height = cached_image[1].shape[0]
        x0 = int(region.x * width)
        y0 = int(region.y * height)
        x1 = int((region.w * width) + x0 - 1)
        y1 = int((region.h * height) + y0 - 1)

        if region.fid not in cropped_images:
            cropped_images[region.fid] = np.zeros_like(cached_image[1])

        cropped_image = cropped_images[region.fid]
        cropped_image[y0:y1, x0:x1, :] = cached_image[1][y0:y1, x0:x1, :]
        cropped_images[region.fid] = cropped_image

    os.makedirs(vid_name, exist_ok=True)
    frames_count = len(cropped_images)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])
    crops = []
    max_w = 0.
    max_h = 0.
    for idx, (_, frame) in enumerate(frames):
        if resolution:
            w = int(frame.shape[1] * resolution)
            h = int(frame.shape[0] * resolution)
            im_to_write = cv.resize(frame, (w, h), fx=0, fy=0,
                                    interpolation=cv.INTER_CUBIC)
            frame = im_to_write
            mask = frame == 0
            all_black = mask.sum(axis=2) == 3
            rows = np.flatnonzero((~all_black).sum(axis=1))
            cols = np.flatnonzero((~all_black).sum(axis=0))
            frame = frame[rows,:,:]
            frame = frame[:,cols,:]
            # cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(10)}.png"),
            #         frame, )
            # get the largest w,h in a batch
            if len(cols) > max_w:
                max_w = len(cols)
            if len(rows) > max_h:
                max_h = len(rows)
            crops.append(frame)
    for idx, crop in enumerate(crops):
        black_canvas = np.zeros((max_h,max_w,3), np.uint8)
        black_canvas[0:crop.shape[0],0:crop.shape[1],:] = crop
        cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(10)}.png"),
                    black_canvas, [cv.IMWRITE_PNG_COMPRESSION, 0])

    # import pdb; pdb.set_trace()
    return frames_count

def crop_images(results, vid_name, images_direc, resolution=None):
    cached_image = None
    cropped_images = {}

    for region in results.regions:
        if not (cached_image and
                cached_image[0] == region.fid):
            image_path = os.path.join(images_direc,
                                      f"{str(region.fid).zfill(10)}.png")
            cached_image = (region.fid, cv.imread(image_path))

        # Just move the complete image
        if region.x == 0 and region.y == 0 and region.w == 1 and region.h == 1:
            cropped_images[region.fid] = cached_image[1]
            continue

        width = cached_image[1].shape[1]
        height = cached_image[1].shape[0]
        x0 = int(region.x * width)
        y0 = int(region.y * height)
        x1 = int((region.w * width) + x0 - 1)
        y1 = int((region.h * height) + y0 - 1)

        if region.fid not in cropped_images:
            cropped_images[region.fid] = np.zeros_like(cached_image[1])

        cropped_image = cropped_images[region.fid]
        cropped_image[y0:y1, x0:x1, :] = cached_image[1][y0:y1, x0:x1, :]
        cropped_images[region.fid] = cropped_image

    os.makedirs(vid_name, exist_ok=True)
    frames_count = len(cropped_images)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])
    for idx, (_, frame) in enumerate(frames):
        if resolution:
            w = int(frame.shape[1] * resolution)
            h = int(frame.shape[0] * resolution)
            im_to_write = cv.resize(frame, (w, h), fx=0, fy=0,
                                    interpolation=cv.INTER_CUBIC)
            frame = im_to_write
        cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(10)}.png"),
                   frame,[cv.IMWRITE_PNG_COMPRESSION, 0] )

    return frames_count

def crop_images_cubic(results, vid_name, images_direc, resolution=None):
    cached_image = None
    cropped_images = {}
    # get the max size
    max_w = 0.
    max_h = 0.
    for region in results.regions:
        max_w = max(region.w, max_w)
        max_h = max(region.h, max_h)
    frames_count = 0
    for idx, region in enumerate(results.regions):
        frames_count += 1
        if not (cached_image and
                cached_image[0] == region.fid):
            image_path = os.path.join(images_direc,
                                      f"{str(region.fid).zfill(10)}.png")
            cached_image = (region.fid, cv.imread(image_path))
            cropped_images[region.fid] = []


        # # Just move the complete image
        # if region.x == 0 and region.y == 0 and region.w == 1 and region.h == 1:
        #     cropped_images[region.fid] = cached_image[1]
        #     continue

        width = cached_image[1].shape[1]
        height = cached_image[1].shape[0]
        x0 = int(region.x * width)
        y0 = int(region.y * height)
        x1 = int((region.w * width) + x0 - 1)
        y1 = int((region.h * height) + y0 - 1)

        # if region.fid not in cropped_images:
        cropped_images[region.fid].append((np.zeros((int(max_h * height), int(max_w * width), 3), np.uint8), (x0,y0,x1,y1)))
        cropped_image = cropped_images[region.fid][-1][0]
        cropped_image[0:y1-y0, 0:x1-x0, :] = cached_image[1][y0:y1, x0:x1, :]
        # import pdb; pdb.set_trace()
        crop_images_list = list(cropped_images[region.fid][-1])
        crop_images_list[0] = cropped_image
        cropped_images[region.fid][-1] = tuple(crop_images_list)

    os.makedirs(vid_name, exist_ok=True)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])
    save_idx = 0
    bbox_offset = {}
    for idx, (_,frame) in enumerate(frames):
        bbox_offset[idx] = []
        for ridx, (region, loc) in enumerate(frame):
            if resolution:
                bbox_offset[idx] = loc
                w = int(region.shape[1] * resolution)
                h = int(region.shape[0] * resolution)
                im_to_write = cv.resize(region, (w, h), fx=0, fy=0,
                                        interpolation=cv.INTER_CUBIC)
                frame = im_to_write
            cv.imwrite(os.path.join(vid_name, f"{str(save_idx).zfill(10)}.png"),
                       frame,[cv.IMWRITE_PNG_COMPRESSION, 0] )
            save_idx = save_idx + 1

    return frames_count, bbox_offset

def padding_images(cropped_images_direc, req_regions):
    images = {}
    for fname in os.listdir(cropped_images_direc):
        if "png" not in fname:
            continue
        fid = int(fname.split(".")[0])

        # Read high resolution image
        high_image = cv.imread(os.path.join(cropped_images_direc, fname))
        width = high_image.shape[1]
        height = high_image.shape[0]

        # Read low resolution image
        low_image = cv.imread(os.path.join(low_images_direc, fname))
        # Enlarge low resolution image
        enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
                                   interpolation=cv.INTER_CUBIC)
        # Put regions in place
        for r in req_regions.regions:
            if fid != r.fid:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0 - 1)
            y1 = int((r.h * height) + y0 - 1)
            enlarged_image[y0:y1, x0:x1, :] = high_image[y0:y1, x0:x1, :]
        cv.imwrite(os.path.join(cropped_images_direc, fname), enlarged_image,[cv.IMWRITE_PNG_COMPRESSION, 0] )
        images[fid] = enlarged_image
    return images

def merge_images_with_zeros(cropped_images_direc, req_regions):
    images = {}
    for fname in os.listdir(cropped_images_direc):
        if "png" not in fname:
            continue
        fid = int(fname.split(".")[0])

        # Read high resolution image
        high_image = cv.imread(os.path.join(cropped_images_direc, fname))
        width = high_image.shape[1]
        height = high_image.shape[0]

        # Read low resolution image
        # low_image = cv.imread(os.path.join(low_images_direc, fname))
        # Enlarge low resolution image
        # enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
        #                            interpolation=cv.INTER_CUBIC)
        # blank_image = np.zeros((height,width,3), np.uint8)
        # Put regions in place
        for r in req_regions.regions:
            if fid != r.fid:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0 - 1)
            y1 = int((r.h * height) + y0 - 1)

            blank_image[y0:y1, x0:x1, :] = high_image[y0:y1, x0:x1, :]
        cv.imwrite(os.path.join(cropped_images_direc, fname), blank_image,[cv.IMWRITE_PNG_COMPRESSION, 0] )
        images[fid] = blank_image
    return images

def merge_images(cropped_images_direc, low_images_direc, req_regions):
    images = {}
    for fname in os.listdir(cropped_images_direc):
        if "png" not in fname:
            continue
        fid = int(fname.split(".")[0])

        # Read high resolution image
        high_image = cv.imread(os.path.join(cropped_images_direc, fname))
        width = high_image.shape[1]
        height = high_image.shape[0]

        # Read low resolution image
        low_image = cv.imread(os.path.join(low_images_direc, fname))
        # Enlarge low resolution image
        enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
                                   interpolation=cv.INTER_CUBIC)
        # Put regions in place
        for r in req_regions.regions:
            if fid != r.fid:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0 - 1)
            y1 = int((r.h * height) + y0 - 1)

            enlarged_image[y0:y1, x0:x1, :] = high_image[y0:y1, x0:x1, :]
        cv.imwrite(os.path.join(cropped_images_direc, fname), enlarged_image,[cv.IMWRITE_PNG_COMPRESSION, 0] )
        images[fid] = enlarged_image
    return images


def compute_regions_size(results, vid_name, images_direc, resolution, qp,
                         enforce_iframes, estimate_banwidth=True, mode = 1):
    if estimate_banwidth:
        # If not simulation then compress and encode images
        # and get size
        if mode == 1:
            vid_name = f"{vid_name}-cropped"
            frames_count = crop_images(results, vid_name, images_direc,
                                       resolution)
            # frames_count = crop_images(results, vid_name, images_direc,
            #                            resolution)
            size = compress_and_get_size(vid_name, 0, frames_count, qp=qp,
                                         enforce_iframes=enforce_iframes,
                                         resolution=1)
            pixel_size = compute_area_of_regions(results)
            return size, pixel_size

        elif mode == 2:
            vid_name = f"{vid_name}-cropped"
            frames_count, bbox_offset = crop_images_cubic(results, vid_name, images_direc,
                                       resolution)
            # print(frames_count)
            size = compress_and_get_size(vid_name, 0, frames_count, qp=qp,
                                         enforce_iframes=enforce_iframes,
                                         resolution=1)
            return size, bbox_offset
        elif mode == 3:
            vid_name = f"{vid_name}-cropped"
            frames_count = squeeze_regions(results, vid_name, images_direc,
                                       resolution)
            # print(frames_count)
            size = compress_and_get_size(vid_name, 0, frames_count, qp=qp,
                                         enforce_iframes=enforce_iframes,
                                         resolution=1)
            return size
    else:
        size = compute_area_of_regions(results)

        return size


def cleanup(vid_name, debug_mode=False, start_id=None, end_id=None):
    if not os.path.isdir(vid_name + "-cropped"):
        return

    if not debug_mode:
        shutil.rmtree(vid_name + "-cropped")
    else:
        if start_id is None or end_id is None:
            print("Need start_fid and end_fid for debugging mode")
            exit()
        os.makedirs("debugging", exist_ok=True)
        leaf_direc = vid_name.split("/")[-1] + "-cropped"
        shutil.move(vid_name + "-cropped", "debugging")
        shutil.move(os.path.join("debugging", leaf_direc),
                    os.path.join("debugging",
                                 f"{leaf_direc}-{start_id}-{end_id}"),
                    copy_function=os.rename)


def get_size_from_mpeg_results(results_log_path, images_path, resolution):
    with open(results_log_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line.rstrip().lstrip() != ""]

    num_frames = len([x for x in os.listdir(images_path) if "png" in x])

    bandwidth = 0
    for idx, line in enumerate(lines):
        if f"RES {resolution}" in line:
            bandwidth = float(lines[idx + 2])
            break
    size = bandwidth * 1024.0 * (num_frames / 10.0)
    return size


def evaluate(results, gt_dict, high_threshold, iou_threshold=0.4):
    gt_results = Results()
    for k, v in gt_dict.items():
        for single_result in v:
            if single_result.conf < 0.3 or (single_result.label != 'vehicle') or single_result.w * single_result.h > 0.01:
            # if single_result.conf < 0.3:
                # print(1)
                # print(single_result.label)
                continue
            single_result.conf = high_threshold
            gt_results.add_single_result(single_result)
    # Save regions count because the regions that match
    # will be removed from the gt_regions to ensure better
    # search speed
    gt_regions_count = len(gt_results)

    deduplicated_results = Results()
    for r in results.regions:
        deduplicated_results.add_single_result(r)
    results = deduplicated_results

    fp = 0.0
    tp = 0.0
    fn = 0.0
    for a in results.regions:
        # Make sure that the region has a high confidence
         # or (a.label != 'vehicle' and a.label != 'no obj')
        if a.conf < high_threshold or (a.label != 'vehicle') or (a.w * a.h > 0.01):
        # if a.conf < high_threshold:
            continue

        # Find match in gt_results
        max_iou = -1
        matching_region = None
        for b in gt_results.regions:
            if a.is_same(b, iou_threshold):
                iou = calc_iou(a, b)
                if iou > max_iou:
                    max_iou = iou
                    matching_region = b
                break

        if matching_region:
            # Remove region from ground truth if
            # it has already matched with a region in results
            gt_results.remove(matching_region)
            tp += 1.0
        else:
            fp += 1.0

    fn = gt_regions_count - tp

    if (fp + tp) == 0 or (fn + tp) == 0:
        return 0, (tp, fp, fn)

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)

    if (precision + recall) == 0:
        return 0, (tp, fp, fn)

    f1 = 2.0 * precision * recall / (precision + recall)

    if math.isnan(f1):
        f1 = 0.0

    return f1, (tp, fp, fn)


def write_stats_txt(fname, vid_name, config, f1, stats,
                    bw, frames_count, mode):
    header = ("video-name,low-resolution,high-resolution,low_qp,high_qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,mode")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{mode}")

    if not os.path.isfile(fname):
        str_to_write = f"{header}\n{stats}\n"
    else:
        str_to_write = f"{stats}\n"

    with open(fname, "a") as f:
        f.write(str_to_write)


def write_stats_csv(fname, vid_name, config, f1, stats, bw,
                    frames_count, mode):
    header = ("video-name,low-resolution,high-resolution,low-qp,high-qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,mode").split(",")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{mode}").split(",")

    results_files = open(fname, "a")
    csv_writer = csv.writer(results_files)
    if not os.path.isfile(fname):
        # If file does not exist write the header row
        csv_writer.writerow(header)
    csv_writer.writerow(stats)
    results_files.close()


def write_stats(fname, vid_name, config, f1, stats, bw,
                frames_count, mode):
    if re.match(r"\w+[.]csv\Z", fname):
        write_stats_csv(fname, vid_name, config, f1, stats, bw,
                        frames_count, mode)
    else:
        write_stats_txt(fname, vid_name, config, f1, stats, bw,
                        frames_count, mode)


def visualize_regions(results, images_direc,
                      low_conf=0.0, high_conf=1.0,
                      label="debugging"):
    idx = 0
    fids = sorted(list(set([r.fid for r in results.regions])))
    while idx < len(fids):
        image_np = cv.imread(
            os.path.join(images_direc, f"{str(fids[idx]).zfill(10)}.png"))
        width = image_np.shape[1]
        height = image_np.shape[0]
        regions = [r for r in results.regions if r.fid == fids[idx]]
        for r in regions:
            if r.conf < low_conf or r.conf > high_conf:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int(r.w * width + x0)
            y1 = int(r.h * height + y0)
            cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv.putText(image_np, f"{fids[idx]}", (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv.imshow(label, image_np)
        key = cv.waitKey()
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("k"):
            idx -= 2

        idx += 1
    cv.destroyAllWindows()


def visualize_single_regions(region, images_direc, label="debugging"):
    image_path = os.path.join(images_direc, f"{str(region.fid).zfill(10)}.png")
    image_np = cv.imread(image_path)
    width = image_np.shape[1]
    height = image_np.shape[0]

    x0 = int(region.x * width)
    y0 = int(region.y * height)
    x1 = int((region.w * width) + x0)
    y1 = int((region.h * height) + y0)

    cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv.putText(image_np, f"{region.fid}, {region.label}, {region.conf:0.2f}, "
               f"{region.w * region.h}",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv.imshow(label, image_np)
    cv.waitKey()
    cv.destroyAllWindows()
