import os
import shutil
import logging
import cv2 as cv
from dds_utils import (Results, Region, calc_intersection_area,
                       compute_area_of_frame, calc_iou,
                       calc_area, merge_images_with_zeros, merge_images, extract_images_from_video)
from .object_detector import Detector
from .classifier import Classifier
from .segmenter import Segmenter
import matplotlib.pyplot as plt
import yaml

with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())


def TicTocGenerator():
    import time
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)



class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.logger.info("Server started")

    def perform_detection(self, images_direc, resolution, fnames=None,
                          images=None):

        if not hasattr(self, 'detector'):
            self.detector = Detector()
        final_results = Results()

        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            self.logger.debug(f"Running detection for {fname}")
            detection_results, multi_scan_results, offsets = self.detector.infer(image)
            # if fid == 5:
                # import pdb; pdb.set_trace()
            self.logger.info(f"Running inference on {len(fnames)} frames")
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                if (self.config.min_object_size and
                        w * h < self.config.min_object_size) or w*h ==0.:
                    # print("Continuing")
                    continue
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                final_results.append(r)
                frame_with_no_results = False

            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))
        return final_results, None, None

    def perform_application(self, images_direc, resolution, fnames=None, images=None, results = None):

        assert results is not None
        if not hasattr(self, 'model'):
            exec(f"self.model = {dds_env['model']}()")

        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                # import pdb; pdb.set_trace()
                image_path = os.path.join(images_direc, fname)
                image = plt.imread(image_path)

            self.logger.debug(f"Running {dds_env['application']} by {dds_env['model']} for {fname}")
            model_results = self.model.infer([image]).cpu().numpy()
            results[fid] = model_results

        return results


    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True, rpn_enlarge_ratio=0.):
        results = Results()
        accepted_results = Results()
        results_for_regions = Results()  # Results used for regions detection

        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "low-res"
                results.add_single_result(single_result,
                                          self.config.intersection_threshold)

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")

        for single_result in results.regions:
            results_for_regions.add_single_result(
                single_result, self.config.intersection_threshold)

        # self.logger.info(f"Returning {len(accepted_results)} "
        #                  f"confirmed results and "
        #                  f"{len(regions_to_query)} regions")

        # all the regions for query
        return results_for_regions

    def emulate_high_query(self, vid_name, low_images_direc, req_regions, high_images_direc = None):
        if dds_env['application'] == 'detection':
            return self.emulate_high_query_detection(vid_name, low_images_direc, req_regions)
        else:
            return self.emulate_high_query_application(vid_name, low_images_direc, req_regions, high_images_direc)

    def emulate_high_query_detection(self, vid_name, low_images_direc, req_regions):
        images_direc = vid_name + "-cropped"
        # Extract images from encoded video
        extract_images_from_video(images_direc, req_regions)

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the "
                              "second iteration was call anyway")
            return Results()

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f])

        # Make seperate directory and copy all images to that directory
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(images_direc, img), merged_images_direc)

        merged_images = merge_images(merged_images_direc, low_images_direc, req_regions)
        results, _, _= self.perform_detection(
            merged_images_direc, self.config.high_resolution, fnames,
            merged_images)

        results_with_detections_only = Results()
        for r in results.regions:
            # if r.label == "no obj" or r.w * r.h == 0.:
            #     continue
            # r.origin = "high-res"
            results_with_detections_only.add_single_result(
                r, self.config.intersection_threshold)
        # shutil.rmtree(merged_images_direc)

        return results_with_detections_only

    def emulate_high_query_application(self, vid_name, low_images_direc, req_regions, high_images_direc):
        images_direc = vid_name + "-cropped"
        # Extract images from encoded video
        extract_images_from_video(images_direc, req_regions)

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the "
                              "second iteration was call anyway")
            return None

        def fid_exists(fid):
            for r in req_regions.regions:
                if fid == r.fid:
                    return True
            return False

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f and fid_exists(int(f.split(".")[0]))])

        self.logger.info("Merging images...")
        # Make seperate directory and copy all images to that directory
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(high_images_direc, img), merged_images_direc)
        # print(low_images_direc)
        merged_images = merge_images(merged_images_direc, low_images_direc, req_regions)
        self.logger.info("Merge complete")

        # get the final result
        results = {}
        self.perform_application(
            merged_images_direc, self.config.high_resolution, fnames,
            None, results = results)

        return results
