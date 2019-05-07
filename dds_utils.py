class ServerConfig:
    def __init__(self, h_thres, l_thres, max_obj_size, tracker_length,
                 boundary):
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.tracker_length = tracker_length
        self.boundary = boundary


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution):
        self.fid = fid
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.label = label
        self.resolution = resolution

    def print_details(self):
        print("{}, "
              "{:0.3f}, {:0.3f}, "
              "{:0.3f}, {:0.3f}, {}".format(self.fid,
                                            self.x, self.y,
                                            self.w, self.h,
                                            self.resolution))

    def is_same(self, region_to_check, threshold=0.7):
        # If the fids are different return
        # then not the same
        if self.fid != region_to_check.fid:
            return False

        # If the intersection to union area
        # ratio is greater than the threshold
        # then the regions are the same
        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False


class Results:
    def __init__(self):
        self.regions = []

    def results_len(self):
        return len(self.regions)

    def is_dup(self, result_to_add, threshold=0.7):
        for existing_result in self.regions:
            if existing_result.is_same(result_to_add):
                return True
        return False

    def combine_results(self, additional_results, intersection_threshold=0.7):
        for result in additional_results.regions:
            if not self.is_dup(result, intersection_threshold):
                self.regions.append(result)
        # Sort final results
        self.regions.sort(key=lambda x: x.fid)

    def add_single_result(self, result_to_add, threshold=0.7):
        temp_results = Results()
        temp_results.regions = [result_to_add]
        self.combine_results(temp_results, threshold)

    def fill_gaps(self, number_of_frames):
        results_to_add = Results()
        max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        for i in range(number_of_frames):
            if i not in fids_in_results:
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     "no obj", 0.1,
                                                     max_resolution))
        self.combine_results(results_to_add)


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
        single_result = Region(fid, x, y, w, h, conf, label, resolution)

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(single_result)

    return results_dict


def read_results_dict(fname, fmat="csv"):
    # TODO: Need to implement a CSV function
    if fmat == "txt":
        return read_results_txt_dict(fname)


def write_results_txt(results, fname):
    results_file = open(fname, "w")
    for result in results.regions:
        # prepare the string to write
        str_to_write = "{},{},{},{},{},{},{},{}\n".format(result.fid, result.x,
                                                          result.y, result.w,
                                                          result.h,
                                                          result.label,
                                                          result.conf,
                                                          result.resolution)
        results_file.write(str_to_write)
    results_file.close()


def write_results(results, fname, fmat="csv"):
    if fmat == "txt":
        write_results_txt(results, fname)


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
