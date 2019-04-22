class ServerConfig:
    def __init__(self, h_thres, l_thres, max_obj_size, tracker_length):
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.tracker_length = tracker_length


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution):
        self.fid = fid
        self.x = x
        self.y = y
        self.w = w
        self.conf = conf
        self.label = label
        self.resolution = resolution


class Results:
    def __init__(self):
        self.regions = []

    def is_dup(self, result_to_add, threshold=0.7):
        for existing_result in self.regions:
            # If the results are from a different frame
            # No need to check intersection area
            if result_to_add.fid != existing_result.fid:
                continue

            # Check if there is significant intersection
            # between a the result we want to add and a
            # region in the same frame
            if calc_iou(result_to_add, existing_result) > threshold:
                return True
        return False

    def combine_results(self, additional_results, intersection_threshold=0.7):
        for result in additional_results:
            if not self.is_dup(result, intersection_threshold):
                self.regions.append(result)
        # Sort final results
        self.single_obj_results.sort(key=lambda x: x.fid)

    def add_single_result(self, result_to_add):
        temp_results = Results()
        temp_results.regions = [result_to_add]
        self.combine_results(temp_results)


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
        label = int(line[5])
        conf = float(line[6])
        resolution = float(line[7])
        single_result = Region(fid, x, y, w, h, conf, label, resolution)

        if fid not in results_dict:
            results_dict[fid] = []
        results_dict[fid].append(single_result)

    return results_dict


def read_results_dict(fname, fmat="csv"):
    # TODO: Need to implement a CSV function
    if fmat == "txt":
        return read_results_txt_dict(fname)


def write_results_txt(results, fname):
    results_file = open(fname, "r")
    for result in results.single_obj_results:
        results_file.write("{},{},{},{},{},{},{},{}".format(result.fid,
                                                            result.x, result.y,
                                                            result.w, result.h,
                                                            result.label,
                                                            result.conf,
                                                            result.resolution))
    results_file.close()


def write_results(results, fname, fmat="csv"):
    if fmat == "txt":
        write_results_txt(results, fname)


def calc_intersection_area(a, b):
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.w, b.y + b.w)
    ri = min(a.x + a.w, b.y + b.w)

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
