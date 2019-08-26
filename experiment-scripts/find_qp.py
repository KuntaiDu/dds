import os
import sys
import subprocess
from multiprocessing import Process, Manager


def encode_and_get_size(qp, outname, images_path):
    encoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-loglevel", "error",
                                      '-i', f"{images_path}/%010d.png",
                                      "-vcodec", "libx264",
                                      "-qp", f"{qp}",
                                      "-pix_fmt", "yuv420p",
                                      outname],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    size = 0
    if encoding_result == 0:
        size = os.path.getsize(outname)
    os.remove(outname)
    return size


def find_sizes(search_range, images_path, original_size, results_list):
    for qp in search_range:
        size = encode_and_get_size(qp, f"test_{qp}.mp4", images_path)
        print(size, qp)
        results_list.append((abs(original_size - size), qp))


def main(images_path, thread_count=1):
    vname = [e for e in os.path.listdir(images_path) if ".mp4" in e][0]
    original_size = os.path.getsize(os.path.join(images_path, vname))

    qp_range = range(1, 41)

    thread_lists = []
    qp_per_thread = int(len(qp_range) / thread_count) + 1
    for thread_id in range(thread_count):
        manager = Manager()
        thread_list = manager.list()
        qp_search_space = qp_range[
            thread_id * qp_per_thread:
            min((thread_id + 1) * qp_per_thread, max(qp_range) + 1)]
        new_thread = Process(target=find_sizes,
                             args=(qp_search_space, images_path,
                                   original_size, thread_list))
        thread_lists.append((new_thread, thread_list))
        print("Starting process {thread_id}")
        new_thread.start()

    all_vals = []
    for thread_handle, thread_list in thread_lists:
        thread_handle.join()
        for e in thread_list:
            all_vals.append(e)

    print(min(all_vals, key=lambda e: e[0]))


if __name__ == "__main__":
    images_path = sys.argv[1]
    num_threads = int(sys.argv[2])
    main(images_path, num_threads)
