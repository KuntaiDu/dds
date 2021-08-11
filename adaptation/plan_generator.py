import os
import json
import numpy as np


def compute_awstream(complete_trace, segments):
    trace = [complete_trace[0], complete_trace[1]]
    segment_ids = [0, 1]
    for idx in range(2, segments):
        bw_for_segment = (
            (complete_trace[idx - 1] + complete_trace[idx - 2]) / 2)
        trace.append(bw_for_segment)
        segment_ids.append(idx)
    return zip(segment_ids, trace)


def compute_dds(complete_trace, segments, window_size):
    trace = [min(complete_trace[0], complete_trace[1])]
    inputs = [window_size]
    for idx in range(2, segments, window_size):
        bw_for_segment = (
            (complete_trace[idx - 1] + complete_trace[idx - 2]) / 2)
        trace.append(bw_for_segment)
        inputs.append(window_size)
    return zip(inputs, trace)


def write_awstream(direc, pcage, awstream_trace):
    file_path = os.path.join(direc, f"aws-{pcage}-trace")
    with open(file_path, "w") as f:
        for seg_id, bw in awstream_trace:
            f.write(f"{seg_id} {bw}\n")


def write_dds(direc, pcage, window_size, dds_trace):
    file_path = os.path.join(direc, f"dds-{pcage}-trace")
    perturbations = []
    with open("perturbation_entry") as f:
        entry = f.read()
    with open(file_path, "w") as f:
        for inputs, bw in dds_trace:
            f.write(f"{inputs} {bw}\n")
            curr_perturbation = json.loads(
                entry.replace("INPUTS", str(window_size))
                .replace("CONSTRAINT", str(bw)))
            perturbations.append(curr_perturbation)
    plan = json.load(open("plan.json"))
    test_case = plan["testCases"][0]
    test_case["perturbations"] = perturbations
    with open(os.path.join(direc, f"plan-{pcage}.json"), "w") as outfile:
        outfile.write(json.dumps(plan, indent=2))


def main():
    mean = 900
    n_segments = 22
    window_size = 2
    for pcage in range(5, 45, 5):
        direc = f"bandwidth-{pcage}pc"
        os.makedirs(direc, exist_ok=True)
        std = (pcage / 100) * mean
        complete_trace = np.random.normal(mean, std, n_segments)
        with open(os.path.join(direc, "complete_trace"), "w") as f:
            for idx, bw in enumerate(complete_trace):
                f.write(f"{idx} {bw}\n")
        awstream_trace = compute_awstream(complete_trace, n_segments)
        dds_trace = compute_dds(complete_trace, n_segments, window_size)
        write_awstream(direc, pcage, awstream_trace)
        write_dds(direc, pcage, window_size, dds_trace)


if __name__ == "__main__":
    main()
