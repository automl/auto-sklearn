import numpy as np

import plot_util
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    _, data = plot_util.read_csv(input_file)
    # Convert to numpy array for computing average.
    data = np.array(data).astype(float)
    time_steps = data[:, 0]
    average_performance = np.mean(data[:, 1:], axis=1)
    # Sanity check
    assert time_steps.shape == average_performance.shape

    with open(output_file, 'w') as fh:
        fh.write("Time,Test Performance\n")
        for i in range(time_steps.shape[0]):
            fh.write("{0},{1}\n".format(time_steps[i], average_performance[i]))


if __name__ == "__main__":
    main()
