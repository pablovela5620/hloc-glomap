import os
import argparse
import collections.abc as collections
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

from hloc import logger
from hloc import pairs_from_retrieval
from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.io import list_h5_names


def main(
    output: Path,
    image_list: Optional[Union[Path, List[str]]] = None,
    features: Optional[Path] = None,
    window_size: Optional[int] = 10,
    quadratic_overlap: bool = True,
    use_loop_closure: bool = False,
    retrieval_path: Optional[Union[Path, str]] = None,
    retrieval_interval: Optional[int] = 2,
    num_loc: Optional[int] = 5,
) -> None:
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            print(image_list)
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    pairs = []
    N = len(names_q)

    for i in range(N - 1):
        for j in range(i + 1, min(i + window_size + 1, N)):
            pairs.append((names_q[i], names_q[j]))

            if quadratic_overlap:
                q = 2 ** (j - i)
                if q > window_size and i + q < N:
                    pairs.append((names_q[i], names_q[i + q]))

    if use_loop_closure:
        retrieval_pairs_tmp = output.parent / f"retrieval-pairs-tmp.txt"

        # match mask describes for each image, which images NOT to include in retrevial match search
        # I.e., no reason to get retrieval matches for matches already included from sequential matching

        query_list = names_q[::retrieval_interval]
        M = len(query_list)
        match_mask = np.zeros((M, N), dtype=bool)

        for i in range(M):
            for k in range(window_size + 1):
                if i * retrieval_interval - k >= 0 and i * retrieval_interval - k < N:
                    match_mask[i][i * retrieval_interval - k] = 1
                if i * retrieval_interval + k >= 0 and i * retrieval_interval + k < N:
                    match_mask[i][i * retrieval_interval + k] = 1

                if quadratic_overlap:
                    if (
                        i * retrieval_interval - 2**k >= 0
                        and i * retrieval_interval - 2**k < N
                    ):
                        match_mask[i][i * retrieval_interval - 2**k] = 1
                    if (
                        i * retrieval_interval + 2**k >= 0
                        and i * retrieval_interval + 2**k < N
                    ):
                        match_mask[i][i * retrieval_interval + 2**k] = 1

        pairs_from_retrieval.main(
            retrieval_path,
            retrieval_pairs_tmp,
            num_matched=num_loc,
            match_mask=match_mask,
            db_list=names_q,
            query_list=query_list,
        )

        retrieval = parse_retrieval(retrieval_pairs_tmp)

        for key, val in retrieval.items():
            for match in val:
                pairs.append((key, match))

        os.unlink(retrieval_pairs_tmp)

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a list of image pairs based on the sequence of images on alphabetic order"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--overlap", type=int, default=10, help="Number of overlapping image pairs"
    )
    parser.add_argument(
        "--quadratic_overlap",
        action="store_true",
        help="Whether to match images against their quadratic neighbors.",
    )
    args = parser.parse_args()
    main(**args.__dict__)
