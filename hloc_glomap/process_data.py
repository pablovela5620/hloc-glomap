from pathlib import Path
from typing import Literal
from hloc_glomap.colmap_utils import CameraModel, colmap_to_json
from hloc_glomap.scripts import CONSOLE, run_command, status
from hloc.reconstruction import create_empty_db, import_images, get_image_ids
from hloc.triangulation import (
    import_features,
    import_matches,
    estimation_and_geometric_verification,
)
from hloc.utils.io import get_keypoints, get_matches
from hloc import pairs_from_sequential
from tqdm import tqdm
from timeit import default_timer as timer

import rerun as rr
import numpy as np
import cv2
from rerun_loader_colmap.main import read_and_log_sparse_reconstruction
from contextlib import contextmanager


class TimingLogger:
    def __init__(self, header: str = ""):
        self.start_time = timer()
        self.header = header
        self.entries = []  # Store (section, time) pairs
        # Initialize markdown table
        self.markdown_table = (
            f"# {self.header}\n"
            "| Pipeline Section | Time Taken |\n"
            "|------------------|------------|\n"
        )
        rr.log(
            "logs",
            rr.TextDocument(self.markdown_table, media_type="text/markdown"),
            static=True,
        )

    @contextmanager
    def log_time(self, section_name: str):
        start_time = timer()
        try:
            yield
        finally:
            time_taken = timer() - start_time
            minutes, seconds = divmod(time_taken, 60)
            time_str = f"{int(minutes)}m {seconds:.1f}s"

            # Add new row to table
            self.markdown_table += f"| {section_name} | {time_str} |\n"

            # Update rerun log
            rr.log(
                "logs",
                rr.TextDocument(self.markdown_table, media_type="text/markdown"),
                static=True,
            )


def log_features(
    bgr_dict: dict[str, np.ndarray],
    features: Path,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"],
    max_dim: int = 640,
) -> None:
    # log keypoints
    for idx, (image_name, bgr) in enumerate(
        tqdm(bgr_dict.items(), desc="Logging keypoints")
    ):
        rr.set_time("features", sequence=idx)
        cam_log_path: str = (
            "camera" if matching_method == "sequential" else f"{image_name}"
        )
        keypoints = get_keypoints(features, image_name)

        # reshape bgr and keypoints to vga like resolution max dim 640
        scale = max_dim / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, (0, 0), fx=scale, fy=scale)
        keypoints = keypoints * scale

        rr.log(
            f"{cam_log_path}/image",
            rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=50),
        )
        rr.log(
            f"{cam_log_path}/image/keypoints",
            rr.Points2D(positions=keypoints, colors=(0, 255, 0), radii=max_dim * 0.001),
        )


def log_matches(
    bgr_dict: dict[str, np.ndarray],
    pairs_path: Path,
    matches_path: Path,
    features_path: Path,
    image_ids: dict[str, int],
) -> None:
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    matched = set()
    cam_log_path = "camera"
    # matched_images = []

    rr.reset_time()  # Clears all set timeline info.
    for idx, (name0, name1) in enumerate(
        tqdm(pairs, desc="Logging matches", total=len(pairs))
    ):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        rr.set_time("matches", sequence=idx)
        # load matches
        matches, scores = get_matches(matches_path, name0, name1)

        # load keypoints
        keypoints0 = get_keypoints(features_path, name0)[matches[:, 0]]
        keypoints1 = get_keypoints(features_path, name1)[matches[:, 1]]
        bgr0 = bgr_dict[name0]
        bgr1 = bgr_dict[name1]
        h0, w0, _ = bgr0.shape
        h1, w1, _ = bgr1.shape
        # merge images horizontally
        bgr = cv2.hconcat([bgr0, bgr1])
        keypoints1[:, 0] += w0

        # resize to vga like resolution max dim 640
        max_dim = 640
        scale = max_dim / max(h0, w0, h1, w1)
        bgr = cv2.resize(bgr, (0, 0), fx=scale, fy=scale)
        keypoints0 *= scale
        keypoints1 *= scale

        line_segments = np.stack([keypoints0, keypoints1], axis=1)  # Shape: (n, 2, 2)
        # matched_images.append(bgr)

        # draw matches
        rr.log(
            f"{cam_log_path}/image",
            rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=25),
        )
        rr.log(f"{cam_log_path}/image/keypoints0", rr.Points2D(positions=keypoints0))
        rr.log(f"{cam_log_path}/image/keypoints1", rr.Points2D(positions=keypoints1))
        rr.log(
            f"{cam_log_path}/image/matches",
            rr.LineStrips2D(strips=line_segments, colors=(0, 255, 0), radii=0.25),
        )

        matched |= {(id0, id1), (id1, id0)}
    # only sends uncompressed images
    # send_data_columns(matched_images=matched_images, cam_log_path=cam_log_path)


def send_data_columns(matched_images: np.ndarray, cam_log_path: str) -> None:
    # Timeline on which the images are distributed.
    start = timer()
    matched_images = np.stack(matched_images)

    sequence = np.arange(0, len(matched_images))
    # Log the ImageFormat and indicator once, as static.

    format_static = rr.components.ImageFormat(
        width=matched_images.shape[2],
        height=matched_images.shape[1],
        color_model="BGR",
        channel_datatype="U8",
    )
    rr.log(
        f"{cam_log_path}/image",
        [format_static, rr.Image.indicator()],
        static=True,
    )

    # Send all images at once.
    rr.send_columns(
        f"{cam_log_path}/image",
        times=[rr.TimeSequenceColumn("matches", sequence)],
        # Reshape the images so `ImageBufferBatch` can tell that this is several blobs.
        #
        # Note that the `ImageBufferBatch` consumes arrays of bytes,
        # so if you have a different channel datatype than `U8`, you need to make sure
        # that the data is converted to arrays of bytes before passing it to `ImageBufferBatch`.
        components=[
            rr.components.ImageBufferBatch(matched_images.reshape(len(sequence), -1))
        ],
    )

    print(f"Time taken to log matches: {timer() - start:.2f} seconds")


def run_hloc_reconstruction(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel = CameraModel.OPENCV,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "sequential",
    feature_type: Literal[
        "sift", "superpoint_aachen", "disk", "xfeat", "aliked-n16"
    ] = "disk",
    matcher_type: Literal[
        "superglue",
        "NN-ratio",
        "NN-mutual",
        "disk+lightglue",
        "superpoint+lightglue",
        "aliked+lightglue",
        "xfeat+lighterglue",
    ] = "disk+lightglue",
    num_matched: int = 50,
    use_single_camera_mode: bool = True,
    use_loop_closure: bool = True,
    colmap_cmd: Literal["colmap", "glomap"] = "glomap",
) -> None:
    """Runs hloc on the images.

    Args:
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        verbose: If True, logs the output of the command.
        matching_method: Method to use for matching images.
        feature_type: Type of visual features to use.
        matcher_type: Type of feature matcher to use.
        num_matched: Number of image pairs for loc.
        use_single_camera_mode: If True, uses one camera for all frames. Otherwise uses one camera per frame.
    """

    import pycolmap
    from hloc import (  # type: ignore
        extract_features,
        match_features,
        pairs_from_exhaustive,
        pairs_from_retrieval,
    )

    timing_logger = TimingLogger(header=f"{colmap_dir.name}")

    if colmap_dir.exists() and colmap_dir.name == "test":
        print("Removing existing colmap_dir")
        import shutil

        shutil.rmtree(colmap_dir)

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sparse" / "0"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    # load bgr images into a dict for logging
    bgr_dict: dict[str, np.ndarray] = {}
    for img_path in sorted(image_dir.iterdir()):
        bgr_dict[img_path.name] = cv2.imread(str(img_path))

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    references: list[str] = sorted(
        [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    )

    with timing_logger.log_time("end-to-end pipeline"):
        with timing_logger.log_time("feature detection"):
            extract_features.main(
                feature_conf, image_dir, image_list=references, feature_path=features
            )  # type: ignore

        with timing_logger.log_time("log feature extraction"):
            log_features(
                bgr_dict=bgr_dict, features=features, matching_method=matching_method
            )

        if matching_method == "exhaustive":
            pairs_from_exhaustive.main(sfm_pairs, image_list=references)  # type: ignore
        elif matching_method == "sequential":
            if use_loop_closure:
                retrieval_path = extract_features.main(
                    retrieval_conf, image_dir, outputs
                )  # type: ignore
            pairs_from_sequential.main(
                output=sfm_pairs,
                image_list=references,
                features=features,
                window_size=10,
                quadratic_overlap=False,
                use_loop_closure=use_loop_closure,
                retrieval_path=retrieval_path,
            )  # type: ignore
        else:
            retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
            num_matched = min(len(references), num_matched)
            pairs_from_retrieval.main(
                retrieval_path, sfm_pairs, num_matched=num_matched
            )  # type: ignore

        with timing_logger.log_time("feature matching"):
            match_features.main(
                matcher_conf, sfm_pairs, features=features, matches=matches
            )  # type: ignore

        image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore

        if use_single_camera_mode:  # one camera per all frames
            camera_mode = pycolmap.CameraMode.SINGLE  # type: ignore
        else:  # one camera per frame
            camera_mode = pycolmap.CameraMode.PER_IMAGE  # type: ignore

        assert features.exists(), features
        assert sfm_pairs.exists(), sfm_pairs
        assert matches.exists(), matches

        sfm_dir.mkdir(parents=True, exist_ok=True)
        database = sfm_dir / "database.db"

        create_empty_db(database)
        # `pycolmap.import_images` changed from `image_list` to `image_names` in
        # newer bindings. Call the binding directly so this works across versions.
        try:
            pycolmap.import_images(
                str(database),
                str(image_dir),
                camera_mode,
                image_names=[],
                options=image_options,
            )
        except TypeError:
            pycolmap.import_images(
                str(database),
                str(image_dir),
                camera_mode,
                image_list=[],
                options=image_options,
            )
        image_ids: dict[str, int] = get_image_ids(database)

        with timing_logger.log_time("log matches"):
            log_matches(
                bgr_dict=bgr_dict,
                pairs_path=sfm_pairs,
                matches_path=matches,
                features_path=features,
                image_ids=image_ids,
            )

        import_features(image_ids, database, features)
        import_matches(
            image_ids,
            database,
            sfm_pairs,
            matches,
            min_match_score=None,
            skip_geometric_verification=False,
        )
        estimation_and_geometric_verification(database, sfm_pairs, verbose)

        # Bundle adjustment
        num_images: int = len(list(image_dir.glob("*")))
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        mapper_cmd = [
            f"{colmap_cmd} mapper",
            f"--database_path {database}",
            f"--image_path {image_dir}",
            f"--output_path {sparse_dir}",
        ]
        if colmap_cmd == "glomap":
            mapper_cmd += [
                f"--TrackEstablishment.max_num_tracks {1000*num_images}",
            ]

        mapper_cmd = " ".join(mapper_cmd)

        with timing_logger.log_time(f"{colmap_cmd} bundle adjustment"):
            with status(
                msg=f"[bold yellow]Running {colmap_cmd} bundle adjustment... (This may take a while)",
                spinner="circle",
                verbose=verbose,
            ):
                run_command(mapper_cmd, verbose=verbose)

        # save to parent directory so it works with nerfstudio out of the box
        colmap_to_json(
            recon_dir=sfm_dir,
            output_dir=colmap_dir.parent,
        )
        rr.reset_time()  # Clears all set timeline info.
        try:
            read_and_log_sparse_reconstruction(
                model_path=sfm_dir, filter_output=False, resize=None, extention=".bin"
            )
        except AttributeError as exc:
            CONSOLE.log(
                f"[yellow]Skipping sparse reconstruction Rerun logging due to SDK API mismatch:[/] {exc}"
            )

        CONSOLE.log(f"[bold green]:tada: Done {colmap_cmd} bundle adjustment.")
