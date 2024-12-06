import gradio as gr
from pathlib import Path
from hloc_glomap.scripts import run_command, CONSOLE
from hloc_glomap.process_data import run_hloc_reconstruction


def run_reconstruction_fn(
    input_zip_file: str,
    password: str | None = None,
    progress=gr.Progress(track_tqdm=True),
) -> None:
    zip_path: Path = Path(input_zip_file)
    assert zip_path.exists(), zip_path
    unzip_cmd: list[str] = [  # noqa: E501
        f"unzip {zip_path}",
        f"-d {zip_path.parent}",
    ]
    if password:
        unzip_cmd: list[str] = [  # noqa: E501
            f"unzip -P {password} {zip_path}",
            f"-d {zip_path.parent}",
        ]
    unzip_cmd: str = " ".join(unzip_cmd)
    CONSOLE.print(f"Running command: {unzip_cmd}")
    run_command(cmd=unzip_cmd, verbose=True)
    CONSOLE.print(f"Unzipped {zip_path} to {zip_path.parent}")
    input_dir: Path = zip_path.parent / zip_path.stem
    run_hloc_reconstruction(
        image_dir=input_dir / "images", colmap_dir=input_dir / "glomap"
    )


with gr.Blocks() as process_block:
    # input_file = gr.File(file_count="single", file_types=["zip"], type="filepath")
    gr.HTML(
        """
        <html>
        <head>
            <title>My Great Game</title>
            <style>
                iframe {
                    width: 100%;
                    height: 800px;
                }
            </style>
        </head>
        <body>
            <iframe loading="lazy" src="https://playcanv.as/b/6ea34b3f"></iframe>
        </body>
        </html>
        """
    )
    # with gr.Accordion(label="Advanced Options", open=False):
    #     password = gr.Textbox(label="Password", type="password")

    # run_btn = gr.Button(value="Run Reconstruction")
    # run_btn.click(
    #     fn=run_reconstruction_fn,
    #     inputs=(input_file, password),
    #     outputs=None,
    # )
