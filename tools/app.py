from hloc_glomap.gradio.gradio_ui import demo_block
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab(label="Process"):
        demo_block.render()

demo.launch(ssr_mode=False)
