import gradio as gr
from ultralytics import YOLO
import numpy as np
import random, os
"""
def change_gallery(Dataset):
    print("[change_gallery] Dataset", Dataset)
    return f"Change {Dataset}!"

def builder(Model, Dataset, Epochs, Input_Size, Exports):
    model = YOLO(Model + '.yaml')
    model.train(data='datasets/' + Dataset + '/data.yaml', epochs=int(Epochs), imgsz=int(Input_Size))
    print(Exports)
    return Model

dataset_option = gr.Dropdown(["HardHat"], label="Dataset", info="")
dataset_gallery = gr.Gallery(label="Dataset_Dashboard", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")

YOLO_detect = gr.Interface(
    builder,
    [   
        gr.Dropdown(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"], value="YOLOv8n", label="Model", info=""),
        gr.Slider(2, 100, value=2, label="Epochs", info=""),
        gr.Slider(240, 2048, value=640, label="Input_Size", info=""),
        gr.CheckboxGroup(["PyTorch", "TorchScript", "ONNX", "OpenVINO", "TensorRT", "CoreML", "TF SavedModel", "TF Lite", "TF Edge TPU", "TF.js", "PaddlePaddle", "ncnn"], label="Exports", info=""),
    ],
    "text",
)

with gr.Blocks() as Ultralytics:
    dataset_option = gr.Dropdown(["HardHat"], label="Dataset", info="")
    dataset_gallery = gr.Gallery(label="Dataset_Dashboard", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")
    gr.Dropdown(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"], value="YOLOv8n", label="Model", info=""),
    gr.Slider(2, 100, value=2, label="Epochs", info=""),
    gr.Slider(240, 2048, value=640, label="Input_Size", info=""),
    gr.CheckboxGroup(["PyTorch", "TorchScript", "ONNX", "OpenVINO", "TensorRT", "CoreML", "TF SavedModel", "TF Lite", "TF Edge TPU", "TF.js", "PaddlePaddle", "ncnn"], label="Exports", info=""),
    su_button.click(flip_text, inputs=text_input, outputs=text_output)
"""

def detect_load(Dataset):
    dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/datasets/' + Dataset + '/train/images'
    images = [(dataset_dir + '/' + filename, str(i)) for i, filename in enumerate([f for f in os.listdir(dataset_dir)])]
    global Selected_Dataset
    Selected_Dataset = Dataset
    return images

def detect_submit(Model, Epochs, Input_Size, Exports):
    print(Model, Epochs, Input_Size, Exports)
    model = YOLO(Model + '.yaml')
    model.train(data='datasets/' + Selected_Dataset + '/data.yaml', epochs=int(Epochs), imgsz=int(Input_Size))
    print(Exports)
    return Model
    return 'OK'

global Selected_Dataset
Selected_Dataset = None
model_trainer = gr.Interface(detect_submit,
                             inputs = [   
                                          gr.Dropdown(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"], label="Model", info="", interactive=True),
                                          gr.Slider(2, 100, value=2, label="Epochs", info=""),
                                          gr.Slider(240, 2048, value=640, label="Input_Size", info=""),
                                          gr.CheckboxGroup(["PyTorch", "TorchScript", "ONNX", "OpenVINO", "TensorRT", "CoreML", "TF SavedModel", "TF Lite", "TF Edge TPU", "TF.js", "PaddlePaddle", "ncnn"], label="Exports", info=""),
                                      ],
                             outputs = gr.Textbox()
)

with gr.Blocks() as Ultralytics:
    gr.Markdown("<span style='font-size:28px; font-weight:bold;'>Hi User, Welcome to ITRI Microservice.</span><span style='font-size:20px; font-weight:bold;'>(powered by Ultralytics)</span>")
    gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[10], rows=[4], object_fit="contain", height="auto")
    Dataset = gr.Dropdown(["HardHat"], label="Dataset", info="")
    btn = gr.Button("Load Dataset")
    btn.click(detect_load, Dataset, gallery)

    with gr.Tabs():
        with gr.TabItem("Detect") as tabDetect:
            with gr.Column(visible=True) as colA:
                with gr.Row(visible=True) as rowA:
                    model_trainer.render()

        with gr.TabItem("Classify") as tabClassify:
            pass
        with gr.TabItem("Segment") as tabSegment:
            pass
        with gr.TabItem("Pose") as tabPose:
            pass

if __name__ == "__main__":
    Ultralytics.launch()