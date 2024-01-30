import gradio as gr
from ultralytics import YOLO
import numpy as np
import random, os

def detect_load(Dataset):
    dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/datasets/' + Dataset + '/train/images'
    images = [(dataset_dir + '/' + filename, str(i)) for i, filename in enumerate([f for f in os.listdir(dataset_dir)])]
    global Selected_Dataset, Task
    Selected_Dataset, Task = Dataset, "Detect"
    return images

def detect_submit(Model, Epochs, Input_Size, Exports):
    if Task == "Detect":
        model = YOLO(Model + '.yaml')
        model.train(data='datasets/' + Selected_Dataset + '/data.yaml', epochs=int(Epochs), imgsz=int(Input_Size))
        output_path = model.export()
        results_dir = output_path.split("weights")[0]
        images = [(results_dir + '/' + filename, filename.split('.')[0]) for filename in [f for f in os.listdir(results_dir)] if '.jpg' in filename or  '.png' in filename]
        return images, Model + " (Detect) training successed."
    else:
        if Task == None:
            return "Dataset are not loaded. please press 'Load Dataset' button."
        else:
            return None, Model + " (Detect) are not support given Dataset."

global Selected_Dataset, Task
Selected_Dataset, Task = None, None
detector = gr.Interface(detect_submit,
                             inputs = [   
                                          gr.Dropdown(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"], label="Model", info="", interactive=True),
                                          gr.Slider(2, 100, value=2, label="Epochs", info=""),
                                          gr.Slider(240, 2048, value=640, label="Input_Size", info=""),
                                          gr.CheckboxGroup(["PyTorch", "TorchScript", "ONNX", "OpenVINO", "TensorRT", "CoreML", "TF SavedModel", "TF Lite", "TF Edge TPU", "TF.js", "PaddlePaddle", "ncnn"], label="Exports", info=""),
                                      ],
                             outputs = [gr.Gallery(preview=True), gr.Textbox()],
                             allow_flagging="never",
)

with gr.Blocks() as Ultralytics:
    gr.Markdown("<span style='font-size:28px; font-weight:bold;'>Hi User, Welcome to ITRI Microservice  </span><span style='font-size:20px; font-weight:bold; color:gray;'>(powered by Ultralytics)</span>")
    gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[10], rows=[3], object_fit="contain", height="auto")
    Dataset = gr.Dropdown(["HardHat"], label="Dataset", info="")
    btn = gr.Button("Load Dataset")
    btn.click(detect_load, Dataset, gallery)

    with gr.Tabs():
        with gr.TabItem("Detect") as tabDetect:
            detector.render()
        with gr.TabItem("Classify") as tabClassify:
            pass
        with gr.TabItem("Segment") as tabSegment:
            pass
        with gr.TabItem("Pose") as tabPose:
            pass

if __name__ == "__main__":
    Ultralytics.launch()