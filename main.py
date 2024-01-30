import gradio as gr
from ultralytics import YOLO

def builder(Model, Dataset, Epochs, Input_Size):
    model = YOLO(Model + '.yaml')
    model.train(data='datasets/' + Dataset + '/data.yaml', epochs=int(Epochs), imgsz=int(Input_Size))
    return f"""{Models} Load Successed."""


YOLO_detect = gr.Interface(
    builder,
    [
        gr.Radio(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"], label="Model", info=""),
        gr.Radio(["HardHat"], label="Dataset", info=""),
        gr.Slider(2, 20, value=2, label="Epochs", info=""),
        gr.Slider(240, 2048, value=640, label="Input_Size", info=""),
    ],

    "text",
)

if __name__ == "__main__":
    YOLO_detect.launch()