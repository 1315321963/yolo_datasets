from ultralytics import YOLO

'''
训练前需要删除的文件:
labels中的 .cache
C:/Users/Administrator/AppData/Roaming/Ultralytics/settings.yaml
'''


def running():
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='coco128.yaml', epochs=4000, patience=4000, batch=-1, workers=10, imgsz=640)

    # # Evaluate the model's performance on the validation set
    # results = model.val()
    #
    # # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')
    #
    # # Export the model to ONNX format
    # success = model.export(format='onnx')


if __name__ == '__main__':
    running()
