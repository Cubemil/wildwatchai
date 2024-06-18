from inference import get_model
import supervision as sv
import cv2
# import the InferencePipeline interface
from inference import InferencePipeline
# import a built in sink called render_boxes (sinks are the logic that happens after inference)
from inference.core.interfaces.stream.sinks import render_boxes

# create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="trail-camera-animal-detection/6", # set the model id to a yolov8x model with in put size 1280
    video_reference="C:\Users\C1\PycharmProjects\wildwatchai\LNW-01\20210612-15\StA16_W\SCEU0016.MOV", # set the video reference (source of video), it can be a link/path to a video file, an RTSP stream url, or an integer representing a device id (usually 0 for built in webcams)
    on_prediction=render_boxes, # tell the pipeline object what to do with each set of inference by passing a function
    api_key=DkyN1NJfe1UNLUWOr9Yk, # provide your roboflow api key for loading models from the roboflow api
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()

# load a pre-trained yolov8n model
model = get_model(model_id="trail-camera-detection/6")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)