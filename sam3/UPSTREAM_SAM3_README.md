---
license: other
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Reporter
    - Other
  geo: ip_location
  By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Meta Privacy Policy: checkbox
extra_gated_description: >-
  The information you provide will be collected, stored, processed and shared in
  accordance with the [Meta Privacy
  Policy](https://www.facebook.com/privacy/policy/).
extra_gated_button_content: Submit
language:
- en
pipeline_tag: mask-generation
library_name: transformers
tags:
- sam3
---

SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor [SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. It achieves 75-80% of human performance on our new [SA-CO benchmark](https://github.com/facebookresearch/sam3/edit/main_readme/README.md#sa-co-dataset) which contains 270K unique concepts, over 50 times more than existing benchmarks.

[Hugging Face 🤗  app](https://huggingface.co/spaces/akhaliq/sam3)

### Basic Usage

```python
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

The official code is publicly released in the [sam3 repo](https://github.com/facebookresearch/sam3).


## Usage with 🤗 Transformers

### SAM3 - Promptable Concept Segmentation (PCS) for Images

SAM3 performs Promptable Concept Segmentation (PCS) on images, taking text and/or image exemplars as prompts and returning segmentation masks for **all matching object instances** in the image.

#### Text-Only Prompts

```python
>>> from transformers import Sam3Processor, Sam3Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam3Model.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3")

>>> # Load image
>>> image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> # Segment using text prompt
>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]

>>> print(f"Found {len(results['masks'])} objects")
>>> # Results contain:
>>> # - masks: Binary masks resized to original image size
>>> # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
>>> # - scores: Confidence scores
```

You can display masks using a simple helper like the following:

```python
import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image
```

Then you can save the resulting composite image or display it in a notebook:

```python
>>> overlay_masks(image, results["masks"])
```

#### Single Bounding Box Prompt

Segment objects using a bounding box:

```python
>>> # Box in xyxy format: [x1, y1, x2, y2] in pixel coordinates
>>> # Example: laptop region
>>> box_xyxy = [100, 150, 500, 450]
>>> input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
>>> input_boxes_labels = [[1]]  # 1 = positive box

>>> inputs = processor(
...     images=image,
...     input_boxes=input_boxes,
...     input_boxes_labels=input_boxes_labels,
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]
```

#### Multiple Box Prompts (Positive and Negative)

Use multiple boxes with positive and negative labels to refine the concept:

```python
>>> # Load kitchen image
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> kitchen_image = Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")

>>> # Define two positive boxes (e.g., dial and button on oven)
>>> # Boxes are in xyxy format [x1, y1, x2, y2] in pixel coordinates
>>> box1_xyxy = [59, 144, 76, 163]  # Dial box
>>> box2_xyxy = [87, 148, 104, 159]  # Button box
>>> input_boxes = [[box1_xyxy, box2_xyxy]]
>>> input_boxes_labels = [[1, 1]]  # Both positive

>>> inputs = processor(
...     images=kitchen_image,
...     input_boxes=input_boxes,
...     input_boxes_labels=input_boxes_labels,
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]
>>> overlay_masks(kitchen_image, results["masks"])
```

#### Combined Prompts (Text + Negative Box)

Use text prompts with negative visual prompts to refine the concept:

```python
>>> # Segment "handle" but exclude the oven handle using a negative box
>>> text = "handle"
>>> # Negative box covering oven handle area (xyxy): [40, 183, 318, 204]
>>> oven_handle_box = [40, 183, 318, 204]
>>> input_boxes = [[oven_handle_box]]

>>> inputs = processor(
...     images=kitchen_image,
...     text=text,
...     input_boxes=input_boxes,
...     input_boxes_labels=[[0]],  # 0 = negative (exclude this region)
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]
>>> # This will segment pot handles but exclude the oven handle
```

#### Batched Inference with Text Prompts

Process multiple images with different text prompts by batch:

```python
>>> cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> images = [
...     Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
...     Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
... ]

>>> text_prompts = ["ear", "dial"]

>>> inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results for both images
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )

>>> print(f"Image 1: {len(results[0]['masks'])} objects found")
>>> print(f"Image 2: {len(results[1]['masks'])} objects found")
```

#### Batched Mixed Prompts

Use different prompt types for different images in the same batch:

```python
>>> # Image 1: text prompt "laptop"
>>> # Image 2: visual prompt (dial box)
>>> box2_xyxy = [59, 144, 76, 163]

>>> inputs = processor(
...     images=images,
...     text=["laptop", None],  # Only first image has text
...     input_boxes=[None, [box2_xyxy]],  # Only second image has box
...     input_boxes_labels=[None, [1]],  # Positive box for second image
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results for both images
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )
>>> # Both images processed in single forward pass
```

#### Semantic Segmentation Output

SAM3 also provides semantic segmentation alongside instance masks:

```python
>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Instance segmentation masks
>>> instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

>>> # Semantic segmentation (single channel)
>>> semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

>>> print(f"Instance masks: {instance_masks.shape}")
>>> print(f"Semantic segmentation: {semantic_seg.shape}")
```

### SAM3 Video - Promptable Concept Segmentation (PCS) for Videos

SAM3 Video performs Promptable Concept Segmentation (PCS) on videos, taking text as prompts and detecting and tracking **all matching object instances** across video frames.

#### Pre-loaded Video Inference

Process a video with all frames already available using text prompts:

```python
>>> from transformers import Sam3VideoModel, Sam3VideoProcessor
>>> from accelerate import Accelerator
>>> import torch

>>> device = Accelerator().device
>>> model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
>>> processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

>>> # Load video frames
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     processing_device="cpu",
...     video_storage_device="cpu",
...     dtype=torch.bfloat16,
... )

>>> # Add text prompt to detect and track objects
>>> text = "person"
>>> inference_session = processor.add_text_prompt(
...     inference_session=inference_session,
...     text=text,
... )

>>> # Process all frames in the video
>>> outputs_per_frame = {}
>>> for model_outputs in model.propagate_in_video_iterator(
...     inference_session=inference_session, max_frame_num_to_track=50
... ):
...     processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
...     outputs_per_frame[model_outputs.frame_idx] = processed_outputs

>>> print(f"Processed {len(outputs_per_frame)} frames")
Processed 51 frames

>>> # Access results for a specific frame
>>> frame_0_outputs = outputs_per_frame[0]
>>> print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
>>> print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
>>> print(f"Scores: {frame_0_outputs['scores'].tolist()}")
>>> print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
>>> print(f"Masks shape: {frame_0_outputs['masks'].shape}")
```

#### Streaming Video Inference

For real-time applications, the Transformers implementation of SAM3 Video supports processing video frames as they arrive:

```python
>>> # Initialize session for streaming
>>> streaming_inference_session = processor.init_video_session(
...     inference_device=device,
...     processing_device="cpu",
...     video_storage_device="cpu",
...     dtype=torch.bfloat16,
... )

>>> # Add text prompt
>>> text = "person"
>>> streaming_inference_session = processor.add_text_prompt(
...     inference_session=streaming_inference_session,
...     text=text,
... )

>>> # Process frames one by one (streaming mode)
>>> streaming_outputs_per_frame = {}
>>> for frame_idx, frame in enumerate(video_frames[:50]):  # Process first 50 frames
...     # First, process the frame using the processor
...     inputs = processor(images=frame, device=device, return_tensors="pt")
...
...     # Process frame using streaming inference - pass the processed pixel_values
...     model_outputs = model(
...         inference_session=streaming_inference_session,
...         frame=inputs.pixel_values[0],  # Provide processed frame - this enables streaming mode
...         reverse=False,
...     )
...
...     # Post-process outputs with original_sizes for proper resolution handling
...     processed_outputs = processor.postprocess_outputs(
...         streaming_inference_session,
...         model_outputs,
...         original_sizes=inputs.original_sizes,  # Required for streaming inference
...     )
...     streaming_outputs_per_frame[frame_idx] = processed_outputs
...
...     if (frame_idx + 1) % 10 == 0:
...         print(f"Processed {frame_idx + 1} frames...")

>>> print(f"✓ Streaming inference complete! Processed {len(streaming_outputs_per_frame)} frames")
✓ Streaming inference complete! Processed 50 frames

>>> # Access results
>>> frame_0_outputs = streaming_outputs_per_frame[0]
>>> print(f"Detected {len(frame_0_outputs['object_ids'])} objects in first frame")
>>> print(f"Boxes are in XYXY format (absolute pixel coordinates): {frame_0_outputs['boxes'].shape}")
>>> print(f"Masks are at original video resolution: {frame_0_outputs['masks'].shape}")
```

<div class="warning">
⚠️ **Note on Streaming Inference Quality**: Streaming inference disables hotstart heuristics that remove unmatched and duplicate objects, as these require access to future frames to make informed decisions. This may result in more false positive detections and duplicate object tracks compared to pre-loaded video inference. For best results, use pre-loaded video inference when all frames are available.
</div>

### SAM3 Tracker - Promptable Visual Segmentation (PVS) for Images

Sam3Tracker performs Promptable Visual Segmentation (PVS) on images, taking interactive visual prompts (points, boxes, masks) to segment a **specific object instance** per prompt. It is an updated version of SAM2 that maintains the same API while providing improved performance, making it a drop-in replacement for SAM2 workflows.

#### Automatic Mask Generation with Pipeline

```python
>>> from transformers import pipeline

>>> generator = pipeline("mask-generation", model="facebook/sam3", device=0)
>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> outputs = generator(image_url, points_per_batch=64)

>>> len(outputs["masks"])  # Number of masks generated
```

#### Basic Image Segmentation

##### Single Point Click

```python
>>> from transformers import Sam3TrackerProcessor, Sam3TrackerModel
>>> from accelerate import Accelerator
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = Accelerator().device

>>> model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
>>> input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

>>> # The model outputs multiple mask predictions ranked by quality score
>>> print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")
```

##### Multiple Points for Refinement

```python
>>> # Add both positive and negative points to refine the mask
>>> input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
>>> input_labels = [[[1, 1]]]  # Both positive clicks

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

##### Bounding Box Input

```python
>>> # Define bounding box as [x_min, y_min, x_max, y_max]
>>> input_boxes = [[[75, 275, 1725, 850]]]

>>> inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

##### Multiple Objects Segmentation

```python
>>> # Define points for two different objects
>>> input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
>>> input_labels = [[[1], [1]]]  # Positive clicks for both objects

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Each object gets its own mask
>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
>>> print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects
```

#### Batch Inference


```python
>>> # Load multiple images
>>> image_urls = [
...     "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
... ]
>>> raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

>>> # Single point per image
>>> input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
>>> input_labels = [[[1]], [[1]]]  # Positive clicks for both images

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Post-process masks for each image
>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")
```

### SAM3 Tracker Video - Promptable Visual Segmentation (PVS) for Videos

Sam3TrackerVideo performs Promptable Visual Segmentation (PVS) on videos, taking interactive visual prompts (points, boxes, masks) to track a **specific object instance** per prompt across video frames. It is an updated version of SAM2 Video that maintains the same API while providing improved performance, making it a drop-in replacement for SAM2 Video workflows.

#### Basic Video Tracking

```python
>>> from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
>>> from accelerate import Accelerator
>>> import torch

>>> device = Accelerator().device
>>> model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
>>> processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

>>> # Load video frames
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Add click on first frame to select object
>>> ann_frame_idx = 0
>>> ann_obj_id = 1
>>> points = [[[[210, 350]]]]
>>> labels = [[[1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Segment the object on the first frame (optional, you can also propagate the masks through the video directly)
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Segmentation shape: {video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

>>> # Propagate through the entire video
>>> video_segments = {}
>>> for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam3_tracker_video_output.frame_idx] = video_res_masks

>>> print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 180 frames
```

#### Multi-Object Video Tracking

Track multiple objects simultaneously across video frames:

```python
>>> # Reset for new tracking session
>>> inference_session.reset_inference_session()

>>> # Add multiple objects on the first frame
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]
>>> input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
>>> input_labels = [[[1], [1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for both objects on first frame (optional, you can also propagate the masks through the video directly)
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )

>>> # Propagate both objects through video
>>> video_segments = {}
>>> for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam3_tracker_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

#### Streaming Video Inference

For real-time applications, Sam3TrackerVideo supports processing video frames as they arrive:

```python
>>> # Initialize session for streaming
>>> inference_session = processor.init_video_session(
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Process frames one by one
>>> for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
...     inputs = processor(images=frame, device=device, return_tensors="pt")
...
...     if frame_idx == 0:
...         # Add point input on first frame
...         processor.add_inputs_to_inference_session(
...             inference_session=inference_session,
...             frame_idx=0,
...             obj_ids=1,
...             input_points=[[[[210, 350], [250, 220]]]],
...             input_labels=[[[1, 1]]],
...             original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
...         )
...
...     # Process current frame
...     sam3_tracker_video_output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
...
...     video_res_masks = processor.post_process_masks(
...         [sam3_tracker_video_output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
...     )[0]
...     print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")
```