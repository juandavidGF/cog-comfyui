# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        image_filenames = kwargs["image_filenames"]

        # Update the workflow for each image
        node_numbers = ["1", "8", "9"]  # Corresponding to subject, background, and light images
        for node_num, filename in zip(node_numbers, image_filenames):
            if filename:
                node = workflow[node_num]["inputs"]
                node["image"] = filename

        # positive_prompt = workflow["6"]["inputs"]
        # positive_prompt["text"] = kwargs["prompt"]

        # negative_prompt = workflow["7"]["inputs"]
        # negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        # sampler = workflow["3"]["inputs"]
        # sampler["seed"] = kwargs["seed"]
        pass

    def predict(
        self,
        subject_image: Path = Input(
            description="Subject image",
            default=None,
        ),
        background_image: Path = Input(
            description="(Optional) Background here",
            default=None,
        ),
        light_image: Path = Input(
            description="(Optional) Light Mask Here",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        image_filenames = []
        for i, image in enumerate([subject_image, background_image, light_image], 1):
            if image:
                image_filename = self.filename_with_extension(image, f"image{i}")
                self.handle_input_file(image, image_filename)
                image_filenames.append(image_filename)
            else:
                image_filenames.append(None)

        print("image_filenames", image_filenames)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            image_filenames=image_filenames,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )