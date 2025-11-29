import torch
import os
from comfy.model_management import get_torch_device
from .vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife, logger
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger
import folder_paths
import time
from polygraphy import cuda
import comfy.model_management as mm
import tensorrt
import json

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "rife")

# Image dimensions for TensorRT engine building
IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 512
IMAGE_DIM_MAX = 3840

# Logger for this module
rife_logger = ColoredLogger("ComfyUI-Rife-Tensorrt")

# Function to load configuration
def load_node_config(config_filename="load_rife_config.json"):
    """Loads node configuration from a JSON file."""
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)

    default_config = {
        "model": {
            "options": ["rife49_ensemble_True_scale_1_sim"],
            "default": "rife49_ensemble_True_scale_1_sim",
            "tooltip": "Default model (fallback from code)"
        },
        "precision": {
            "options": ["fp16", "fp32"],
            "default": "fp16",
            "tooltip": "Default precision (fallback from code)"
        }
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        rife_logger.info(f"Successfully loaded configuration from {config_filename}")
        return config
    except FileNotFoundError:
        rife_logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        rife_logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        rife_logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config

# Load the configuration once when the module is imported
LOAD_RIFE_NODE_CONFIG = load_node_config()

class LoadRifeTensorrtModel:
    @classmethod
    def INPUT_TYPES(cls):
        # Use the pre-loaded configuration
        model_config = LOAD_RIFE_NODE_CONFIG.get("model", {})
        precision_config = LOAD_RIFE_NODE_CONFIG.get("precision", {})

        # Provide sensible defaults if keys are missing in the config
        model_options = model_config.get("options", ["rife49_ensemble_True_scale_1_sim"])
        model_default = model_config.get("default", "rife49_ensemble_True_scale_1_sim")
        model_tooltip = model_config.get("tooltip", "Select a RIFE model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        return {
            "required": {
                "model": (model_options, {"default": model_default, "tooltip": model_tooltip}),
                "precision": (precision_options, {"default": precision_default, "tooltip": precision_tooltip}),
            }
        }

    RETURN_NAMES = ("rife_trt_model",)
    RETURN_TYPES = ("RIFE_TRT_MODEL",)
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load RIFE tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_rife_tensorrt_model"

    def load_rife_tensorrt_model(self, model, precision):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "rife")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

        # Build tensorrt model path with detailed naming
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = 1, 1, 1
        engine_min_h, engine_opt_h, engine_max_h = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        engine_min_w, engine_opt_w, engine_max_w = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/{model}.onnx"
                rife_logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                rife_logger.info(f"ONNX model found at: {onnx_model_path}")

            rife_logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16=True if precision == "fp16" else False,
                input_profile=[
                    {
                        "img0": [(engine_min_batch, engine_channel, engine_min_h, engine_min_w), (engine_opt_batch, engine_channel, engine_opt_h, engine_opt_w), (engine_max_batch, engine_channel, engine_max_h, engine_max_w)],
                        "img1": [(engine_min_batch, engine_channel, engine_min_h, engine_min_w), (engine_opt_batch, engine_channel, engine_opt_h, engine_opt_w), (engine_max_batch, engine_channel, engine_max_h, engine_max_w)],
                    }
                ],
            )
            e = time.time()
            rife_logger.info(f"Time taken to build: {(e-s)} seconds")

        rife_logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        return (engine,)

class RifeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame interpolation"}),
                "rife_trt_model": ("RIFE_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear CUDA cache after processing this many frames"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "tooltip": "Frame interpolation multiplier"}),
                "use_cuda_graph": ("BOOLEAN", {"default": True, "tooltip": "Use CUDA graph for better performance"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model loaded in memory after processing"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "tensorrt"
    OUTPUT_NODE=True

    def vfi(
        self,
        frames,
        rife_trt_model,
        clear_cache_after_n_frames=100,
        multiplier=2,
        use_cuda_graph=True,
        keep_model_loaded=False,
    ):
        B, H, W, C = frames.shape
        shape_dict = {
            "img0": {"shape": (1, 3, H, W)},
            "img1": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H, W)},
        }

        cudaStream = cuda.Stream()

        # Use the provided model directly
        engine = rife_trt_model
        logger(f"Using loaded TensorRT engine")

        # Activate and allocate buffers for the engine
        engine.activate()
        engine.allocate_buffers(shape_dict=shape_dict)

        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep):
            timestep_t = torch.tensor([timestep], dtype=torch.float32).to(get_torch_device())
            # s = time.time()
            output = engine.infer({"img0": frame_0, "img1": frame_1, "timestep": timestep_t}, cudaStream, use_cuda_graph)
            # e = time.time()
            # print(f"Time taken to infer: {(e-s)*1000} ms")

            result = output['output']
            return result

        result = generate_frames_rife(frames, clear_cache_after_n_frames, multiplier, return_middle_frame)
        out = postprocess_frames(result)

        if not keep_model_loaded:
            engine.reset()

        return (out,)


NODE_CLASS_MAPPINGS = {
    "RifeTensorrt": RifeTensorrt,
    "LoadRifeTensorrtModel": LoadRifeTensorrtModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RifeTensorrt": "⚡ Rife Tensorrt",
    "LoadRifeTensorrtModel": "Load Rife Tensorrt Model",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
