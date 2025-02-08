import numpy as np
import random
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL
from live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images
from PIL import Image
import os
import json
import re
from multiprocessing import Process, Queue
import torch.multiprocessing as mp

MAX_SEED = np.iinfo(np.int32).max


import random

# Expanded list of artistic styles (20 different styles)
styles = [
    "A hyper-realistic close-up of a human hand with futuristic cybernetic enhancements",
    "A digital painting of a human hand adorned with steampunk gears and brass elements",
    "A watercolor illustration of a human hand with delicate floral patterns and pastel hues",
    "A minimalist black and white sketch of a human hand in a meditative pose",
    "A surreal image of a human hand emerging from swirling clouds",
    "A vibrant pop art depiction of a human hand with bold colors and geometric patterns",
    "A detailed ink drawing of a human hand decorated with intricate henna designs",
    "A futuristic render of a human hand with holographic interfaces and digital data streams",
    "A dreamy oil painting of a human hand reaching out towards a starry sky",
    "A gritty urban street art style image of a human hand with spray paint splatters",
    "A photorealistic macro shot of a human hand with dew drops and detailed skin textures",
    "A stylized digital sketch of a human hand with abstract geometric overlays",
    "A classic Renaissance-style drawing of a human hand with delicate shading and composition",
    "A surreal digital collage of a human hand blended with cosmic and nature elements",
    "A vibrant neon-lit cyberpunk illustration of a human hand interacting with holograms",
    "An expressive impressionist painting of a human hand with fluid brush strokes and vibrant colors",
    "A high-contrast, black and white photograph of a human hand with dramatic shadows",
    "A steampunk-inspired illustration of a human hand fused with mechanical parts and gears",
    "A futuristic sci-fi concept art of a human hand manipulating digital interfaces",
    "A delicate pastel drawing of a human hand with ethereal soft gradients"
]

# Variations in the number of fingers
finger_variations = [
    "with four fingers",
    "with five fingers",
    "with six elongated fingers",
    "with seven gracefully curved fingers",
    "with eight slender fingers"
]

# Additional artistic conditions or moods
conditions = [
    "in vibrant neon colors",
    "under dramatic lighting",
    "set against a cosmic background",
    "with intricate details and subtle shadows",
    "in a mysterious, foggy atmosphere"
]

# Generate all possible combinations from the lists
all_prompts = [
    f"{style} {finger}, {condition}."
    for style in styles
    for finger in finger_variations
    for condition in conditions
]

# Randomly select 200 unique prompts from the total combinations
prompts = random.sample(all_prompts, 250)

# (Optional) For example, print the first five prompts:
for prompt in prompts[:5]:
    print(prompt)

# Remove global model loading - we'll load per GPU instead
def setup_pipeline(gpu_id):
    """Initialize the pipeline on a specific GPU"""
    device = f"cuda:{gpu_id}"
    dtype = torch.bfloat16
    
    print(f"Initializing models on GPU {gpu_id}")
    taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
    good_vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device)
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype, vae=taef1).to(device)
    
    pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
    
    return pipe, good_vae

def sanitize_folder_name(prompt):
    # Convert prompt to a valid folder name by keeping only alphanumeric chars and replacing spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)
    sanitized = sanitized.strip().replace(' ', '_')
    # Limit length and convert to lowercase
    return sanitized[:50].lower()

def generate_images_on_gpu(gpu_id, prompt_queue, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28):
    """Generate images on a specific GPU"""
    torch.cuda.set_device(gpu_id)  # Explicitly set the GPU
    print(f"GPU {gpu_id}: Starting with {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f}GB total memory")
    
    pipe, good_vae = setup_pipeline(gpu_id)
    
    while True:
        # Get next task from queue
        task = prompt_queue.get()
        if task is None:  # Poison pill to stop the process
            break
            
        prompt, output_dir, start_idx = task
        
        print(f"GPU {gpu_id}: Memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f}GB")
        
        for i in range(25):  # Generate 10 images per prompt
            seed = random.randint(0, MAX_SEED)
            generator = torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed)
            
            print(f"GPU {gpu_id}: Generating image {i+1}/10 with seed: {seed}")
            
            for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="pil",
                    good_vae=good_vae,
                ):
                    output_path = os.path.join(output_dir, f"image_{i+1}_seed_{seed}.png")
                    img.save(output_path)
                    print(f"GPU {gpu_id}: Saved image {i+1} to {output_path}")
        
        # Clear CUDA cache after each prompt
        torch.cuda.empty_cache()
        print(f"GPU {gpu_id}: Memory after cleanup: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f}GB")

def generate_images():
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    # Create main output directory
    base_output_dir = "generated_hands"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Dictionary to store prompt mapping
    prompt_mapping = {}
    
    # Create prompt mapping and directories first
    for idx, prompt in enumerate(prompts, 1):
        folder_name = f"{idx:03d}_{sanitize_folder_name(prompt)}"
        output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        prompt_mapping[folder_name] = prompt
    
    # Save prompt mapping before generation
    mapping_file = os.path.join(base_output_dir, "prompt_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(prompt_mapping, f, indent=2)
    print(f"Prompt mapping saved to {mapping_file}")
    
    # Create a queue for distributing work
    prompt_queue = Queue()
    
    # Add all prompts to the queue
    for idx, prompt in enumerate(prompts, 1):
        folder_name = f"{idx:03d}_{sanitize_folder_name(prompt)}"
        output_dir = os.path.join(base_output_dir, folder_name)
        prompt_queue.put((prompt, output_dir, idx))
    
    # Add poison pills to stop processes
    for _ in range(num_gpus):
        prompt_queue.put(None)
    
    # Start processes for each GPU with reduced dimensions for memory constraints
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=generate_images_on_gpu, args=(
            gpu_id, 
            prompt_queue,
            768,  # reduced width
            768,  # reduced height
            3.5,  # guidance_scale
            20,   # reduced num_inference_steps
        ))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"All images generated. Prompt mapping saved to {mapping_file}")

if __name__ == "__main__":
    generate_images()