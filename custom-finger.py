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

import random

# Expanded list of 100 artistic styles for hand depictions
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
    "A delicate pastel drawing of a human hand with ethereal soft gradients",
    "A charcoal sketch of a human hand with subtle cross-hatching and dynamic contours",
    "A vintage sepia-toned photograph of a human hand, evoking nostalgic elegance",
    "A digital matte painting of a human hand emerging from a stormy, epic sky",
    "A modern abstract digital art piece featuring a human hand with fragmented shapes",
    "An art nouveau illustration of a human hand with flowing lines and floral motifs",
    "A vibrant graffiti style depiction of a human hand with bold, urban textures",
    "A delicate pastel and ink drawing of a human hand with intricate lace patterns",
    "A fantastical digital rendering of a human hand with bioluminescent markings",
    "A dramatic chiaroscuro painting of a human hand set against deep shadows",
    "A luminous, ethereal digital painting of a human hand with magical glow effects",
    "A retro-futuristic illustration of a human hand with vintage sci-fi elements",
    "A cosmic surrealist artwork of a human hand with swirling galaxies and nebulae",
    "A dreamy, soft-focus photograph of a human hand with ambient light flares",
    "A crisp, vector-style illustration of a human hand with minimalist design",
    "A baroque-inspired painting of a human hand with rich textures and opulent details",
    "A high-dynamic-range digital render of a human hand with hyper-realistic textures",
    "A surreal, multi-exposure image of a human hand intertwined with nature",
    "A vibrant digital collage of a human hand with layers of abstract patterns",
    "A futuristic cybernetic illustration of a human hand with glowing circuits",
    "A soft, impressionistic painting of a human hand in a serene, pastoral setting",
    "An avant-garde digital artwork of a human hand with chaotic, expressive brushstrokes",
    "A modern pop surrealism depiction of a human hand with exaggerated, whimsical features",
    "A richly textured oil painting of a human hand with deep, moody shadows",
    "A dynamic, kinetic art-inspired illustration of a human hand in mid-motion",
    "A dreamlike, fantastical image of a human hand with shimmering, magical elements",
    "A sleek, minimalistic 3D render of a human hand with sharp edges and polished surfaces",
    "A glitch art-inspired digital image of a human hand with pixelated distortions",
    "A vibrant, expressive brush-stroke painting of a human hand in a modern art style",
    "A surrealist image of a human hand morphing into a cascade of abstract shapes",
    "A neon-drenched futuristic scene of a human hand set against a backdrop of digital rain",
    "A high-resolution CGI render of a human hand with ultra-realistic textures",
    "A mystical illustration of a human hand entwined with magical runes and symbols",
    "A soft pastel drawing of a human hand surrounded by floating, delicate orbs",
    "An atmospheric foggy scene featuring a human hand emerging from the mist",
    "A richly detailed ink and watercolor hybrid of a human hand with elaborate tattoos",
    "A digital art piece featuring a human hand with cyberpunk neon glow and digital glitches",
    "A vibrant abstract painting of a human hand with splashes of bold color",
    "A serene, meditative depiction of a human hand in a calm, monochromatic setting",
    "A highly stylized pop art version of a human hand with comic book flair",
    "A meticulously detailed graphite drawing of a human hand in dynamic pose",
    "A contemporary digital sculpture rendering of a human hand with smooth surfaces",
    "A vintage comic strip illustration of a human hand in an action scene",
    "A mysterious, shadowy portrait of a human hand with selective lighting",
    "A playful, cartoonish drawing of a human hand with exaggerated features",
    "A surreal, Dali-inspired melting clock effect integrated into a human hand",
    "A vibrant, modern street art mural of a human hand in an urban setting",
    "A futuristic holographic projection of a human hand in a digital space",
    "A soft-focus, dreamlike illustration of a human hand with blurred, gentle edges",
    "A richly textured impasto painting of a human hand with thick, expressive brushstrokes",
    "A digital illustration of a human hand with fractal patterns and kaleidoscopic colors",
    "A luminous, ethereal scene of a human hand surrounded by cosmic dust and starlight",
    "A cyberpunk-inspired graphic novel panel featuring a human hand with glowing accents",
    "A surreal, double-exposure image combining a human hand with natural landscapes",
    "A modern minimalist line art drawing of a human hand with a single continuous stroke",
    "A futuristic biomechanical rendering of a human hand integrated with organic circuits",
    "A vibrant, textured digital painting of a human hand with explosive color contrasts",
    "A haunting, monochromatic photograph-like image of a human hand with stark contrasts",
    "A digital mosaic of a human hand constructed from intricate, tiny geometric shapes",
    "A conceptual, abstract expressionist depiction of a human hand with chaotic forms",
    "A detailed pen and ink illustration of a human hand with ornate, swirling details",
    "A high-speed motion capture image of a human hand in dynamic, blurred motion",
    "A mystical, otherworldly digital painting of a human hand with shimmering auras",
    "A crisp, hyper-detailed technical drawing of a human hand with anatomical precision",
    "A layered digital collage of a human hand interwoven with urban and natural motifs",
    "A vibrant, modern vector art rendition of a human hand with bold lines and flat colors",
    "A dreamy, ethereal watercolor painting of a human hand with soft pastel tones",
    "A stark, minimalist digital composition featuring a solitary human hand silhouette",
    "A retro-inspired pixel art depiction of a human hand with a nostalgic vibe",
    "A high-energy, graffiti-infused digital art piece of a human hand in action",
    "A surreal, imaginative illustration of a human hand that defies natural form",
    "A visually striking digital portrait of a human hand with hyper-saturated colors",
    "A futuristic digital render of a human hand emerging from a data matrix",
    "A textured, expressive oil painting of a human hand with dramatic brushwork",
    "A cinematic, film noir style image of a human hand with deep shadows and high contrast",
    "A luminous, glowing digital illustration of a human hand with a halo of light",
    "A vibrant, abstract geometric composition centered on a human hand silhouette",
    "A moody, atmospheric digital painting of a human hand enveloped in twilight hues",
    "A richly detailed baroque portrait of a human hand with intricate ornamental details",
    "A stylized, futuristic illustration of a human hand with digital glitch effects",
    "A fantastical, dreamlike digital artwork of a human hand adorned with surreal motifs"
]

# Expanded list of finger variations with 13 different configurations
finger_variations = [
    "with no fingers",
    "with one oversized finger",
    "with two delicate fingers",
    "with three uniquely shaped fingers",
    "with four perfectly arranged fingers",
    "with five natural fingers",
    "with six elongated fingers",
    "with seven gracefully curved fingers",
    "with eight slender fingers",
    "with nine surreal digits",
    "with ten bizarrely arranged fingers",
    "with eleven extra digits",
    "with twelve whimsical fingers"
]

# Additional artistic conditions or moods
conditions = [
    "in vibrant neon colors",
    "under dramatic lighting",
    "set against a cosmic background",
    "with intricate details and subtle shadows",
    "in a mysterious, foggy atmosphere"
]


# Generate all possible combinations (Total combinations: 50 x 13 x 5 = 3250)
all_prompts = [
    f"{style} {finger}, {condition}."
    for style in styles
    for finger in finger_variations
    for condition in conditions
]

# Randomly select 200 unique prompts from the complete set
selected_prompts = random.sample(all_prompts, 200)

# Print the 200 generated prompts
for prompt in selected_prompts:
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
        try:
            # Get next task from queue
            task = prompt_queue.get()
            if task is None:  # Poison pill to stop the process
                break
                
            prompt, output_dir, start_idx = task
            
            print(f"GPU {gpu_id}: Processing prompt: {prompt}")
            print(f"GPU {gpu_id}: Output directory: {output_dir}")
            
            # Verify output directory exists
            if not os.path.exists(output_dir):
                print(f"GPU {gpu_id}: Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            print(f"GPU {gpu_id}: Memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f}GB")
            
            # Store images in memory
            images_to_save = []
            
            for i in range(25):  # Generate 25 images per prompt
                seed = random.randint(0, MAX_SEED)
                generator = torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed)
                
                print(f"GPU {gpu_id}: Generating image {i+1}/25 with seed: {seed}")
                
                # Keep track of the last image in the iteration
                last_image = None
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
                        # Only keep the last image (fully denoised)
                        last_image = img
                
                # Store only the final denoised image
                if last_image is not None:
                    images_to_save.append((last_image, f"image_{i+1}_seed_{seed}.png"))
                    print(f"GPU {gpu_id}: Generated image {i+1}")
                else:
                    print(f"GPU {gpu_id}: Warning - No image generated for iteration {i+1}")
            
            # Batch save all final images for this prompt
            print(f"GPU {gpu_id}: Attempting to save {len(images_to_save)} images...")
            for img, filename in images_to_save:
                output_path = os.path.join(output_dir, filename)
                try:
                    img.save(output_path)
                    print(f"GPU {gpu_id}: Successfully saved {filename}")
                except Exception as e:
                    print(f"GPU {gpu_id}: Error saving {filename}: {str(e)}")
            
            print(f"GPU {gpu_id}: Completed saving batch of {len(images_to_save)} images to {output_dir}")
            
            # Clear memory
            images_to_save.clear()
            torch.cuda.empty_cache()
            print(f"GPU {gpu_id}: Memory after cleanup: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f}GB")
            
        except Exception as e:
            print(f"GPU {gpu_id}: Error processing task: {str(e)}")

def generate_images():
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    # Create main output directory and prompt mapping in a single pass
    base_output_dir = "generated_hands"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create prompt mapping and directories in a single pass
    prompt_mapping = {
        f"{idx:03d}_{sanitize_folder_name(prompt)}": prompt 
        for idx, prompt in enumerate(selected_prompts, 1)
    }
    
    # Create directories and save mapping
    for folder_name in prompt_mapping.keys():
        os.makedirs(os.path.join(base_output_dir, folder_name), exist_ok=True)
    
    mapping_file = os.path.join(base_output_dir, "prompt_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(prompt_mapping, f, indent=2)
    
    # Create a queue for distributing work
    prompt_queue = Queue()
    
    # Add all prompts to the queue
    for idx, (folder_name, prompt) in enumerate(prompt_mapping.items(), 1):
        output_dir = os.path.join(base_output_dir, folder_name)
        prompt_queue.put((prompt, output_dir, idx))
    
    # Add poison pills to stop processes
    for _ in range(num_gpus):
        prompt_queue.put(None)
    
    # Start processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=generate_images_on_gpu, args=(
            gpu_id, 
            prompt_queue,
            768,  # width
            768,  # height
            3.5,  # guidance_scale
            20,   # num_inference_steps
        ))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"All images generated. Prompt mapping saved to {mapping_file}")

if __name__ == "__main__":
    generate_images()