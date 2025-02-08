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


prompts = [
    "A hyper-realistic close-up of a human hand with futuristic cybernetic enhancements, glowing neon circuits, in cyberpunk style.",
    "A digital painting of a human hand adorned with steampunk gears and brass elements, intricate mechanical details.",
    "A watercolor illustration of a human hand with delicate floral patterns and pastel hues, soft and dreamy.",
    "A minimalist black and white sketch of a human hand in a meditative pose, emphasizing fluid lines and simplicity.",
    "A surreal image of a human hand emerging from swirling clouds, with a dreamlike and ethereal quality.",
    "A realistic portrait of a human hand holding a glowing orb, radiating mystical energy in a fantasy setting.",
    "A vibrant pop art depiction of a human hand with bold colors, geometric patterns, and retro comic style.",
    "A detailed ink drawing of a human hand decorated with intricate henna designs, capturing cultural artistry.",
    "A futuristic render of a human hand with holographic interfaces and digital data streams, sci-fi aesthetic.",
    "A stylized charcoal sketch of a human hand gracefully holding a vintage key, evoking mystery and nostalgia.",
    "A concept art piece featuring a human hand with bioluminescent markings, set in a futuristic jungle.",
    "An abstract painting of a human hand composed of swirling colors and dynamic brush strokes, expressive and energetic.",
    "A low-poly 3D render of a human hand in a modern minimalist style, sharp edges and clean lines.",
    "A surreal digital collage of a human hand merging with natural elements like leaves and branches.",
    "A high-contrast silhouette of a human hand set against a vibrant sunset, bold and dramatic.",
    "A delicate pastel drawing of a human hand with soft textures and gentle gradients, conveying warmth and serenity.",
    "A retro 80s neon style image of a human hand with glowing outlines and vibrant color palettes.",
    "A steampunk-inspired illustration of a human hand with mechanical joints and brass embellishments.",
    "A dreamy oil painting of a human hand reaching out towards a starry sky, filled with cosmic wonder.",
    "A dynamic motion blur capture of a human hand in mid-gesture, emphasizing movement and energy.",
    "A detailed anatomical drawing of a human hand, showcasing muscles and bone structure in a scientific style.",
    "A mystical depiction of a human hand casting magical spells with radiant energy and sparkles.",
    "A digital graffiti art piece featuring a human hand with urban street art influences and bold spray paint effects.",
    "A futuristic augmented reality interface held by a human hand, with transparent layers and interactive elements.",
    "A cosmic-themed image of a human hand with galaxies swirling in the palm, blending space and art.",
    "A romantic vintage illustration of a human hand holding a delicate rose, rendered in soft sepia tones.",
    "A playful cartoon-style drawing of a human hand giving a thumbs-up, with exaggerated features and bright colors.",
    "A rugged, weathered human hand gripping an ancient relic, evoking a sense of adventure and mystery.",
    "A magical realism scene of a human hand releasing a flock of ethereal butterflies, blending fantasy and nature.",
    "A highly detailed digital portrait of a human hand with luminous tattoos, merging art and technology.",
    "A high-fidelity render of a human hand with realistic skin textures and intricate vein details.",
    "A surreal scene of a human hand intertwined with wild vines and blooming flowers, symbolizing nature and growth.",
    "A cyberpunk illustration of a human hand with neon-lit cybernetic implants, set against a futuristic cityscape.",
    "A soft focus, vintage-style photo of a human hand delicately holding a timeworn locket.",
    "A vibrant, abstract expressionist painting of a human hand with energetic strokes and bold splashes of color.",
    "A mystical, alchemical illustration of a human hand with arcane symbols and ancient runes glowing softly.",
    "A futuristic sci-fi concept art of a human hand interacting with a digital hologram of the Earth.",
    "A dreamy pastel doodle of a human hand with a cascade of falling cherry blossoms.",
    "A realistic charcoal sketch of a human hand holding a tiny, intricate mechanical insect.",
    "A cosmic surrealist depiction of a human hand cradling a miniature solar system.",
    "A vibrant, colorful pop surrealism art piece of a human hand with melting, liquid-like textures.",
    "A clean, vector illustration of a human hand in a gesture of peace, with crisp lines and minimal detail.",
    "A mystical digital painting of a human hand emitting radiant, cosmic energy, with a star-filled background.",
    "A detailed pencil drawing of a human hand entwined with ivy and ancient script, evoking an enchanted forest.",
    "A futuristic cyborg hand with transparent skin and visible electronic circuits, rendered in 3D.",
    "A romantic, soft-focus painting of a human hand gently cradling a glowing heart-shaped object.",
    "A surrealist image of a human hand transforming into swirling patterns of fractal geometry.",
    "A high-definition render of a human hand with realistic weathered skin and subtle imperfections.",
    "A dynamic urban illustration of a human hand with graffiti textures and street art flair.",
    "A magical digital collage of a human hand adorned with celestial constellations and nebula clouds.",
    "A modern, minimalist design featuring a human hand icon in a simple, flat style with bold colors.",
    "A dramatic chiaroscuro painting of a human hand in deep shadow and striking highlights.",
    "A vibrant fantasy illustration of a human hand wielding a sword made of pure energy.",
    "A delicate, nature-inspired sketch of a human hand merging with a cascade of autumn leaves.",
    "A futuristic neon digital art piece of a human hand interacting with augmented reality displays.",
    "A textured oil painting of a human hand holding a small, glowing crystal, rich in detail and depth.",
    "A surreal digital illustration of a human hand with an explosion of vibrant, abstract shapes.",
    "A whimsical cartoon of a human hand with expressive features and playful gestures.",
    "A refined, classical portrait of a human hand with realistic shading and soft background gradients.",
    "A modern abstract composition featuring a fragmented human hand, with geometric patterns.",
    "A highly stylized digital artwork of a human hand with intricate tribal markings and bold colors.",
    "A dreamy, ethereal image of a human hand surrounded by floating orbs of light in a dark void.",
    "A striking, ultra-realistic render of a human hand with water droplets glistening on the skin.",
    "A fantastical illustration of a human hand morphing into a swirling vortex of energy.",
    "A conceptual art piece of a human hand fused with digital elements, symbolizing the blend of humanity and technology.",
    "A vibrant, multi-colored digital painting of a human hand with a mosaic of dynamic patterns.",
    "A soft, impressionist painting of a human hand lightly touching a delicate, glowing flower.",
    "A dynamic, high-energy concept art piece featuring a human hand with fluid, flowing motion.",
    "A surreal 3D render of a human hand emerging from a portal of vibrant light and color.",
    "A minimalistic line art drawing of a human hand interwoven with abstract patterns and symbols.",
    "A dreamy, celestial-themed image of a human hand with stardust and cosmic elements swirling around.",
    "A retro-futuristic digital illustration of a human hand with bold neon accents and vintage computer graphics.",
    "A detailed digital sketch of a human hand holding a delicate, ancient scroll, evoking mystery and wisdom.",
    "A vibrant, energetic graffiti art style image of a human hand with splashes of color and urban vibe.",
    "A high-resolution portrait of a human hand with realistic textures and lifelike details, captured in macro photography style.",
    "A surreal, double-exposure style image of a human hand merging with a forest landscape.",
    "A modern, flat design illustration of a human hand in a simple gesture, with clean lines and bold color blocks.",
    "A fantastical concept art of a human hand with swirling cosmic energy and nebula-inspired hues.",
    "A dramatic, low-key lighting image of a human hand set against a dark, moody background.",
    "A delicate, intricately detailed ink drawing of a human hand with ornate filigree patterns.",
    "A futuristic digital art piece featuring a human hand manipulating glowing holographic elements.",
    "A vibrant, psychedelic illustration of a human hand with swirling, kaleidoscopic patterns.",
    "A surreal, atmospheric painting of a human hand emerging from a misty, enchanted forest.",
    "A minimalist, monochrome vector illustration of a human hand in a dynamic pose.",
    "A hyper-realistic digital painting of a human hand with extremely detailed skin textures and fine lines.",
    "A playful, cartoonish depiction of a human hand holding a whimsical, oversized prop.",
    "A steampunk-inspired render of a human hand with mechanical gears and vintage clockwork aesthetics.",
    "A dreamy, soft pastel illustration of a human hand gently cradling a glowing orb of light.",
    "A bold, expressionist artwork of a human hand with vivid, contrasting colors and dynamic brush strokes.",
    "A serene, meditative image of a human hand with subtle, ethereal light patterns.",
    "A striking, realistic render of a human hand with dramatic lighting and intricate details, emphasizing texture.",
    "A magical, fantasy illustration of a human hand conjuring shimmering spells and radiant energy.",
    "A modern digital illustration of a human hand integrated with abstract geometric shapes.",
    "A high-detail, photorealistic macro shot of a human hand with dew drops on the skin.",
    "A conceptual, mixed media collage featuring a human hand with elements of digital glitch art.",
    "A vibrant, color-saturated portrait of a human hand adorned with futuristic tattoos and glowing accents.",
    "A surreal image of a human hand dissolving into a cascade of digital particles, blending reality and fantasy.",
    "A delicate watercolor painting of a human hand with soft gradients and subtle textures.",
    "A highly stylized digital art piece featuring a human hand with angular, cubist influences.",
    "A modern abstract illustration of a human hand with fragmented, mosaic-like details.",
    "A dramatic, noir-style black and white image of a human hand emerging from shadowy depths.",
    "A high-energy, futuristic scene of a human hand interacting with floating holographic interfaces.",
    "A vibrant, dynamic painting of a human hand with swirling patterns of energy and light.",
    "A realistic digital illustration of a human hand with lifelike shading and meticulously rendered details.",
    "A surreal, dreamlike image of a human hand morphing into intricate, swirling fractal patterns.",
    "A whimsical, animated-style drawing of a human hand with exaggerated features and vibrant colors.",
    "A futuristic cybernetic hand with illuminated circuits and metallic textures, rendered in 3D.",
    "A soft, impressionistic painting of a human hand reaching out towards a field of blooming flowers.",
    "A highly detailed ink sketch of a human hand with ornate, decorative line work and delicate shading.",
    "A vibrant, abstract expressionist depiction of a human hand with explosive bursts of color.",
    "A realistic, high-contrast image of a human hand in a dramatic, action-packed pose.",
    "A surreal digital art piece of a human hand entwined with floating, luminous ribbons of light.",
    "A modern, minimalist icon-style illustration of a human hand, using simple lines and bold shapes.",
    "A retro-inspired digital illustration of a human hand with a distressed, vintage texture.",
    "A detailed, photorealistic rendering of a human hand holding a delicate butterfly, emphasizing natural beauty.",
    "A conceptual, futuristic image of a human hand fused with intricate digital circuitry and glowing elements.",
    "A soft pastel illustration of a human hand gently cradling a cluster of tiny, sparkling stars.",
    "A dynamic, motion-blurred render of a human hand in mid-gesture, capturing a moment of fluid movement.",
    "A surreal, dreamlike composition featuring a human hand dissolving into a swirl of vivid colors.",
    "A highly detailed digital painting of a human hand with luminous, otherworldly tattoos and markings.",
    "A minimalist line drawing of a human hand, balanced and elegant, set against a solid colored background.",
    "A vibrant, pop art style image of a human hand with bold outlines and a striking, colorful palette.",
    "A realistic digital render of a human hand with hyper-detailed textures, emphasizing every crease and line.",
    "A magical, fantasy-inspired illustration of a human hand summoning shimmering, ethereal energy.",
    "A high-contrast black and white photograph-like image of a human hand with deep shadows and bright highlights.",
    "A futuristic concept art piece featuring a human hand integrated with glowing digital elements.",
    "A serene, abstract watercolor of a human hand blending into a wash of soft, harmonious colors.",
    "A detailed, surreal digital illustration of a human hand morphing into a cascade of vibrant fractal shapes.",
    "A modern vector art depiction of a human hand with clean lines and a bold, simplified style.",
    "A vibrant, dynamic digital painting of a human hand set against a background of swirling cosmic energy.",
    "A dreamy, ethereal illustration of a human hand with delicate, transparent overlays and soft glows.",
    "A high-definition, photorealistic image of a human hand with meticulously rendered skin details.",
    "A retro-futuristic digital collage of a human hand overlaid with vintage computer graphics and neon accents.",
    "A mystical, surreal depiction of a human hand surrounded by floating, illuminated symbols and glyphs.",
    "A conceptual artwork of a human hand merging with digital elements, set in a futuristic urban environment.",
    "A highly detailed, realistic pencil sketch of a human hand with expressive shading and intricate textures.",
    "A vibrant, futuristic illustration of a human hand with geometric patterns and bright neon colors.",
    "A surreal, dreamlike image of a human hand releasing a cascade of sparkling, digital particles.",
    "A modern, minimalist digital art piece featuring a human hand with crisp lines and bold, flat colors.",
    "A soft, atmospheric painting of a human hand with a gentle glow, set against a twilight sky.",
    "A dynamic, high-energy 3D render of a human hand in an action-packed gesture with motion blur effects.",
    "A detailed, intricate digital illustration of a human hand with ornate, decorative patterns.",
    "A surreal, conceptual image of a human hand disintegrating into a shower of vibrant, glowing pixels.",
    "A realistic digital portrait of a human hand with a fine balance of light and shadow, emphasizing texture.",
    "A modern abstract composition featuring a human hand with fragmented, collage-like elements.",
    "A bold, expressionist digital painting of a human hand with vibrant color contrasts and dynamic strokes.",
    "A mystical, fantastical illustration of a human hand holding a glowing, enchanted artifact.",
    "A soft, ethereal watercolor of a human hand merging with swirling, translucent layers of color.",
    "A futuristic digital artwork of a human hand with luminous, neon accents and sleek metallic textures.",
    "A retro-inspired illustration of a human hand rendered in a vintage comic book style with bold lines.",
    "A surreal digital art piece of a human hand enveloped in swirling, cosmic clouds and stardust.",
    "A minimalist, black and white illustration of a human hand with elegant simplicity and subtle details.",
    "A vibrant, futuristic rendering of a human hand interacting with dynamic holographic displays.",
    "A dynamic, motion-captured digital illustration of a human hand in a powerful, expressive gesture.",
    "A highly detailed, photorealistic macro shot of a human hand with delicate textures and lifelike precision.",
    "A surreal, dreamlike composition of a human hand emerging from an explosion of vivid, abstract light.",
    "A vibrant, digital illustration of a human hand holding a spark of creative energy, surrounded by bursts of color.",
    "A mesmerizing, abstract rendering of a human hand with swirling neon patterns and fractal details.",
    "A classical oil painting style depiction of a human hand in a graceful, elongated pose, rich in texture.",
    "A high-contrast silhouette of a human hand against a luminous full moon, evoking mystery and intrigue.",
    "A modern, flat design graphic of a human hand with sharp geometric shapes and a bold, monochrome palette.",
    "A delicate, vintage-style etching of a human hand with fine cross-hatching and timeless detail.",
    "A surreal digital montage of a human hand blending into a collage of urban textures and graffiti elements.",
    "A dreamy, pastel-toned illustration of a human hand softly illuminated by gentle, ambient light.",
    "A detailed, 3D modeled image of a human hand with realistic textures and subtle, lifelike imperfections.",
    "A cosmic, psychedelic vision of a human hand with swirling galactic patterns and vibrant cosmic hues.",
    "A refined, minimalist sketch of a human hand using only a few continuous, elegant lines.",
    "A fantastical depiction of a human hand with translucent skin revealing swirling, iridescent inner energy.",
    "A dynamic digital painting of a human hand captured in mid-motion, with streaks of color trailing behind.",
    "A high-definition render of a human hand with hyper-detailed pores and lifelike shadow play.",
    "A mysterious, dark-themed illustration of a human hand emerging from fog, with eerie, soft lighting.",
    "A vibrant, graffiti-inspired digital mural of a human hand with explosive color splashes and urban grit.",
    "A serene, meditative depiction of a human hand resting on a bed of soft, illuminated petals.",
    "A surreal concept art piece of a human hand that appears to be dissolving into abstract digital code.",
    "A futuristic digital painting of a human hand with sleek, metallic textures and holographic glows.",
    "A romantic, softly lit illustration of a human hand intertwined with delicate vines and blooming roses.",
    "A bold, abstract art piece of a human hand composed entirely of dynamic, intersecting lines and shapes.",
    "A detailed, realistic sketch of a human hand using cross-hatching techniques for a vintage feel.",
    "A vibrant, animated-style digital illustration of a human hand bursting with colorful, dynamic energy.",
    "A surreal, ethereal image of a human hand with light trails and a backdrop of shifting, prismatic hues.",
    "A modern digital render of a human hand with crisp, clean edges and a flat, vector aesthetic.",
    "A richly textured oil painting of a human hand with deep, dramatic contrasts and expressive brush strokes.",
    "A dreamy digital collage of a human hand interlaced with delicate, transparent geometric patterns.",
    "A futuristic neon illustration of a human hand reaching out, with vivid, glowing outlines and digital sparks.",
    "A classical, Renaissance-style drawing of a human hand with meticulous attention to anatomical detail.",
    "A surreal, multi-exposure image of a human hand blended with scenes of nature and urban decay.",
    "A minimalist, contemporary line art piece of a human hand with soft curves and gentle gradients.",
    "A vibrant, digital abstract of a human hand with splattered paint textures and energetic brush strokes.",
    "A whimsical, storybook illustration of a human hand holding a tiny, enchanted lantern, glowing softly.",
    "A bold, futuristic concept art of a human hand interacting with virtual, three-dimensional data streams.",
    "A high-contrast, cinematic render of a human hand with dramatic shadows and vibrant highlight accents.",
    "A delicate, hand-drawn ink illustration of a human hand with fine, intricate detailing and soft shading.",
    "A surreal digital fantasy of a human hand merging seamlessly with an explosion of colorful digital fractals.",
    "A modern minimalist depiction of a human hand in an abstract, deconstructed form with vibrant accents.",
    "A striking, ultra-realistic digital painting of a human hand with lifelike textures and high dynamic range.",
    "A mystical, atmospheric illustration of a human hand cradling a luminous, otherworldly artifact.",
    "A dynamic, expressive digital sketch of a human hand with fluid lines and energetic, vibrant strokes.",
    "A serene, nature-inspired painting of a human hand intertwined with soft, ethereal watercolor landscapes.",
    "A conceptual digital artwork of a human hand with layers of transparent, overlapping geometric shapes.",
    "A futuristic, hyper-stylized digital render of a human hand with radiant neon effects and abstract overlays."
]

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
    
    # Create prompt mapping and output directories first
    for idx, prompt in enumerate(prompts, 1):
        folder_name = f"{idx:03d}_{sanitize_folder_name(prompt)}"
        output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        prompt_mapping[folder_name] = prompt
    
    # Save prompt mapping before starting generation
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