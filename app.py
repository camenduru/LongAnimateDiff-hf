import os
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint


pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
inference_config_path = "configs/inference/long-inference.yaml"

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

examples = [
    # 12-EpicRealism
    [
        "epiCRealismNaturalSin.safetensors", 
        "photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        512, 512, 32, "1490157606650685400"
    ],
    # 2-EpicRealism
    [
        "epiCRealismNaturalSin.safetensors", 
        "a young man is dancing in a paris nice street",
        "wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation render, illustration, deformed, distorted, disfigured, doll, poorly drawn, bad anatomy, wrong anatomy deformed, naked, nude, breast (worst quality low quality: 1.4)",
        512, 512, 32, "1"
    ],
    # 3-EpicRealism
    [
        "epiCRealismNaturalSin.safetensors", 
        "photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        512, 512, 32, "13100322578370451493"
    ]
]
print(f"### Cleaning cached examples ...")
os.system(f"rm -rf gradio_cached_examples/")


class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir      = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir                = os.path.join(self.basedir, "samples")
        os.makedirs(self.savedir, exist_ok=True)

        
        self.selected_base_model    = None
        self.selected_motion_module = None
        
        self.refresh_motion_module()
        self.refresh_personalized_model()
        
        # config models
        self.inference_config      = OmegaConf.load(inference_config_path)

        self.tokenizer             = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.vae                   = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda()
        self.text_encoder          = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").cuda()
        self.unet                  = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()
        self.base_model_list = ['epiCRealismNaturalSin.safetensors']
        self.motion_module_list = ['lt_long_mm_32_frames.ckpt']

        print(self.base_model_list[0])
        self.update_base_model(self.base_model_list[0])
        self.update_motion_module(self.motion_module_list[0])


        
        
    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        base_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.base_model_list = [os.path.basename(p) for p in base_model_list]


    def update_base_model(self, base_model_dropdown):
        self.selected_base_model = base_model_dropdown
        
        base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
        base_model_state_dict = {}
        with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
            for key in f.keys(): base_model_state_dict[key] = f.get_tensor(key)
                
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
        self.vae.load_state_dict(converted_vae_checkpoint)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)

    def update_motion_module(self, motion_module_dropdown):
        self.selected_motion_module = motion_module_dropdown
        
        motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
        motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
        _, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0
    
    
    def animate(
        self,
        # base_model_dropdown,
        prompt_textbox,
        negative_prompt_textbox,
        width_slider,
        height_slider,
        video_length,
        seed_textbox,
    ):
        # if base_model_dropdown != self.selected_base_model: self.update_base_model(base_model_dropdown)
        # if motion_module_dropdown != self.selected_motion_module: self.update_motion_module(motion_module_dropdown)
        
        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")
        
        if int(seed_textbox) > 0: seed = int(seed_textbox)
        else: seed = random.randint(1, 1e16)
        torch.manual_seed(int(seed))
        
        assert seed == torch.initial_seed()
        print(f"### seed: {seed}")
        
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        
        sample = pipeline(
            prompt_textbox,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = 25,
            guidance_scale      = 8.,
            width               = width_slider,
            height              = height_slider,
            video_length        = video_length,
            generator           = generator,
        ).videos

        save_sample_path = os.path.join(self.savedir, f"sample.mp4")
        save_videos_grid(sample, save_sample_path)
    
        json_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "width": width_slider,
            "height": height_slider,
            "seed": seed,
            "base_model": base_model_dropdown,
        }
        return save_sample_path, json_config
        
print(f'gradio version is {gr.__version__}')
controller = AnimateController()



def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # [LongAnimateDiff](https://github.com/Lightricks/LongAnimateDiff)
            [Sapir Weissbuch](https://github.com/SapirW), [Naomi Ken Korem](https://github.com/Naomi-Ken-Korem), [Daniel Shalem](https://github.com/dshalem), [Yoav HaCohen](https://github.com/yoavhacohen) | Lightricks Research
            """
        )
        gr.Markdown(
            """
            ### Quick Start
            1. Select desired `Base DreamBooth Model`.
            2. Provide `Prompt` and `Negative Prompt` for each model. You are encouraged to refer to each model's webpage on CivitAI to learn how to write prompts for them. Below are the DreamBooth models in this demo. Click to visit their homepage.
                - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                - [`epiCRealismNatural.safetensors`](https://civitai.com/models/25694/epicrealism)
            3. Select 'Length' to set the length of the generated video. 
               (When you are working with ComfyUI try all possible length, with different motion_scale)
            4. Click `Generate`, wait for ~2 min, and enjoy. 
            5. In order to effectively utilize 'lt_long_mm_16_64_frames' model, it is highly recommended to use the ComfyUI interface, which enables to easily increase 'motion_scale' parameter and facilitates using the model in a video-to-video context.
            """
        )
        with gr.Row():
            with gr.Column():
                # base_model_dropdown     = gr.Dropdown( label="Base DreamBooth Model", choices=controller.base_model_list,    value=controller.base_model_list[0],    interactive=True )
                # motion_module_dropdown  = gr.Dropdown( label="Motion Module",  choices=controller.motion_module_list, value=controller.motion_module_list[0], interactive=True )

                # base_model_dropdown.change(fn=controller.update_base_model,       inputs=[base_model_dropdown],    outputs=[base_model_dropdown])
                # motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])

                prompt_textbox          = gr.Textbox( label="Prompt",          lines=3 )
                negative_prompt_textbox = gr.Textbox( label="Negative Prompt", lines=3, value="worst quality, low quality, nsfw, logo")
                video_length = gr.Slider(  label="Length", value=32, minimum=16, maximum=32, step=4 )


                with gr.Accordion("Advance", open=False):
                    with gr.Row():
                        width_slider  = gr.Slider(  label="Width",  value=512, minimum=256, maximum=1024, step=64 )
                        height_slider = gr.Slider(  label="Height", value=512, minimum=256, maximum=1024, step=64 )

                    with gr.Row():
                        seed_textbox = gr.Textbox( label="Seed",  value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e16)), inputs=[], outputs=[seed_textbox])

                generate_button = gr.Button( value="Generate", variant='primary' )

            with gr.Column():
                result_video = gr.Video( label="Generated Animation", interactive=False )
                json_config  = gr.Json( label="Config", value=None )

            inputs  = [base_model_dropdown, prompt_textbox, negative_prompt_textbox, width_slider, height_slider, video_length, seed_textbox]
            
            outputs = [result_video, json_config]
            
            generate_button.click( fn=controller.animate, inputs=inputs, outputs=outputs )
                
        gr.Examples( fn=controller.animate, examples=examples, inputs=inputs, outputs=outputs, cache_examples=True )
        
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(max_size=20)
    demo.launch()