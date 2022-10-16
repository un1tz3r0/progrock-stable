import argparse
import glob
import os
import random
import subprocess
import sys
import time
#import accelerate
from contextlib import contextmanager, nullcontext
from itertools import islice
from types import SimpleNamespace

import json5 as json
import k_diffusion as K
import k_diffusion.external
import numpy as np
import PIL
import torch
from einops import rearrange, repeat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageStat
from PIL.PngImagePlugin import PngInfo
from pytorch_lightning import seed_everything
from torch import autocast, nn
from tqdm import tqdm, trange

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging
    logging.set_verbosity_error()
except:
    pass

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def get_resampling_mode():
    try:
        from PIL import Image, __version__
        major_ver = int(__version__.split('.')[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def convert_img(pil_img):
    image = pil_img.convert("RGB")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_mask(path):
    image = Image.open(path).convert("GRAY")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def thats_numberwang(dir, wildcard):
    # get the highest numbered file in the out directory, and add 1. So simple.
    import re
    files = os.listdir(dir)
    filenums = []
    filenum = 0
    for file in files:
        if wildcard in file:
            m = re.match(".*?([0-9][0-9][0-9]*)", str(file))
            #start = file.rfind('-')
            #end = file.rfind('.')
            try:
                #filenum = file[start + 1:end]
                filenum = m[1]
                filenum = int(filenum)
            except:
                print(f'Improperly named file "{file}" in output directory')
                print(f'Tried to turn "{filenum}" into numberwang, but "{filenum}" is not numberwang!')
                print(f'Please make sure output filenames use the name-1234.png format')
                quit()
            filenums.append(filenum)
    if not filenums:
        numberwang = 0
    else:
        numberwang = max(filenums) + 1
    return numberwang

def slerp(device, t, v0:torch.Tensor, v1:torch.Tensor, DOT_THRESHOLD=0.9995):
    """Summary:
        Spherical/linear interpolation between two vectors.
    
    Args:
        device (torch.device): device to use
        t (float): interpolation parameter, in range [0, 1]
        v0 (torch.Tensor): vector at t=0
        v1 (torch.Tensor): vector at t=1
        DOT_THRESHOLD (float): threshold for dot product of v0 and v1. If the dot product
            is greater than this value, then the vectors are too close and we
            just linearly interpolate between them.
            
    Returns:
        torch.Tensor: interpolated vector
    """
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2).to(device)

    return v2

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def do_run(device, model, opt):
    #print(f'Starting render!')
    seed_everything(opt.seed)

    #os.makedirs(opt.outdir, exist_ok=True)
    #outpath = opt.outdir

    batch_size = 1

    # prompt = opt.prompt
    data = [batch_size * [opt.prompt]]
    # data = opt.prompt

    grid_count = 0 #thats_numberwang(outpath, opt.batch_name)

    sampler = DDIMSampler(model, device)

    if opt.init_image is not None:
        if isinstance(opt.init_image, str):
            assert os.path.isfile(opt.init_image)
            init_image = load_img(opt.init_image).to(device).half()
        else:
            init_image = convert_img(opt.init_image).to(device).half() # potentially needs to not be .half on mps and cpu modes
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        # strength is like skip_steps I think
        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        #print(f"target t_enc is {t_enc} steps")
    else:
        model_wrap = K.external.CompVisDenoiser(model)
        sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
        init_image = None

    start_code = None
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    # apple silicon support
    if device.type == 'mps':
        precision_scope = nullcontext

    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])

                        # process the prompt for randomizers and dynamic values
                        newprompts = []
                        for prompt in prompts:
                            prompt = randomize_prompt(prompt)
                            prompt = dynamic_value(prompt)
                            newprompts.append(prompt)
                        prompts = newprompts

                        print(f'\nPrompt for this image:\n   {prompts}\n')

                        # save a settings file for this image
                        if opt.save_settings:
                            save_settings(opt, prompts[0], grid_count)

                        c = model.get_learned_conditioning(prompts)

                        if init_image is None:
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            if opt.method != "ddim":
                                sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                                if isinstance(opt.seed, int):
                                    torch.manual_seed(opt.seed)
                                    # use regular starting noise
                                    x = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                                elif isinstance(opt.seed, list):
                                    # opt.seed is a list of (seed, weight) tuples. accumulate by progressive spherical interpolation
                                    x = None
                                    tw = 0 # sum([w for s, w in opt.seed])
                                    for s, w in opt.seed:
                                        torch.manual_seed(s)
                                        tw = tw + w
                                        sx = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                                        if x != None:
                                            x = slerp(device, max(0.0, min(1.0, w/tw)), x, sx)
                                        else:
                                            x = sx
                                else:
                                    raise TypeError("opt.seed must be an int or list of (seed: int, weight: float) tuples")
                                '''else:
                                    # add a little extra random noise to get varying output with same seed
                                    torch.manual_seed(opt.seed)
                                    base_x = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                                    torch.manual_seed(opt.variance_seed + n)
                                    target_x = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                                    x = slerp(device, max(0.0, min(1.0, opt.variance)), base_x, target_x)'''
                            
                                model_wrap_cfg = CFGDenoiser(model_wrap)
                                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                                if opt.method == "k_euler":
                                    samples_ddim = K.sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args)
                                elif opt.method == "k_euler_ancestral":
                                    samples_ddim = K.sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args)
                                elif opt.method == "k_heun":
                                    samples_ddim = K.sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args=extra_args)
                                elif opt.method == "k_dpm_2":
                                    samples_ddim = K.sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args)
                                elif opt.method == "k_dpm_2_ancestral":
                                    samples_ddim = K.sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args)
                                else: # k_lms
                                    samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args)
                            else:
                                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                conditioning=c,
                                                                batch_size=batch_size,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=opt.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=opt.ddim_eta,
                                                                x_T=start_code)

                        else:
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,)


                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        metadata = PngInfo()
                        if opt.hide_metadata == False:
                            metadata.add_text("prompt", '\x00'.join(prompts))
                            metadata.add_text("seed", str(opt.seed))
                            if opt.variance > 0.0:
                            	metadata.add_text("variance", str(opt.variance))
                            	metadata.add_text("variance_seed", str(opt.variance_seed + n))
                            metadata.add_text("steps", str(opt.ddim_steps))
                            metadata.add_text("scale", str(opt.scale))
                            metadata.add_text("ETA", str(opt.ddim_eta))
                            metadata.add_text("method", str(opt.method))

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            #output_filename = os.path.join(outpath, f'{opt.batch_name}{opt.device_id}-{grid_count:04}{opt.filetype}')
                            output_image = Image.fromarray(x_sample.astype(np.uint8))
                            yield output_image, metadata
                            #output_image.save(output_filename, pnginfo=metadata, quality = opt.quality)
                            #output_image.close()
                            #print(f'\nOutput saved as "{output_filename}"')
                            #output_jsonname = os.path.join(outpath, f'{opt.batch_name}{opt.device_id}-{grid_count:04}.json')
                            #with open(output_jsonname, "wt") as jsonfh:
                            #    json.dump({'opt': opt.__dict__, 'prompt': prompts}, jsonfh)
                            #print(f'Settings saved as "{output_jsonname}"\n')
                            grid_count += 1

                toc = time.time()
    #return output_filename

#functions for GO BIG
def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
    return(im)

# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices: # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source

def grid_coords(target, original, overlap):
    #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    #target should be the size for the gobig result, original is the size of each chunk being rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x,y)) #center chunk
    uy = y #up
    uy_list = []
    dy = y #down
    dy_list = []
    lx = x #left
    lx_list = []
    rx = x #right
    rx_list = []
    while uy > 0: #center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y: #center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
    render_edgex = last_coordx + original_x # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)

# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size): 
    width, height = og_size # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap)
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x+width, y+height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, new_size

def parse_args():
    my_parser = argparse.ArgumentParser(
        prog='ProgRock-Stable',
        description='Generate images from text prompts, based on Stable Diffusion.',
    )
    my_parser.add_argument(
        '-s',
        '--settings',
        action='append',
        required=False,
        default=['settings.json'],
        help='A settings JSON file to use, best to put in quotes. Multiples are allowed and layered in order.'
    )
    my_parser.add_argument(
        '-o',
        '--output',
        action='store',
        required=False,
        help='What output directory to use within images_out'
    )
    my_parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        action='store',
        required=False,
        help='Override the prompt'
    )
    my_parser.add_argument(
        '-c',
        '--cpu',
        type=int,
        nargs='?',
        action='store',
        required=False,
        default=False,
        const=0,
        help='Force use of CPU instead of GPU, and how many threads to run'
    )
    my_parser.add_argument(
        '-n',
        '--n_batches',
        type=int,
        action='store',
        required=False,
        help='How many batches of images to generate, each batch will be --n_iter images with the same randomized prompt and seed, but the --variance-seed will be different among them.'
    )
    my_parser.add_argument(
        '-i',
        '--n_iter',
        type=int,
        action='store',
        required=False,
        help='How many images to generate within a batch, varying the --variance-seed and mixing with the --seed noise proportional '
    )
    my_parser.add_argument(
        '--steps',
        type=int,
        action='store',
        required=False,
        help='Specify the number of steps to diffuse over'
    )
    my_parser.add_argument(
        '--scale',
        action='store',
        type=float,
	required=False,
        help='Specify cfg_scale or prompt-strictness parameter'
    )
    my_parser.add_argument(
        '--init-strength',
        action='store',
        type=float,
	required=False,
        help='Specify the strength of the init image, from 0.0 (noise) to 1.0 (unchanged)'
    )
    my_parser.add_argument(
        '--init-image',
        action='store',
        type=str,
	required=False,
        help='Specify an image to blend with the initial noise'
    )
    my_parser.add_argument(
        '--seed',
        type=int,
        action='store',
        required=False,
        help='Specify the numeric seed to be used to generate the initial noise for each batch'
    )
    my_parser.add_argument(
        '--frozen-seed',
        action='store_true',
        required=False,
        help="Do not increment seed value for each image of batch. Variance seed will still be incremented. Useful for getting variations of a particular seed and prompt using --frozen-seed."
    )
    my_parser.add_argument(
        '--variance-seed',
        type=int,
        action='store',
        required=False,
        default=None,
        help='Specify the numeric seed to be used for init noise variance.'
    )
    my_parser.add_argument(
        '--variance',
        type=float,
        action='store',
        required=False,
        default=0.1,
        help='Specify blending between the --seed noise and the --variance-seed noise.'
    )
    my_parser.add_argument(
        '--width', '-W',
        type=int,
        action='store',
        required=False,
        #default=512,
        help='Specify the output width'
    )
    my_parser.add_argument(
        '--height', '-H',
        type=int,
        action='store',
        required=False,
        #default=512,
        help='Specify the output height'
    )
    my_parser.add_argument(
        '-f',
        '--from_file',
        action='store',
        required=False,
        help='A text file with prompts (one per line)'
    )
    my_parser.add_argument(
        '--gobig',
        action='store_true',
        required=False,
        help='After generation, the image is split into sections and re-rendered, to double the size.'
    )
    my_parser.add_argument(
        '--gobig_init',
        action='store',
        required=False,
        help='An image to use to kick off GO BIG mode, skipping the initial render.'
    )
    my_parser.add_argument(
        '--gobig_scale',
        action='store',
        type=int,
        default = 2,
        required=False,
        help='What scale to multiply your original image by. 2 is a good value. 3 is insane. Anything more and I wish you luck.'
    )
    my_parser.add_argument(
        '--device',
        action='store',
        default = "cuda:0",
        required=False,
        help='The device to use for pytorch.'
    )
    my_parser.add_argument(
        '--interactive',
        action='store_true',
        required=False,
        help='Advanced option for bots and such. Wait for a job file, render it, then wait some more.'
    )

    return my_parser.parse_args()

# Simple check to see if a key is present in the settings file
def is_json_key_present(json, key, subkey="none"):
    try:
        if subkey != "none":
            buf = json[key][subkey]
        else:
            buf = json[key]
    except KeyError:
        return False
    if type(buf) == type(None):
        return False
    return True

import pathlib


def findfile(relpath, searchdirs=[".", str(pathlib.Path(__file__).parent)]):
    for d in searchdirs:
        f = pathlib.Path(d) / relpath
        if f.exists():
            return str(f)
    raise FileNotFoundError(f"Cannot find relative path {relpath} under any of: {searchdirs}")

# pick a random item from the cooresponding text file
def randomizer(category):
    random.seed()
    randomizers = []
    with open(findfile(f'settings/{category}.txt'), encoding="utf-8") as f:
        for line in f:
            randomizers.append(line.strip())
    random_item = random.choice(randomizers)
    return(random_item)

# replace anything surrounded by underscores with a random entry from the matching text file
def randomize_prompt(prompt):
    while "_" in prompt:
        start = prompt.index('_')
        end = prompt.index('_', start+1)
        swap = prompt[(start + 1):end]
        swapped = randomizer(swap)
        prompt = prompt.replace(f'_{swap}_', swapped, 1)
    return prompt

# Dynamic value - takes ready-made possible options within a string and returns the string with an option randomly selected
# Format is "I will return <Value1|Value2|Value3> in this string"
# Which would come back as "I will return Value2 in this string" (for example)
# Optionally if a value of ^^# is first, it means to return that many dynamic values,
# so <^^2|Value1|Value2|Value3> in the above example would become:
# "I will return Value3 Value2 in this string"
# note: for now assumes a string for return. TODO return a desired type
def dynamic_value(incoming):
    if type(incoming) == str:  # we only need to do something if it's a string...
        if incoming == "auto" or incoming == "random":
            return incoming
        elif "<" in incoming:   # ...and if < is in the string...
            text = incoming
            while "<" in text:
                start = text.find('<')
                end = text.find('>')
                swap = text[(start + 1):end]
                value = ""
                count = 1
                values = swap.split('|')
                if "^^" in values[0]:
                    count = values[0]
                    values.pop(0)
                    count = int(count[2:])
                random.shuffle(values)
                for i in range(count):
                    value = value + values[i] + " "
                value = value[:-1]  # remove final space
                text = text.replace(f'<{swap}>', value, 1)
            return text
        else:
            return incoming
    else:
        return incoming

class Settings:
    prompt = "A druid in his shop, selling potions and trinkets, fantasy painting by raphael lacoste and craig mullins"
    batch_name = "default"
    out_path = "./out"
    n_batches = 1
    steps = 50
    eta = 0.0
    n_iter = 1
    width = 512
    height = 512
    scale = 5.0
    dyn = None
    from_file = None
    seed = "random"
    variance = 0.0
    vary_seed = None
    frozen_seed = False
    init_image = None
    init_strength = 0.5
    gobig = False
    gobig_maximize = True
    gobig_overlap = 64
    gobig_realesrgan = False
    gobig_keep_slices = False
    cool_down = 0.0
    checkpoint = "./models/sd-v1-4.ckpt"
    inf_config = "./configs/stable-diffusion/v1-inference.yaml"
    use_jpg = False
    hide_metadata = False
    method = "k_lms"
    save_settings = False
    
    def clone(self):
        o = Settings()
        for k in self.__dict__:
            o.__dict__[k] = self.__dict__[k]
        return o
    
    def apply_settings_file(self, filename, settings_file):
        print(f'Applying settings file: {filename}')
        if is_json_key_present(settings_file, 'prompt'):
            self.prompt = (settings_file["prompt"])
        if is_json_key_present(settings_file, 'batch_name'):
            self.batch_name = (settings_file["batch_name"])
        if is_json_key_present(settings_file, 'out_path'):
            self.out_path = (settings_file["out_path"])
        if is_json_key_present(settings_file, 'n_batches'):
            self.n_batches = (settings_file["n_batches"])
        if is_json_key_present(settings_file, 'steps'):
            self.steps = (settings_file["steps"])
        if is_json_key_present(settings_file, 'eta'):
            self.eta = (settings_file["eta"])
        if is_json_key_present(settings_file, 'n_iter'):
            self.n_iter = (settings_file["n_iter"])
        if is_json_key_present(settings_file, 'width'):
            self.width = (settings_file["width"])
        if is_json_key_present(settings_file, 'height'):
            self.height = (settings_file["height"])
        if is_json_key_present(settings_file, 'scale'):
            self.scale = (settings_file["scale"])
        if is_json_key_present(settings_file, 'dyn'):
            self.dyn = (settings_file["dyn"])
        if is_json_key_present(settings_file, 'from_file'):
            self.from_file = (settings_file["from_file"])
        if is_json_key_present(settings_file, 'seed'):
            self.seed = (settings_file["seed"])
            if self.seed == "random":
                self.seed = random.randint(1, 10000000)
        if is_json_key_present(settings_file, 'variance'):
            self.variance = (settings_file["variance"])
        if is_json_key_present(settings_file, 'frozen_seed'):
            self.frozen_seed = (settings_file["frozen_seed"])
        if is_json_key_present(settings_file, 'init_strength'):
            self.init_strength = (settings_file["init_strength"])
        if is_json_key_present(settings_file, 'init_image'):
            self.init_image = (settings_file["init_image"])
        if is_json_key_present(settings_file, 'gobig'):
            self.gobig = (settings_file["gobig"])
        if is_json_key_present(settings_file, 'gobig_maximize'):
            self.gobig_maximize = (settings_file["gobig_maximize"])
        if is_json_key_present(settings_file, 'gobig_overlap'):
            self.gobig_overlap = (settings_file["gobig_overlap"])
        if is_json_key_present(settings_file, 'gobig_realesrgan'):
            self.gobig_realesrgan = (settings_file["gobig_realesrgan"])
        if is_json_key_present(settings_file, 'gobig_keep_slices'):
            self.gobig_keep_slices = (settings_file["gobig_keep_slices"])
        if is_json_key_present(settings_file, 'cool_down'):
            self.cool_down = (settings_file["cool_down"])
        if is_json_key_present(settings_file, 'checkpoint'):
            self.checkpoint = (settings_file["checkpoint"])
        if is_json_key_present(settings_file, 'inf_config'):
            self.inf_config = (settings_file["inf_config"])
        if is_json_key_present(settings_file, 'use_jpg'):
            self.use_jpg = (settings_file["use_jpg"])
        if is_json_key_present(settings_file, 'hide_metadata'):
            self.hide_metadata = (settings_file["hide_metadata"])
        if is_json_key_present(settings_file, 'method'):
            self.method = (settings_file["method"])
        if is_json_key_present(settings_file, 'save_settings'):
            self.save_settings = (settings_file["save_settings"])

def save_settings(options, prompt, filenum):
    setting_list = {
        'prompt' : prompt,
        'batch_name' : options.batch_name,
        'steps' : options.ddim_steps,
        'eta' : options.ddim_eta,
        'n_iter' : options.n_iter,
        'width' : options.W,
        'height' : options.H,
        'scale' : options.scale,
        'dyn' : options.dyn,
        'seed' : options.seed,
        'frozen_seed' : options.frozen_seed,
        'variance' : options.variance,
        'variance_seed' : options.variance_seed,
        'init_image' : options.init_image,
        'init_strength' : 1.0 - options.strength,
        'gobig' : options.gobig,
        'gobig_maximize' : options.gobig_maximize,
        'gobig_overlap' : options.gobig_overlap,
        'gobig_realesrgan' : options.gobig_realesrgan,
        'gobig_keep_slices' : options.gobig_keep_slices,
        'use_jpg' : "true" if options.filetype == ".jpg" else "false",
        'hide_metadata' : options.hide_metadata,
        'method' : options.method
    }
    with open(f"{options.outdir}/{options.batch_name}-{filenum:04}.json",  "w+", encoding="utf-8") as f:
        json.dump(setting_list, f, ensure_ascii=False, indent=4)

def esrgan_resize(input, id):
    input.save(f'_esrgan_orig{id}.png')
    input.close()
    try:
        subprocess.run(
            ['realesrgan-ncnn-vulkan', '-i', '_esrgan_orig.png', '-o', '_esrgan_.png'],
            stdout=subprocess.PIPE
        ).stdout.decode('utf-8')
        output = Image.open('_esrgan_.png').convert('RGBA')
        return output
    except Exception as e:
        print('ESRGAN resize failed. Make sure realesrgan-ncnn-vulkan is in your path (or in this directory)')
        print(e)
        quit()

def do_gobig(gobig_init, device, model, opt):
    overlap = opt.gobig_overlap
    outpath = opt.outdir
    # get our render size for each slice, and our target size
    input_image = Image.open(gobig_init).convert('RGBA')
    opt.W, opt.H = input_image.size
    target_W = opt.W * opt.gobig_scale
    target_H = opt.H * opt.gobig_scale
    if opt.gobig_realesrgan:
        input_image = esrgan_resize(input_image, opt.device_id)
    target_image = input_image.resize((target_W, target_H), get_resampling_mode())
    slices, new_canvas_size = grid_slice(target_image, overlap, (opt.W, opt.H))
    if opt.gobig_maximize == True:
        # increase our final image size to use up blank space
        target_image = input_image.resize(new_canvas_size, get_resampling_mode())
        slices, new_canvas_size = grid_slice(target_image, overlap, (opt.W, opt.H))
    input_image.close()
    # now we trigger a do_run for each slice
    betterslices = []
    slice_image = f'slice{opt.device_id}.png'
    opt.seed = opt.seed + 1
    for count, chunk_w_coords in enumerate(slices):
        chunk, coord_x, coord_y = chunk_w_coords
        chunk.save(slice_image)
        chunk.close()
        opt.init_image = slice_image
        opt.save_settings = False # we don't need to keep settings for each slice, just the main image.
        opt.n_iter = 1 # no point doing multiple iterations since only one will be used
        result = do_run(device, model, opt)
        resultslice = Image.open(result).convert('RGBA')
        betterslices.append((resultslice.copy(), coord_x, coord_y))
        resultslice.close()
        if opt.gobig_keep_slices == False:
            os.remove(result)
    # create an alpha channel for compositing the slices
    alpha = Image.new('L', (opt.W, opt.H), color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    a = 0
    i = 0
    a_overlap = overlap # int(overlap / 2) # we want the alpha gradient to be half the size of the overlap, otherwise we always see some of the original background underneath
    shape = ((opt.W, opt.H), (0,0))
    while i < overlap:
        alpha_gradient.rectangle(shape, fill = a)
        a += int(255 / a_overlap)
        a = 255 if a > 255 else a
        i += 1
        shape = ((opt.W - i, opt.H - i), (i,i))
    mask = Image.new('RGBA', (opt.W, opt.H), color=0)
    mask.putalpha(alpha)
    # now composite the slices together
    finished_slices = []
    for betterslice, x, y in betterslices:
        finished_slice = addalpha(betterslice, mask)
        finished_slices.append((finished_slice, x, y))
    final_output = grid_merge(target_image, finished_slices)
    # name the file in a way that hopefully doesn't break things
    print(f'result is {result}')
    result = result.replace('.png','')
    result_split = result.rsplit('-', 1)
    result_split[0] = result_split[0] + '_gobig-'
    result = result_split[0] + result_split[1]
    print(f'Gobig output saving as {result}{opt.filetype}')
    final_output.save(f'{result}{opt.filetype}', quality = opt.quality)
    final_output.close()

from contextlib import contextmanager


class nonedict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.__missing_values__ = {}
    
    def set_missing(self, key, value):
        self.__missing_values__[key] = value
    
    def __missing__(self, key):
        return None if key not in self.__missing_values__ else self.__missing_values__[key]

@contextmanager
def loadedmodel(*args, **kwargs):
    import pathlib

    #print('\nPROG ROCK STABLE')
    #print('----------------')
    # rolling a d20 to see if I should pester you about supporting PRD.
    # Apologies if this offends you. At least it's only on a critical miss, right?
    #d20 = random.randint(1, 20)
    #if d20 == 1:
    #    print('Please consider supporting my Patreon. Thanks! https://is.gd/rVX6IH')
    #else:
    #    print('')
    #kwargs = parse_args()
    kwargs = nonedict(kwargs)
    kwargs.set_missing('settings', [])
    kwargs.set_missing('device', 'cuda:0')
    
    # Load the JSON config files
    settings = Settings()
    for setting_arg in kwargs['settings']:
        setting_arg = str(pathlib.Path(__file__).parent / setting_arg)
        try:
            with open(setting_arg, 'r', encoding="utf-8") as json_file:
                print(f'Parsing {setting_arg}')
                settings_file = json.load(json_file)
                settings.apply_settings_file(setting_arg, settings_file)
        except Exception as e:
            print('Failed to open or parse ' + setting_arg + ' - Check formatting.')
            print(e)
            quit()

    def apply_settings_from_args(settings, kwargs):
        ''' for key, value in kwargs.items():
            if value is not None:
                settings.set(key, value)'''
        
        # override settings from files with anything coming in from the command line
        if 'prompt' in kwargs.keys():
            settings.prompt = kwargs['prompt']

        if 'output' in kwargs.keys():
            settings.batch_name = kwargs['output']

        if 'n_batches' in kwargs.keys():
            settings.n_batches = kwargs['n_batches']

        if 'n_iter' in kwargs.keys():
            settings.n_iter = kwargs['n_iter']

        if 'from_file' in kwargs.keys():
            settings.from_file = kwargs['from_file']

        if 'steps' in kwargs.keys():
            settings.steps = kwargs['steps']

        if 'scale' in kwargs.keys():
            settings.scale = kwargs['scale']

        if 'init_image' in kwargs.keys():
            settings.init_image = kwargs['init_image']

        if 'init_strength' in kwargs.keys():
            settings.init_strength = kwargs['init_strength']

        if 'seed' in kwargs.keys():
            settings.seed = kwargs['seed']

        if 'frozen_seed' in kwargs.keys():
            settings.frozen_seed = kwargs['frozen_seed']

        if 'variance' in kwargs.keys():
            settings.variance = kwargs['variance']

        if 'variance_seed' in kwargs.keys():
            settings.variance_seed = kwargs['variance_seed']

        if 'width' in kwargs.keys():
            settings.width = kwargs['width']

        if 'height' in kwargs.keys():
            settings.height = kwargs['height']

        return settings
    
    settings = apply_settings_from_args(settings, kwargs)
    
    outdir = (f'./out/{settings.batch_name}')
    filetype = ".jpg" if settings.use_jpg else ".png"
    quality = 97 if settings.use_jpg else 100

    valid_methods = ['k_lms', 'k_dpm_2_ancestral', 'k_dpm_2', 'k_heun', 'k_euler_ancestral', 'k_euler', 'ddim']
    if any(settings.method in s for s in valid_methods):
        print(f'Using {settings.method} sampling method.')
    else:
        print(f'Method {settings.method} is not available. The valid choices are:')
        print(valid_methods)
        print()
        print(f'Falling back k_lms')
        settings.method = 'k_lms'

    # setup the model
    ckpt = settings.checkpoint # "./models/sd-v1-3-full-ema.ckpt"
    inf_config = settings.inf_config # "./configs/stable-diffusion/v1-inference.yaml"
    print(f'Loading the model and checkpoint ({ckpt})...')
    config = OmegaConf.load(str(pathlib.Path(__file__).parent / f"{inf_config}"))
    model = load_model_from_config(config, str(pathlib.Path(__file__).parent / f"{ckpt}"), verbose=False)

    # setup the device
    device_id = "" # leave this blank unless it's a cuda device
    if torch.cuda.is_available() and "cuda" in kwargs['device']:
        device = torch.device(f"{kwargs['device']}")
        device_id = ("_" + kwargs['device'].rsplit(':',1)[1]) if "0" not in kwargs['device'] else ""
    elif ("mps" in kwargs['device']) or (torch.backends.mps.is_available()):
        device = torch.device("mps")
        settings.method = "ddim" # k_diffusion currently not working on anything other than cuda
    else:
        # fallback to CPU if we don't recognize the device name given
        device = torch.device("cpu")
        cores = os.cpu_count()
        torch.set_num_threads(cores)
        settings.method = "ddim" # k_diffusion currently not working on anything other than cuda

    print('Pytorch is using device:', device)

    if "cuda" in str(device):
        model.cuda()
    model.eval()

    # load the model to the device
    if "cuda" in str(device):
        model = model.half() # half-precision mode for gpus, saves vram, good good
    model = model.to(device)

    #self.device = device
    #self.settings = Settings()
    #self.model = model
    
    def run(**kwargs):
        settings = Settings()
        apply_settings_from_args(settings, kwargs)
        
        # pack up our settings into a simple namespace for the renderer
        opt = {
            "prompt" : settings.prompt,
            "batch_name" : settings.batch_name,
            "ddim_steps" : settings.steps,
            "ddim_eta" : settings.eta,
            "n_iter" : settings.n_iter,
            "W" : settings.width,
            "H" : settings.height,
            "C" : 4,
            "f" : 8,
            "scale" : settings.scale,
            "dyn" : settings.dyn,
            "seed" : settings.seed,
            "frozen_seed": settings.frozen_seed,
            "variance": settings.variance,
            "variance_seed": 0 if not 'variance_seed' in settings.__dict__.keys() else settings.variance_seed,
            "precision": "autocast",
            "init_image": settings.init_image,
            "strength": 1.0 - settings.init_strength,
            "gobig": False,
            "gobig_scale": 2.0,
            "gobig_maximize": settings.gobig_maximize,
            "gobig_overlap": settings.gobig_overlap,
            "gobig_realesrgan": settings.gobig_realesrgan,
            "gobig_keep_slices": settings.gobig_keep_slices,
            "config": config,
            "filetype": ".png",
            "hide_metadata": settings.hide_metadata,
            "quality": 100,
            "device_id": device_id,
            "method": settings.method,
            "save_settings": settings.save_settings,
        }
        opt = SimpleNamespace(**opt)
        return do_run(device, model, opt)

    yield run

def main():
    import pathlib

    #print('\nPROG ROCK STABLE')
    #print('----------------')
    # rolling a d20 to see if I should pester you about supporting PRD.
    # Apologies if this offends you. At least it's only on a critical miss, right?
    #d20 = random.randint(1, 20)
    #if d20 == 1:
    #    print('Please consider supporting my Patreon. Thanks! https://is.gd/rVX6IH')
    #else:
    #    print('')

    cl_args = parse_args()

    # Load the JSON config files
    settings = Settings()
    for setting_arg in kwargs['settings']:
        setting_arg = str(pathlib.Path(__file__).parent / setting_arg)
        try:
            with open(setting_arg, 'r', encoding="utf-8") as json_file:
                print(f'Parsing {setting_arg}')
                settings_file = json.load(json_file)
                settings.apply_settings_file(setting_arg, settings_file)
        except Exception as e:
            print('Failed to open or parse ' + setting_arg + ' - Check formatting.')
            print(e)
            quit()

    kwargs = cl_args.__dict__

    # override settings from files with anything coming in from the command line
    if kwargs['prompt']:
        settings.prompt = kwargs['prompt']

    if kwargs['output']:
        settings.batch_name = kwargs['output']

    if kwargs['n_batches']:
        settings.n_batches = kwargs['n_batches']

    if kwargs['n_iter']:
        settings.n_iter = kwargs['n_iter']

    if kwargs['from_file']:
        settings.from_file = kwargs['from_file']

    if kwargs['steps']:
        settings.steps = kwargs['steps']

    if kwargs['scale']:
        settings.scale = kwargs['scale']

    if kwargs['init_image']:
        settings.init_image = kwargs['init_image']

    if kwargs['init_strength']:
        settings.init_strength = kwargs['init_strength']

    if kwargs['seed']:
        settings.seed = kwargs['seed']

    if kwargs['frozen_seed']:
        settings.frozen_seed = kwargs['frozen_seed']

    if kwargs['variance']:
        settings.variance = kwargs['variance']

    if kwargs['variance_seed']:
        settings.variance_seed = kwargs['variance_seed']

    if kwargs['width']:
        settings.width = kwargs['width']

    if kwargs['height']:
        settings.height = kwargs['height']

    outdir = (f'./out/{settings.batch_name}')
    filetype = ".jpg" if settings.use_jpg else ".png"
    quality = 97 if settings.use_jpg else 100

    valid_methods = ['k_lms', 'k_dpm_2_ancestral', 'k_dpm_2', 'k_heun', 'k_euler_ancestral', 'k_euler', 'ddim']
    if any(settings.method in s for s in valid_methods):
        print(f'Using {settings.method} sampling method.')
    else:
        print(f'Method {settings.method} is not available. The valid choices are:')
        print(valid_methods)
        print()
        print(f'Falling back k_lms')
        settings.method = 'k_lms'

    # setup the model
    ckpt = settings.checkpoint # "./models/sd-v1-3-full-ema.ckpt"
    inf_config = settings.inf_config # "./configs/stable-diffusion/v1-inference.yaml"
    print(f'Loading the model and checkpoint ({ckpt})...')
    config = OmegaConf.load(str(pathlib.Path(__file__).parent / f"{inf_config}"))
    model = load_model_from_config(config, str(pathlib.Path(__file__).parent / f"{ckpt}"), verbose=False)

    # setup the device
    device_id = "" # leave this blank unless it's a cuda device
    if torch.cuda.is_available() and "cuda" in kwargs['device']:
        device = torch.device(f"{kwargs['device']}")
        device_id = ("_" + kwargs['device'].rsplit(':',1)[1]) if "0" not in kwargs['device'] else ""
    elif ("mps" in kwargs['device']) or (torch.backends.mps.is_available()):
        device = torch.device("mps")
        settings.method = "ddim" # k_diffusion currently not working on anything other than cuda
    else:
        # fallback to CPU if we don't recognize the device name given
        device = torch.device("cpu")
        cores = os.cpu_count()
        torch.set_num_threads(cores)
        settings.method = "ddim" # k_diffusion currently not working on anything other than cuda

    print('Pytorch is using device:', device)

    if "cuda" in str(device):
        model.cuda()
    model.eval()

    # load the model to the device
    if "cuda" in str(device):
        model = model.half() # half-precision mode for gpus, saves vram, good good
    model = model.to(device)

    there_is_work_to_do = True
    while there_is_work_to_do:
        if kwargs['interactive']:
            # Interactive mode waits for a job json, runs it, then goes back to waiting
            job_json = ("job_" + kwargs['device'] + ".json").replace(":","_")
            print(f'\nInteractive Mode On! Waiting for {job_json}')
            job_ready = False
            while job_ready == False:
                if os.path.exists(job_json):
                    print(f'Job file found! Processing.')
                    try:
                        with open(job_json, 'r', encoding="utf-8") as json_file:
                            settings_file = json.load(json_file)
                            settings.apply_settings_file(job_json, settings_file)
                            prompts = []
                            prompts.append(settings.prompt)
                        job_ready = True
                    except Exception as e:
                        print('Failed to open or parse ' + job_json + ' - Check formatting.')
                        print(e)
                        os.remove(job_json)
                else:
                    time.sleep(0.5)

        #outdir = (f'{settings.out_path}/{settings.batch_name}')
        outdir = os.path.join(settings.out_path, settings.batch_name)
        print(f'Saving output to {outdir}')
        filetype = ".jpg" if settings.use_jpg == True else ".png"
        quality = 97 if settings.use_jpg else 100

        prompts = []
        if settings.from_file is not None:
            with open(settings.from_file, "r", encoding="utf-8") as f:
                prompts = f.read().splitlines()
        else:
            prompts.append(settings.prompt)


        for p in range(len(prompts)):
            for i in range(settings.n_batches):
                # pack up our settings into a simple namespace for the renderer
                opt = {
                    "prompt" : prompts[p],
                    "batch_name" : settings.batch_name,
                    "outdir" : outdir,
                    "ddim_steps" : settings.steps,
                    "ddim_eta" : settings.eta,
                    "n_iter" : settings.n_iter,
                    "W" : settings.width,
                    "H" : settings.height,
                    "C" : 4,
                    "f" : 8,
                    "scale" : settings.scale,
                    "dyn" : settings.dyn,
                    "seed" : settings.seed + (0 if 'variance_seed' in settings.__dict__.keys() else i),
                    "frozen_seed": settings.frozen_seed,
                    "variance": settings.variance,
                    "variance_seed": (settings.seed + i + 1) if not 'variance_seed' in settings.__dict__.keys() else (settings.variance_seed + i),
                    "precision": "autocast",
                    "init_image": settings.init_image,
                    "strength": 1.0 - settings.init_strength,
                    "gobig": kwargs['gobig'],
                    "gobig_scale": kwargs['gobig_scale'],
                    "gobig_maximize": settings.gobig_maximize,
                    "gobig_overlap": settings.gobig_overlap,
                    "gobig_realesrgan": settings.gobig_realesrgan,
                    "gobig_keep_slices": settings.gobig_keep_slices,
                    "config": config,
                    "filetype": filetype,
                    "hide_metadata": settings.hide_metadata,
                    "quality": quality,
                    "device_id": device_id,
                    "method": settings.method,
                    "save_settings": settings.save_settings,
                }
                opt = SimpleNamespace(**opt)

                # render the image(s)!
                if kwargs['gobig_init'] == None:
                    # either just a regular render, or a regular render that will next go_big
                    gobig_init = do_run(device, model, opt)
                else:
                    gobig_init = kwargs['gobig_init']
                if kwargs['gobig'] or settings.gobig:
                    do_gobig(gobig_init, device, model, opt)
                if settings.cool_down > 0 and ((i < (settings.n_batches - 1)) or p < (len(prompts) - 1)):
                    print(f'Pausing {settings.cool_down} seconds to give your poor GPU a rest...')
                    time.sleep(settings.cool_down)
            if not 'variance_seed' in settings.__dict__.keys():
                settings.seed = settings.seed + 1
        if kwargs['interactive'] == False:
            #only doing one render, so we stop after this
            there_is_work_to_do = False
        else:
            print('\nJob finished! And so we wait...\n')
            os.remove(job_json)

if __name__ == "__main__":
    main()
