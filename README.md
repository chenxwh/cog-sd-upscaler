# cog-sd-x2-latent-upscaler


Web Demoe and API: 
[![Replicate](https://replicate.com/cjwbw/sd-x2-latent-upscaler/badge)](https://replicate.com/cjwbw/sd-x2-latent-upscaler)

An implementation of [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler) in [Cog](https://github.com/replicate/cog), and pushing it to Replicate.


First, download the weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i image=@...

Or, push to a Replicate page:

    cog push r8.im/...
