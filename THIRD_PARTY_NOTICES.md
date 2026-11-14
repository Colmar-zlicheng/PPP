# Third-party notices

The PPP code is released under the Apache License 2.0 ([LICENSE](LICENSE)). Some files are adapted
from the third-party projects listed below and remain subject to their original licenses. Each
adapted file starts with a short header naming its source.

## Code adapted in this repository

### LeWorldModel (MIT License)

Source: https://github.com/lucas-maes/le-wm

Files: `lib/models/lewm/jepa.py`, `lib/models/lewm/module.py`

```
MIT License

Copyright (c) 2026 Lucas Maes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Seer (Apache License 2.0)

Source: https://github.com/InternRobotics/Seer

Files: `lib/models/seer/seer_net.py`, `lib/models/seer/perceiver.py`, `lib/models/seer/gpt2.py`,
`lib/models/seer/mae_encoder.py`

These files port Seer's network to this codebase and were modified, for example by removing the
`einops_exts` dependency and adding the options used by Seer-PPP. Parameter names are kept so that
Seer state dicts load unchanged. The Apache License 2.0 text is in [LICENSE](LICENSE).

- `lib/models/seer/gpt2.py` follows Seer's GPT-2 module, which is adapted from the GPT-2
  implementation of Hugging Face Transformers (https://github.com/huggingface/transformers, Apache
  License 2.0).
- `lib/models/seer/mae_encoder.py` follows Seer's MAE module, which implements the encoder of MAE
  (https://github.com/facebookresearch/mae, CC BY-NC 4.0 License).

## External dependencies and assets (not part of this repository)

- **LIBERO** (https://github.com/Lifelong-Robot-Learning/LIBERO, MIT License): simulator, benchmark
  and demonstrations, installed separately into `external/LIBERO`. The processed data on Hugging Face
  (`assets/latent_extract/libero/libero_full_consolidated.h5`) is derived from the LIBERO
  demonstrations.
- **MAE ViT-B/16 weights** (`checkpoints/vit_mae/mae_pretrain_vit_base.pth`, from
  https://github.com/facebookresearch/mae): CC BY-NC 4.0 License. The Seer and Seer-PPP models use
  them as their frozen image encoder.
- **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) and **all-MiniLM-L6-v2**
  (`sentence-transformers/all-MiniLM-L6-v2`): used to compute the instruction embeddings in
  `assets/` and in the consolidated data.
