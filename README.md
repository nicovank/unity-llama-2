# Requesting and setting up a node

Seems like the largest models pretty much require an entire
[GIGABYTE G262](https://docs.unity.rc.umass.edu/technical/nodelist.html) node.

```bash
srun --pty -c 64 --mem 500G -C a100-80g -p gpu -G 4 -t 1-00:00:00 --mail-type=BEGIN zsh
```

## Modules

```bash
module load miniconda/22.11.1-1
module load cuda/11.8.0
```

## Conda

```bash
conda activate llama # If the environment already exists.

conda create --name llama python=3.11
conda activate llama
pip install -r requirements.txt
# TODO: transformers.
```

# Getting the weights

> **Note**
> You should be able to just use the weights in my directory and skip this section.

## Download

You should request download links [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
./download.sh # Requires download link.
```

> **Note**
> For CodeLlama, the weights seem to be open access on HuggingFace.

```bash
git clone https://github.com/facebookresearch/codellama.git
cd codellama
./download.sh # Requires download link.
```

## Converting weights to HuggingFace

```bash
# This example is for the 70B chat model.
ln -s `pwd`/tokenizer.model llama-2-70b-chat/tokenizer.model
python3 /work/pi_emeryb_umass_edu/nvankempen/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir llama-2-70b-chat/ --model_size 70B --output_dir llama-2-70b-chat-hf
```

# Example

## Llama 2

```python
import torch
import transformers

tokenizer = transformers.LlamaTokenizer.from_pretrained("./llama-2-70b-chat-hf")
streamer = transformers.TextStreamer(tokenizer)
model = transformers.LlamaForCausalLM.from_pretrained("./llama-2-70b-chat-hf", device_map="auto")

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    device_map="auto",
)

with open("/home/nvankempen_umass_edu/cwhy-short", "r") as f:
    prompt = f.read()

prompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>w

{prompt} [/INST]
"""

inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    max_length=4096,
    streamer=streamer
)
```

## Code Llama

```python
import torch
import transformers

model_name = "codellama/CodeLlama-34b-Instruct-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
streamer = transformers.TextStreamer(tokenizer)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    device_map="auto",
)

with open("/home/nvankempen_umass_edu/cwhy-short", "r") as f:
    prompt = f.read()

prompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>w

{prompt} [/INST]
"""

inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    max_length=16384,
    streamer=streamer
)
```
