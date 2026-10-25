# Kaggle p100m English prose run

The script in `kaggle/sunnynet_p100.py` and its `kaggle/kernel-metadata.json` produced a 102,126,662-parameter SunnyNet checkpoint after 24,000 steps. Validation loss was **2.475938** (perplexity **11.8929**). This is a text-completion model, not an instruction-following chatbot; prose remains grammatically weak.

The checkpoint and its matching BPE tokenizer are hosted in [CodeDoes/SunnyNet-P100M-Kaggle](https://github.com/CodeDoes/SunnyNet-P100M-Kaggle). The 442,190,099-byte checkpoint is stored there with Git LFS (SHA-256 `cde00ce3309a21f6f5b5e9666727bb467e50d37d6e80c2bab3b5e2c977fdce38`). Clone that repository next to SunnyNet and run `git lfs pull` there if the checkpoint is only a pointer.

From the SunnyNet directory, after installing its Python dependencies with `python -m pip install -e .`, try:

```sh
python -m sunnynet.cli --ckpt ../SunnyNet-P100M-Kaggle/models/p100m_en.pt --tokenizer ../SunnyNet-P100M-Kaggle/models/p100m_en_tokenizer.json --prompt "Once upon a time, " --max-new 40 --temperature 0.7 --top-k 40 --seed 42 --threads 1
```

## Observed input and output

These continuations were generated from the checkpoint using the command above, changing only `--prompt`. The model outputs only the continuation; trailing spaces are trimmed below.

**Input:** `Once upon a time, `

```text
and was horning to the days in the
hand who had take the place of the
```

**Input:** `The morning sun was `

```text
a little to do of the scape of the
turb which clear the bark
```

**Input:** `It was the best of times, `

```text
and was scornce like the scape of the
visic of the place, of
```

The Kaggle script and metadata contain no API credentials. Running `kaggle kernels push -p kaggle` from this repository uploads the script and starts a new training run; it is not needed to test the downloaded checkpoint.
