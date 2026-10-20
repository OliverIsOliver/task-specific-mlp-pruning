# Task-specific MLP pruning on Gemma 2 2B

This project’s goal is to find how many MLP neurons can be removed for a task while keeping its accuracy. The PyTorch pipeline ranks neurons by activation × downward sensitivity on a training set, then learns which combination of neurons to keep across all 26 MLP layers for a chosen percentage of neurons (e.g., 70%, 80%, or 90%). It removes the rest and checks performance on a separate testing set.

[See in-depth results](https://oliveramaher.github.io/task-specific-mlp-pruning/?v=d9ec1b1)

# Process

### 1. Measure how much each neuron affects training loss

For each training example, give Gemma the prompt and its correct answer. Compute the average loss on the answer tokens and calculate how sensitive that loss is to each MLP neuron's activation. A **neuron’s activation** tells us its output strength. Its **gradient** tells us how sensitive the loss is to that activation. Then calculate the importance score for each neuron using its activations and loss gradients:

$$
F_j = \frac{1}{2M}\sum_{i=1}^{M}\left(\sum_t a_{itj}\frac{\partial L_i}{\partial a_{itj}}\right)^2
$$

Here, $a_{itj}$ is neuron $j$’s activation at token $t$, $L_i$ is the training answer loss, and $M=50$ for 50 training examples. This is a **diagonal empirical Fisher approximation** which estimates the loss sensitivity rather than counts how often a neuron fires. A low score makes a neuron more likely to remove, but it does not prove that the neuron is unused. The next stage learns which combination of neurons to keep.

Code: [`src/fisher.py`](src/fisher.py).

### 2. Create the starting mask

Use the scores across all 26 layers and select the lowest-scoring channels to remove. A 20% cut removes **47,923 of 239,616 MLP channels**. Individual layers lose different amounts.

Fisher estimates are used to rank initally. Then each neuron recieved a gate, using **1 to keep a neuron and 0 to remove it**.

### 3. Learn a better selection at the same budget

For a 20% cut, the optimizer can change **which** 47,923 neurons are removed, but it must always remove 47,923. It learns which to select from the training answers.

Give every neuron a trainable gate score, initialized by the Fisher-ranking order. On each pass, keep the highest-scoring channels and mask the rest. After backpropagation, update the scores and create a new mask, but **original model weights are always frozen**.

Normally, removing a neuron means the model cannot learning whether it should be kept instead. A **straight-through estimator** calculates how useful each neuron would be, even if it is currently removed. That lets the optimizer restore initially removed neurons and remove other ones while keeping the total count the same.

Code: [`src/masks.py`](src/masks.py).

### 4. Slice the matrices, then test the smaller model

For each mask, use the original weights and physically cut the MLP matrices, removing the selected rows and columns. The remaining weights don't change, as well as attention, embeddings, and the output head.

Run the resulting smaller model on the same 100 held-out prompts per task and then grade the answers. Food orders are compared to food/quantity pairs regardless of order, appointment and policy outputs are compared to matching JSONs, and addition is compared to the exact number.

Code: [`src/model.py`](src/model.py), [`src/generation.py`](src/generation.py), and [`src/evaluation.py`](src/evaluation.py).

# Results

![Task accuracy by percentage of MLP neurons cut](assets/task-accuracy.png?v=27fc9073)

The chart shows accuracy on testing sets for each task, comparing the original model with pruned models that remove 10%, 20%, or 30% of MLP channels. Food-order extraction and return-policy decisions score highest at a 30% cut, appointment scheduling stays nearly flat, and integer addition is highest at a 10% cut. These results are for these four specific test sets and does not show that pruning improves general performance.

# Run

Use Python 3.13 to exactly match the results. Accept the [Gemma model license](https://huggingface.co/google/gemma-2-2b) and sign in to Hugging Face before downloading weights.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

See the saved masks and all responses without loading the model:

```bash
python verify.py
python -m unittest discover -s tests
```

Test the saved food-order masks, or recalculate Fisher scores and create new masks from the 50 training examples:

```bash
python run.py --tasks food --cuts 0 10 20 30
python run.py --tasks food --cuts 10 20 30 --fit --output runs/refit
```

The other task names are `appointments`, `returns`, and `addition`. Excluding `--tasks` runs all of them. Files will go to `runs/`, and the masks and results stay inside of `masks/` and `results/`.

# Results website

The site displays the saved masks and responses; browsing it requires no model or Python server. Use Node.js 22.12 or newer:

```bash
cd visualization
npm ci
npm run dev
```

`npm run build` produces the static site in `visualization/dist/`. To regenerate the site's cached data from this repository's datasets, masks, and results, run `python visualization/export.py` from the repository root with NumPy installed.

For GitHub Pages, select **Settings → Pages → Source → GitHub Actions**. The included [workflow](.github/workflows/pages.yml) builds and publishes the site on pushes to `main`, or when run manually. The site will be available at `https://oliveramaher.github.io/task-specific-mlp-pruning/` after deployment.
