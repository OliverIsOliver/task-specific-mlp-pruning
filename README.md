# Task-specific MLP pruning on Gemma 2 2B

This project’s goal is to find how many MLP neurons can be removed for a task while keeping its accuracy. The PyTorch pipeline ranks neurons by activation × downward sensitivity on a training set, then learns which combination of neurons to keep across all 26 MLP layers for a chosen percentage of neurons (e.g., 70%, 80%, or 90%). Lastly, it removes the rest of the neurons and checks performance on a separate testing set.

[See in-depth results](https://oliveramaher.github.io/task-specific-mlp-pruning)

# Process

### 1. Measure how much each neuron affects training loss

To begin, give Gemma the prompt and its correct answer for every training example. Then calculate the average loss of the answer tokens and how sensitive that loss is to each neuron's activation. The **neuron’s activation** tells us its output strength and its **gradient** tells us how sensitive the loss is to that activation, so using its activations and loss gradients, we can calculate the importance score for each neuron using this formula:

$$
F_j = \frac{1}{2M}\sum_{i=1}^{M}\left(\sum_t a_{itj}\frac{\partial L_i}{\partial a_{itj}}\right)^2
$$

$a_{itj}$ is neuron $j$’s activation at token $t$, $L_i$ is the training answer loss, and $M=50$ (for 50 training examples). The equation uses a **diagonal empirical Fisher approximation** to estimate the loss sensitivity rather than actually count how often the neuron fires. If a neuron recieves a low score, its more likely to be removed although we cannot with 100% certainty know that the neuron is unused.

Code: [`src/fisher.py`](src/fisher.py).

### 2. Create the starting mask

From the scores, we select the neurons across the 26 MLP layers to remove, and for reference, a 20% cut removes **47,923 of 239,616 MLP channels**, however different layers lose different amounts. Before neurons are quite removed from the matrices, each neuron recieves a gate, using **1 to keep a neuron and 0 to remove it**.

### 3. Learn a better selection at the same budget

For a 20% cut, the optimizer can change **which** 47,923 neurons are removed, but the amount removed must stay the same.

We give every neuron a trainable gate score that starts out based on the Fisher-ranking order. Then on each example, we keep the highest-scoring neurons and mute the rest by making their gate 0. After backpropagation, we update the scores and create a new mask, but **the original weights never change**.

Normally, removing a neuron makes it so the model cannot learning whether it should stay removed or be kept ,but a **straight-through estimator** calculates how useful each neuron would be, even if it is already removed. That lets the optimizer bring back removed neurons and remove other ones so the total count stays the same.

Code: [`src/masks.py`](src/masks.py).

### 4. Slice the matrices, then test the smaller model

With the final mask, use the original weights and physically split up the MLP matrices to  removing the selected rows and columns. Still, the weights don't change, as well as attention, embeddings, and the output head.

Finally, run the resulting shrunken model on the same 100 held-out prompts and then grade the answers. Food orders must match food/quantity pairs in any order, appointment and policy outputs must match the JSON answers, and addition must equal the exact number.

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

The site shows the saved masks and responses, so viewing it requires no model or Python server. Use Node.js 22.12 or newer:

```bash
cd visualization
npm ci
npm run dev
```

`npm run build` makes the static site in `visualization/dist/`. To rebuild the site's svaed data from the repo's datasets, masks, and results, run `python visualization/export.py` from the repo root with NumPy.