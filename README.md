# Retina Normalization

An experimental benchmark for exploring bio- and neuroinspired algorithmics in
convolutional neural networks.

The project asks a practical question: how do retina-inspired input mechanisms
change the accuracy, robustness, efficiency, and training behavior of different
CNN architectures? It provides a small, reproducible setup for comparing a
baseline CNN with models augmented by biologically motivated preprocessing and
normalization ideas.

> **Status:** Early research prototype. APIs, experiment design, and reported
> results are expected to change.

## Research Direction

The experiments are intended to make architectural comparisons measurable and
repeatable. Current building blocks include:

- **CNN model comparisons** using torchvision architectures and architecture
	families such as ResNet, MobileNet, ConvNeXt, EfficientNet, and RegNet.
- **Retina-inspired preprocessing** as a front-end for models that receive
	visual input.
- **Controlled input corruptions** including Gaussian noise, blur, brightness
	shifts, and contrast shifts for robustness measurements.
- **Efficiency measurements** such as parameter counts and model compute
	estimates, alongside training and evaluation metrics.
- **Timestamped reports and plots** saved under `reports/` for later analysis,
  with a dedicated visual-output folder for each model.

The current default benchmark uses CIFAR-10, a capped sample of the training
and test splits, ImageNet-pretrained model weights, and a newly adapted
10-class classifier head. These defaults are deliberately lightweight so that
experiments can be iterated on a local machine before scaling up.

## Project Layout

| Directory | Purpose |
| --- | --- |
| `dataset_functions/` | Dataset readers and data loader construction |
| `experiment_setup/` | Experiment orchestration and input corruptions |
| `measurement_functions/` | Compute and efficiency measurements |
| `model_functions/` | Model catalogs, loading, adaptation, and training |
| `retina_mechanisms/` | Retina-inspired preprocessing components |
| `utility_functions/` | Argument parsing and visualization |
| `reports/` | Generated experiment reports and plots |

## Setup

Requirements:

- Python 3.12
- A working PyTorch installation for your hardware
- Optional CUDA support for GPU experiments

### Windows PowerShell

```powershell
. .\setup_venv.ps1
```

### WSL or Linux

```bash
source ./setup_venv.sh
```

The setup scripts create and activate the project virtual environment and
install the dependencies declared in `pyproject.toml`.

## Run A Benchmark

Run the default experiment:

```bash
python main.py
```

Select individual models by repeating `--model`:

```bash
python main.py --model resnet18 --model mobilenet_v3_small
```

Select a cataloged architecture family with `--family`:

```bash
python main.py --family resnet
python main.py --family mobilenetv3 --family efficientnet
```

When no model is specified, the benchmark uses `mobilenet_v3_small` as its
baseline. New results are written to a timestamped directory such as
`reports/experiment_MMDD_HHMM/`. Each model's metrics and confusion-matrix
visuals are stored in a subdirectory named after that model.

Confusion-matrix entries include the matrix itself, row totals, column totals,
the grand total, and dataset label names when the dataset exposes them.

## Add A Custom Dataset

Custom datasets use the `CustomDataset` blueprint in
`dataset_functions/datasets.py`. To add one:

1. Create a subclass of `CustomDataset`.
2. Implement `__len__()` to return the number of samples in the requested
	split.
3. Implement `__getitem__(index)` to return an `(image, label)` pair.
4. Return images as float tensors with shape `[channels, height, width]` and
	values in the `[0, 1]` range. Repeat grayscale channels to produce three
	channels for the pretrained models.
5. Apply `self.transform` to each image when it is not `None`.
6. Update the `custom=True` branch in `build_dataloaders()` to instantiate
	your subclass.

For example, a small in-memory dataset can be written as:

```python
class MyDataset(CustomDataset):
	def __init__(self, data_root, train, transform=None):
		super().__init__(data_root, train, transform)
		self.samples = [(torch.zeros(3, 32, 32), 0), (torch.ones(3, 32, 32), 1)]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		image, label = self.samples[index]
		if self.transform is not None:
			image = self.transform(image)
		return image, torch.tensor(label, dtype=torch.int64)
```

The `train` attribute indicates whether the training or test split is being
loaded, and `data_root` points to the local `data/` directory. The custom
branch can then be exercised through the loader with:

```python
build_dataloaders(dataset_name="MyDataset", custom=True)
```

`dataset_name` remains the name used by the torchvision branch; the custom
branch uses the subclass selected in `build_dataloaders()`.

## Interpreting Results

Each experiment records the available device, training configuration, and
metric histories in `report.json`. Plots in the same report directory make it
possible to inspect learning curves and compare model behavior across runs.

Because this is an exploratory setup, results should be treated as directional
until experiments use fixed seeds, larger dataset splits, repeated trials, and
explicit comparisons between clean and corrupted inputs.

## Contributing Experiments

Useful contributions are small, well-isolated experiments that make one
bio-inspired hypothesis testable. When adding a mechanism or benchmark,
document:

1. The biological or neuroscientific motivation.
2. The model interface and computational cost.
3. The baseline and control condition.
4. The datasets, corruption settings, and evaluation metrics.
5. The limitations and the conditions under which the result was obtained.

## License

No license has been declared yet.