# Phenopack

phenopack is a Python package based on the paper "Blooming Insights: Deep Learning Reveals Key Climatic Drivers and Shifting Sensitivities of Phenology" by Lahlou et al. (2023). It provides tools for forecasting, analyzing, and visualizing phenological data, focusing on the impact of different climatic factors on plant cycles.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Phenopack.

```bash
pip install phenopack
```

## Usage

```python
import phenopack


```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)


# 1) setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,docs]"

# 2) run tests
pytest -q

# 3) run example workflow
your-cli run --config src/phenofusion/config/default.yaml --out out/
