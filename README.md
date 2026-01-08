# Counterfactual Fairness Evaluation under Graph Uncertainty

Framework for evaluating counterfactual fairness of machine learning classifiers under causal graph uncertainty using a multi-world approach.

## Installation

### Prerequisites
- Python 3.8+
- Java (required for Tetrad)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd simplified-counterfactual-fairness-graph-uncertainty
```

2. Create and activate virtual environment:
```bash
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. External dependencies:
   - Diffusion models: `external/DiffusionBasedCausalModels/`
   - VACA models: `external/DiffusionBasedCausalModels/VACA_modified/`
   - pytetrad: `src/causal_discovery/pytetrad`
   - Set up according to their respective README files

## Project Structure

```
simplified-counterfactual-fairness-graph-uncertainty/
├── src/
│   ├── causal_discovery/          # Causal discovery using Tetrad
│   ├── causality/                 # Causal models and counterfactual generation
│   ├── classification/            # Classifier training and evaluation
│   ├── dataset/                   # Dataset loading and preprocessing
│   ├── metrics/                   # Evaluation metrics
│   ├── graph/                     # Graph utilities
│   └── plot/                      # Visualization utilities
├── config/                        # Configuration files
├── data/                          # Dataset files
├── scripts/                       # Bash scripts for pipelines
├── notebooks/                     # Jupyter notebooks
├── external/                      # External dependencies
├── generate_mw_counterfactuals.py
├── fit_causal_models_gen_eval_counterfactuals.py
├── evaluate_classifiers_with_cf.py
└── plot_all_results.py
```
## Quick Start

```bash
# Generate causal worlds
python generate_mw_counterfactuals.py --dataset adult --knowledge med --num_samples 10

# Fit one causal model on first world
python fit_causal_models_gen_eval_counterfactuals.py \
    --model-type linear \
    --output-dir output/adult/med/linear \
    --world-indexes "0"

# Evaluate one classifier
python evaluate_classifiers_with_cf.py --dataset adult --classifier LR --knowledge med
```
## Usage

### 1. Generate Causal Worlds

Generate multiple causal graphs using bootstrap sampling:

```bash
python generate_mw_counterfactuals.py \
    --dataset adult \
    --knowledge med \
    --num_samples 100 \
    --num_workers 1
```

**Parameters**:
- `--dataset`: Dataset name (e.g., `adult`)
- `--knowledge`: Knowledge level (e.g., `med`)
- `--num_samples`: Number of bootstrap samples (default: 100)
- `--num_workers`: Number of parallel workers (default: 1)

**Output**: `output/{dataset}/{knowledge}/causal_worlds.pkl`

### 2. Fit Causal Models and Generate Counterfactuals

Fit causal models on each causal world and generate counterfactuals:

```bash
python fit_causal_models_gen_eval_counterfactuals.py \
    --model-type linear \
    --output-dir output/adult/med/linear \
    --world-indexes "0,1,2,3,4"
```

**Supported model types**:
- `linear`: Linear regression-based causal model
- `lgbm`: LightGBM-based causal model
- `diffusion`: Diffusion-based causal model
- `causalflow`: Causal flow model

**Parameters**:
- `--model-type`: Type of causal model to fit
- `--output-dir`: Output directory for results
- `--world-indexes`: Comma-separated list of world indexes (optional, processes all if not specified)
- `--config`: Path to configuration file (optional)

**Output**:
- `counterfactuals_world_{idx:03d}.csv` for each world
- `{model_type}_metrics.csv`

### 3. Evaluate Classifiers with Counterfactual Fairness

Evaluate classifiers on counterfactuals and compute fairness metrics:

```bash
python evaluate_classifiers_with_cf.py \
    --dataset adult \
    --classifier LR \
    --knowledge med \
    --tune-hyperparameters \
    --n-trials 50
```

**Supported classifiers**:
- `LR`: Logistic Regression
- `RF`: Random Forest
- `GB`: Gradient Boosting
- `FAIRGBM`: FairGBM classifier

**Parameters**:
- `--dataset`: Dataset name
- `--classifier`: Classifier type
- `--knowledge`: Knowledge level
- `--tune-hyperparameters`: Enable hyperparameter tuning (optional)
- `--n-trials`: Number of tuning trials (default: 50)
- `--output-dir`: Custom output directory (optional)

**Output**:
- `classifier_evaluation.json`
- `cf_evaluation.csv`

### 4. Generate Visualizations

```bash
python plot_all_results.py
```

Generates visualizations for graph uncertainty, score variance, counterfactual metrics, and model performance.

## Complete Pipeline Example

```bash
# Step 1: Generate causal worlds
python generate_mw_counterfactuals.py --dataset adult --knowledge med --num_samples 100

# Step 2: Fit causal models
python fit_causal_models_gen_eval_counterfactuals.py --model-type linear --output-dir output/adult/med/linear
python fit_causal_models_gen_eval_counterfactuals.py --model-type lgbm --output-dir output/adult/med/lgbm
python fit_causal_models_gen_eval_counterfactuals.py --model-type causalflow --output-dir output/adult/med/causalflow
python fit_causal_models_gen_eval_counterfactuals.py --model-type diffusion --output-dir output/adult/med/diffusion

# Step 3: Evaluate classifiers
python evaluate_classifiers_with_cf.py --dataset adult --classifier LR --knowledge med
python evaluate_classifiers_with_cf.py --dataset adult --classifier RF --knowledge med
python evaluate_classifiers_with_cf.py --dataset adult --classifier GB --knowledge med
python evaluate_classifiers_with_cf.py --dataset adult --classifier FAIRGBM --knowledge med

# Step 4: Generate visualizations
python plot_all_results.py
```

## Output Structure

```
output/
└── {dataset}/
    └── {knowledge}/
        ├── causal_worlds.pkl              # All causal worlds
        ├── mw_counterfactuals.pkl         # Multi-world counterfactuals
        ├── {causal_model}/                # Per causal model
        │   ├── counterfactuals_world_{idx}.csv
        │   └── {causal_model}_metrics.csv
        ├── {classifier}/                  # Per classifier
        │   ├── classifier_evaluation.json
        │   └── cf_evaluation.csv
        └── classification/                 # Aggregated results
            └── {classifier}/
                ├── classifier_evaluation.json
                └── cf_evaluation.csv
```

## Configuration

### Causal Model Configuration

`config/causal_models_adult.yaml`:
```yaml
linear:
  random_state: 42

lgbm:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100

diffusion:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 64

causalflow:
  hidden_features: [128, 128]
  epochs: 200
  lr: 0.001
  batch_size: 512
```

### Classifier Configuration

`config/classifiers/`:
- `logistic_regression.yaml`
- `random_forest.yaml`
- `gradient_boosting.yaml`
- `fairgbm.yaml`

### Hyperparameter Tuning

`config/tuning/tuning_config.yaml`: Search spaces, objectives, trials, cross-validation settings

### Causal Discovery Configuration

`causal_discovery_params/boss.yaml`: Tetrad algorithm selection, search parameters, knowledge constraints

### Knowledge Files

`data/{dataset}/{knowledge}_knowledge.txt`: Temporal ordering, forbidden/required edges

Format follows Tetrad format's:
```
/knowledge
addtemporal
0* age native-country race sex
1 capital-gain capital-loss education ...
forbiddirect
requiredirect
```

## Supported Models and Datasets

### Causal Models
- **Linear**: Linear regression-based SCM (DoWhy GCM)
- **LGBM**: LightGBM-based causal model
- **Diffusion**: Diffusion-based causal model
- **CausalFlow**: Causal flow model (normalizing flows)

### Classifiers
- **LR**: Logistic Regression
- **RF**: Random Forest
- **GB**: Gradient Boosting
- **FAIRGBM**: FairGBM with fairness constraints
- **FairLearn**: Post-processing methods (demographic parity, equalized odds, equal opportunity, predictive equality)

### Datasets
- `adult`: UCI Adult dataset
- `compas`: COMPAS recidivism dataset
- `bank_marketing`: Bank marketing dataset
- `credit_card`: Credit card default dataset
- `law_school`: Law school admissions dataset
- `diabetes_hospital`: Diabetes hospital readmission dataset
- `synthetic`: Synthetic datasets

Each dataset includes:
- `metadata.json`: Feature types
- `{knowledge}_knowledge.txt`: Domain constraints

## Scripts and Examples

### Bash Scripts

`scripts/`:
- `fit_causal_models_generate_cfs.bash`: Batch processing of causal models
- `evaluate_classifiers_with_cf.bash`: Batch classifier evaluation
- `slurm_fit_causal_models_generate_cfs.bash`: SLURM cluster submission




## Metrics

### Counterfactual Fairness Metrics
- **PSR (Positive Switch Rate)**: Proportion of negative predictions that switch to positive in counterfactual
- **NSR (Negative Switch Rate)**: Proportion of positive predictions that switch to negative in counterfactual

### Counterfactual Quality Metrics
- Density, coverage, statistical similarity

### Model Metrics
- Performance: accuracy, precision, recall, F1
- Group fairness: demographic parity, equalized odds

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `networkx`: Graph operations
- `dowhy`: Causal inference
- `fairlearn`: Fairness metrics
- `mlflow`: Experiment tracking
- `optuna`: Hyperparameter optimization
- `py-tetrad`: Tetrad causal discovery wrapper
- `torch`: Deep learning (diffusion models)
- `causalflows`: Causal flow models
