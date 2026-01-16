# PE-DB: Prime Editing Database and Model Ensemble

A comprehensive platform for prime editing efficiency data management and model evaluation.

## Project Overview

This project consists of two main services:

### 1. PE Database Service
A standalone FastAPI application that serves prime editing efficiency data from various sources in standardized or model-specific formats.

**Features:**
- Load data from multiple datasets (DeepPrime, PRIDICT, PRIDICT2, etc.)
- Convert between different data formats
- Standardize data from various sources
- REST API for easy data access
- Docker support for easy deployment

### 2. PE Ensemble Service (Coming Soon)
An interface for training and evaluating state-of-the-art prime editing efficiency prediction models.

**Features:**
- Multiple model architectures support
- Unified training and evaluation pipeline
- Automatic data fetching from PE Database
- Model ensemble capabilities

## Project Structure

```
pe-db/
├── packages/
│   └── pe-common/              # Shared utilities package
│       ├── pe_common/
│       │   ├── constants.py    # Project-wide constants
│       │   ├── sequence_utils.py  # Sequence manipulation
│       │   └── features.py     # Feature calculations
│       └── setup.py
├── services/
│   ├── pe-db/                  # PE Database service
│   │   ├── app/
│   │   │   ├── main.py         # FastAPI application
│   │   │   └── data_prep/      # Data conversion modules
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── pe-ensemble/            # PE Ensemble service (TBD)
├── datasets/                    # Data directory
│   ├── raw/                    # Original datasets
│   └── standardized/           # Converted standardized data
├── vendor/models/              # Third-party model code
├── docker-compose.yml
└── setup-dev.sh               # Development setup script
```

## Quick Start

### Option 1: Local Development

1. **Setup environment:**
```bash
./setup-dev.sh
```

2. **Activate virtual environment:**
```bash
source venv/bin/activate
```

3. **Run PE Database service:**
```bash
cd services/pe-db
uvicorn app.main:app --reload
```

Access the API at: http://localhost:8000
API documentation: http://localhost:8000/docs

### Option 2: Docker

1. **Build and run with Docker Compose:**
```bash
docker-compose up pe-db
```

The service will be available at http://localhost:8000

## PE Database API Usage

### Get Data
```bash
# Get data in standardized format
curl "http://localhost:8000/api/data?cell_line=HEK293T&pe_system=PE2&source_model=dp&target_format=std"

# Get data in DeepPrime format
curl "http://localhost:8000/api/data?cell_line=A549&pe_system=PE2max&source_model=dp_ft&target_format=deepprime"
```

### List Available Datasets
```bash
curl "http://localhost:8000/api/datasets"
```

### Convert Data
```bash
curl -X POST "http://localhost:8000/api/convert?source=deepprime&cell_line=HEK293T&pe_system=PE2"
```

## Data Organization

The database contains curated data from various studies on prime editing efficiency: 

### Datasets Directory Structure
```
datasets/
├── raw/                    # Original data files
│   ├── deepprime/
│   ├── minsepie/
│   ├── pridict1/
│   └── pridict2/
└── standardized/          # Converted standardized format
    ├── deepprime/
    ├── pridict1/
    └── pridict2/
```

### Data Formats Supported
- **Standard Format**: Unified format for all datasets
- **DeepPrime Format**: Compatible with DeepPrime model
- **PRIDICT Format**: Compatible with PRIDICT model
- **PRIDICT2 Format**: Compatible with PRIDICT 2.0 model
- **OPED Format**: Compatible with OPED model

## Shared Package: pe-common

The `pe-common` package provides shared utilities used by both services:

### Constants
```python
from pe_common import DATA_ROOT, MODEL_ROOT, DEVICE
```

### Sequence Utilities
```python
from pe_common.sequence_utils import align_wt_mut_sequences, remove_padding
```

### Feature Calculations
```python
from pe_common.features import calculate_mfe, calculate_mt_wallace, calculate_gc_content
```

## Development

### Installing in Development Mode

Install the shared package:
```bash
pip install -e packages/pe-common
```

### Module Imports

With this structure, you can now reliably import modules:

```python
# Instead of relative imports
from pe_common import DATA_ROOT, DEVICE
from pe_common.sequence_utils import align_wt_mut_sequences
from pe_common.features import calculate_gc_content
```

No more unstable relative path imports!

## Contributing to the Database

Although I am trying my best to scour the internet for all the relevant data, I am sure there are many studies that I have missed. If you have data that you would like to contribute to the database, please convert it to the format specified below and submit a pull request.

#### Contribution Format

To start with, the metadata of the study should be included in the pull request, containing the following information for advanced search and filtering:

- `Study`: The name of the study that the data originated from
- `Published Time`: The time that the study was published, in YYYYMM format

For each dataset, you should indicate:

- `PE System`: The version of the prime editor used in the study
- `Cell Line`: The cell line used in the study
- `Dataset Type`: The type of study, which can be either `Library`(0), `Off-target`(1), `Endogenous`(2)

The data should be in the form of a csv file, containing the following columns:

- `WT Sequence`: The wild type sequence of the target loci
- `MT Sequence`: The mutated sequence of the target loci after prime editing
- `protospacer Location`: The relative index of the pegRNA in the WT and MT sequence, in the format of `start-end`, both inclusive
- `PBS Location`: The relative index of the PBS in the WT and MT sequence
- `RT Location WT`: The relative index of the RT in the WT sequence, note that this would be differet from the MT sequence if there is an insertion or deletion
- `RT Location MT`: The relative index of the RT in the MT sequence
- `Efficiency`: The efficiency of the prime editing, which is the percentage of the MT sequence in the total sequence

The rest of the columns are optional, but can be included if available:

- `Chromatin State`: The chromatin state of the target loci
- `Indel Percentage`: The percentage of indels in the total sequence

## Tables

The database is organized as follows:

### Studies

Storing high level information about the study that the dataset originated from

- `Study ID` (P): A unique identifier for the study originating a set of datasets 
- `Published Time`: the time that the originating study was published, in YYYYMM format
- `Authors`: the contributing authors of the study

### Datasets

High level information about each dataset

- `Dataset ID` (P): the unique identifier of the dataset
- `Study ID` (F): the name of the study that the dataset originated from
- `PE System`: the version of the prime editor used in the study
- `Cell Line`: the cell line used in the study
- `Study Type`: the type of study, which can be `Library`(0), `Off-target`(1), `Endogenous`(2)

### Sequence Tables



The target as well as the corresponding pegRNA sequence

- `Dataset ID` (F): the unique identifier of the dataset
- `Sequence ID` (P): the unique identifier of the sequence, hashed from the target loci sequence
- `Loci ID`: the unique identifier of the target loci, hashed from the protospacer sequence, useful for train-test-val split as pegRNA targeting the same loci are grouped 

Rest of the columns are the same as specified in the contribution format, including `WT Sequence`, `MT Sequence`, `protospacer Location`, `PBS Location`, `RT Location WT`, `RT Location MT`, `Efficiency`

# PE Ensemble

To try out the complete version of our app and create ensembles of your own, additional steps should be taken:

The included models need to be downloaded with `git submodule update --remote`

# Citation

If you found this data repo useful in your study, please consider citing our publication:
