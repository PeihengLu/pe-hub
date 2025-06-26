# PE Database

An PostgresSQL repository of prime editing related experimental data.

It contains the curated data from various studies containing on target prime editing efficiency using specific pegRNA and prime editor on a given target loci. 

## Usage

### Query and Export

The front end application supports querying using study published time, pe version, cell line, as well as the type of study.

To facilitate easier benchmark with current state of the art models, you can also specify the format required by the model to run, which are PRIDICT, PRIDICT 2.0, as well as DeepPrime 

### Contributing to the Database

Although I am trying my best to scour the internet for all the relevant data, I am sure there are many studies that I have missed. If you have data that you would like to contribute to the database, please convert it to the format specified below and submit a pull request.

#### Contribution Format

To start with, the metadata of the study should be included in the pull request, containing the following information for advanced search and filtering:

- `Study`: The name of the study that the data originated from
- `Published Time`: The time that the study was published
- `PE Version`: The version of the prime editor used in the study
- `Cell Line`: The cell line used in the study
- `Type`: The type of study, which can be either .......

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

- Study Name (P): the name of the model if the data originated from a study developing a machine learning model for efficiency prediction. Otherwise, the bibtext citation key of the study.
- Published Time: the time that the originating study was published
- Authors: the contributing authors of the study


### Datasets

High level information about each dataset

- Dataset ID (P): the unique identifier of the dataset
- Study Name (F): the name of the study that the dataset originated from
- PE Version: the version of the prime editor used in the study
- Cell Line: the cell line used in the study
- Type: the type of study, which can be `In Vitro`, `In Vivo`, `Library Screening` ....

### Sequences 

The target as well as the corresponding pegRNA sequence

- Dataset ID (F): the unique identifier of the dataset
- Sequence ID (P): the unique identifier of the sequence, hashed from the target loci sequence
- Loci ID: the unique identifier of the target loci, hashed from the protospacer sequence

Rest of the columns are the same as specified in the contribution format, including `WT Sequence`, `MT Sequence`, `protospacer Location`, `PBS Location`, `RT Location WT`, `RT Location MT`, `Efficiency`

## Citation

If you found this data repo useful in your study, please consider citing our publication: