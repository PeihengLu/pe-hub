[//]: # (data/Readme)
# Data 

## OPED

### Training

- With a lack of training script, the training format was defined by myself

### Inferencing

- For inferencing, OPED only requires three columns in a dataframe:
- `Target(47bp)`: Starting from 4bp upstream of the spacer
- `PBS`
- `RT`
- All sequences were read from $5'$ to $3'$ end of the PE, with the 47bp sequence being the wildtype sequence.
- The PBS and RT sequences are the reverse complement to the target sequence.

## Standardized Format

- It is the uniformed format storing only the essential sequence information of the PE system and the target loci.

### Columns

- `wt-sequence`: The target loci before edits, can be various length, depends on the source data
- `mut-sequence`: The target loci after edits are installed
- 
- 