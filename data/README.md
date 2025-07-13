# Data Note

## OPED

- For inferencing, OPED only requires three columns in a dataframe:
  - 'Target(47bp)'
    - Starting from 3bp upstream of the spacer
  - 'PBS'
  - 'RT'
- All sequences were read from $5'$ to $3'$ end, with the 47bp sequence being the wildtype sequence.
- The PBS and RT sequences are complementary to the target sequence, and also in the reverse order due to the 5' to 3' reading direction.

## Standardized Format

### Columns

- `wt-sequence`: 
- `mut-sequence`:
- 