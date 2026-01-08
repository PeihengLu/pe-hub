# State of the art models

## Inferencing using the pre-trained models

### DeepPrime



### PRIDICT and PRIDICT 2.0



### OPED
- For inferencing, OPED only requires three columns in a dataframe:
  - 'Target(47bp)'
  - 'PBS'
  - 'RT'
- Path to the trained weights:
  - `/home/peiheng/development/pe-db/models/OPED/pegRNA_PredictingCodes`
- The pickled model was training using pytorch 1.18.1, compatibility issue will occur with pytorch 2.0.0 and above.

