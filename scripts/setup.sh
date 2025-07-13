# merge the chopped pridict data into one file
python chop_and_restore_pridict.py -a restore 
# split the deepprime excel file into multiple csv files
python split_deepprime_excel.py
# perform conversions from various data to std format
