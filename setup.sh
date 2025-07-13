# merge the chopped pridict data into one file
python scripts/chop_and_restore_pridict.py -a restore 
# split the deepprime excel file into multiple csv files
python scripts/split_deepprime_excel.py
# perform conversions from various data to std format

# install the required packages
pip install -r requirements.txt
# install current project
pip install -e .