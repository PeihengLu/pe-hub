# merge the chopped pridict data into one file
echo "Merging chopped pridict data..."
python scripts/chop_and_restore_pridict.py -a restore 
# split the deepprime excel file into multiple csv files
echo "Splitting DeepPrime Excel file to CSV files..."
python scripts/convert_deepprime_to_csv.py
# perform conversions from various data to std format

# install the required packages
pip install -r requirements.txt
# install current project
# pip install -e .