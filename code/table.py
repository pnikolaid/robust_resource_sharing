from parameters import code_directory, mu

import tabula
import pickle

# code

# PDF file
pdf = code_directory + "ts_136213v120400p.pdf"

# Read the PDF and extract tables from pages 63 to 65
tables = tabula.read_pdf(pdf, pages="63-69", multiple_tables=False)

# Assuming Table 7.1.7.2.1-1 is the first table on page 63
table_data = tables[0]

# Store the table data in a list
table_list = table_data.values.tolist()
new_table = []
# Display the extracted table data
for row in table_list:
    row = [x for x in row if str(x) != 'nan']
    new_table.append(row)

# Find all rows of a specific I_TBS
table = []
for i_tbs in range(0, 34):
    temp_list = []
    for row in new_table:
        try:
            if int(row[0]) == i_tbs and int(row[1]) != int(row[0]) + 1:
                new_row = [mu * int(float(x)) for x in row if x != 'nan']  # consider mu x mu MIMO!
                new_row.pop(0)
                temp_list += new_row
        except ValueError:
            continue
    if len(temp_list) != 110:
        print("WTF", i_tbs)
        print("Size:", len(temp_list))
        print(temp_list)
    table.append(temp_list)

# store in original bits per ms values BUT WITH mu x mu MIMO
with open(code_directory + '3GPP-table.pkl', 'wb') as f:
    pickle.dump(table, f)
