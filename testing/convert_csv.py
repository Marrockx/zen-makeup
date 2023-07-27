import csv

# Function to read CSV file and exclude the first row
def read_csv_and_exclude_first_row(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the first row
        for row in csvreader:
            data.append(row)
    return data

