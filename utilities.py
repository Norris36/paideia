import subprocess
from tqdm import tqdm
import sys

def check_pip_installation(modules):
    """
    Checks if each module in the list is installed, installs if not.
    """
    with tqdm(modules) as pbar:
        for module in pbar:
            pbar.set_description(f"Checking installation for {module}")
            try:
                __import__(module)
                #print(f"{module} is already installed.")
            except ImportError:
                #print(f"{module} is not installed. Installing now...")
                subprocess.check_call(["python", "-m", "pip", "install", module])

modules = ["pandas", "datetime", "regex", "pathlib", "pyperclip", "shutil",  "tqdm", "matplotlib.pyplot", "numpy"]
# check_pip_installation(modules)

from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils.dataframe import dataframe_to_rows

from datetime import datetime
import matplotlib.pyplot as plt
from random import randint
from pathlib import Path
from tqdm import tqdm 
import pandas as pd
import time as time
import regex as re
import numpy as np
import pyperclip
import pathlib
import shutil
from openai import AzureOpenAI
import json
import ast
import os

def copy_extras():
    source_path = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\util_extras.py"
    destination_path = "./util_extras.py"
    shutil.copyfile(source_path, destination_path)

##
## these are typical use, such as importing and regular cleaning
##

def import_dataframe(path):
    # write a quick test to see if the dataframe contains semicolons or commas
    # if it contains semicolons, then use the semicolon as the delimiter
    
    df = pd.read_csv(path)
    
    if df.columns[0] == ';':
        df = pd.read_csv(path, sep = ';')
        
    df = rename_columns(df)
    
    # now we have a dataframe with the correct delimiter
    # then it would be nice to ensure that all names are lowercase, and all spaces are replaced with underscores, and all special characters are removed
        
    return df

def log_if_camel_case(s):
    """
    Log if a string is in camel case.

    Args:
    s (str): The string to check.

    Returns:
    None
    """
    # Define a regular expression pattern for camel case
    pattern = re.compile(r'^(?:[a-z]+(?:[A-Z][a-z0-9]*)*|[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)*)$')
    
    # Check if the string matches the pattern
    if pattern.match(s):
        return True
    else:
        return False

def camel_to_snake(s):
    """
    Convert a camel case string to a snake case string.

    Args:
    s (str): The camel case string to convert.

    Returns:
    str: The converted snake case string.
    """
    # Insert a '_' before each uppercase letter and then convert the whole string to lowercase
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z]))', r'_\1', s).lower()

def rename_columns(df: pd.DataFrame, return_dict=False):
    """
    Rename the columns of a DataFrame and optionally return a dictionary of original to new column names.

    Args:
    df (pd.DataFrame): The DataFrame whose columns are to be renamed.
    return_dict (bool): If True, returns a dictionary mapping original column names to new column names.

    Returns:
    pd.DataFrame or (pd.DataFrame, dict): The DataFrame with renamed columns. If return_dict is True, also returns the dictionary.
    """
    removables = [' ', '(', ')', '.', ',', '-', '/', '?', '!', "'", '"']
    bad_names = ['name', 'product', 'sum']
    rename_dict = {}

    for col in df.columns:

        if log_if_camel_case(col):
            new_col = camel_to_snake(col)
        else: 
            new_col = col.lower()
            
        for r in removables:
            if r in [' ', '-']:
                new_col = new_col.replace(r, '_')
            else:
                new_col = new_col.replace(r, '')


        if new_col in bad_names:
            new_col += '_'

        if new_col == 'sku':
            new_col = 'product_sku'

        rename_dict[col] = new_col

    df.rename(columns=rename_dict, inplace=True)

    if return_dict:
        return df, rename_dict
    else:
        return df

def analog_count(dataframe, column, simple = True, normalize = False, plot = False):
    """This function returns a dataframe with the counts of the values in a column.
        IF simple = False, it also returns the cumulative counts and the cumulative percentage.
        IF normalize = True, it returns the percentage of the counts.

    Args:
        dataframe (pd.DataFrame): a dataframe containing the column.
        column (str): the name of the column
        simple (bool, optional): defining if it is a simple vc or with cumulative sums and percentages. Defaults to True.
        normalize (bool, optional): with normalised values. Defaults to False.

    Returns:
        _type_: _description_
    """
    y = dataframe[column].value_counts(normalize = normalize).rename_axis(column).reset_index(name='counts')
    
    if simple != True:
        y['h_counts'] = y['counts'].cumsum()
        y['g_counts'] = y['h_counts'] / y['counts'].sum()
    
    try:    
        if plot and simple :
            if simple == True:
                y['h_counts'] = y['counts'].cumsum()
                y['g_counts'] = y['h_counts'] / y['counts'].sum()
            
            my_plot = pareto_distribution(y)
            my_plot.show()                


    except Exception as e:
        print(e)
        return y
    return y

def calculate_column_overlap(df_a, df_b):
    """
    Calculate the percentage overlap between each pair of columns from two DataFrames based on unique values.

    Args:
    df_a (pd.DataFrame): The first DataFrame.
    df_b (pd.DataFrame): The second DataFrame.

    Returns:
    pd.DataFrame: A DataFrame in long format showing the overlap percentage between columns of df_a and df_b.
    """
    # Create a DataFrame to store the overlap data
    data_overlap = pd.DataFrame(index=df_a.columns, columns=df_b.columns)

    # Iterate over each column pair and calculate overlap
    for col_a in data_overlap.index:
        for col_b in data_overlap.columns:
            unique_value_column_a = set(df_a[col_a].unique())
            unique_value_column_b = set(df_b[col_b].unique())

            # Calculate the intersection and percentage overlap
            intersection = len(unique_value_column_a.intersection(unique_value_column_b))
            length = len(df_a)
            data_overlap.loc[col_a, col_b] = round((intersection / length) * 100, 2)

    # Reshape the data_overlap DataFrame to a long format
    data_overlap_long = data_overlap.stack().reset_index()
    data_overlap_long.columns = ['column_a', 'column_b', 'overlap']

    # Sort by overlap and reset the index
    data_overlap_long.sort_values('overlap', ascending=False, inplace=True)
    data_overlap_long.reset_index(drop=True, inplace=True)

    return data_overlap_long

def pareto_distribution(value_counts):
    
    index = value_counts.index
    values = value_counts.g_counts
    
    # Plotting the cumulative distribution
    plt.figure(figsize=(10, 6))
    plt.plot(index, values, label='Cumulative Distribution')

    # Adding percentile markers
    percentiles = [10, 25, 50, 75] + list(range(80, 101, 5))
    for percentile in percentiles:
        x_value = np.percentile(index, percentile)
        y_value = np.percentile(values, percentile)
        plt.scatter(x_value, y_value, color='red')  # Mark the percentile
        
        # Adjust text to display percentile and x_value, position bottom-right of the marker
        plt.text(x_value, y_value, f'{percentile}% ({x_value:.2f}, {y_value:.2f})', 
                fontsize=9, 
                verticalalignment='top',
                horizontalalignment='left',
                rotation=(360 - 25))

    # Enhancing the plot
    plt.xlabel('Index')
    plt.ylabel('Cumulative Sum Percentage')
    plt.title('Pareto Distribution')
    plt.grid(True)
    plt.legend()
    
    return plt

## These are functions for service offerings files 
##
##

def assert_runtime():
    # Specify the path of the directory that contains the files you are interested in
    repo_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\00_First Rotation\Admin_tasks\util repo'

    # Loop through each file in the directory
    for filename in os.listdir(repo_path):

        # Construct the full path of the file by joining the directory path and the filename
        file_path = os.path.join(repo_path, filename)

        # Check if the path is indeed a file and not a directory
        if os.path.isfile(file_path):
            
            # Get the timestamp of when the file was last modified
            timestamp = os.path.getmtime(file_path)

            # Convert the timestamp into a datetime object
            file_time = datetime.fromtimestamp(timestamp)

            # Get the current date and time
            now = datetime.now()

            # Check if the difference between the current time and the last modification time of the file is less than 8 hours
            # If it is, print False
            if (now - file_time).total_seconds() / 3600 < 8:
                return False
            # If the difference is more than 8 hours, print True
            else:
                pass
    return True

def write_new_file(path,lines):
    f = open(path, 'w')
    with open(path, "a") as file:
    # set a list of lines to add:
    #lines = ["Hey there!", "LearnPython.com is awesome!"]
    # write to file and add a separator
        file.writelines(s + '\n' for s in lines)
        file.close()

def copy_file(source_file, dest_path):
    shutil.copy(source_file,dest_path)

def use_able_files(list):
    # oh it would be nice to have a list of all the utilities folders on all of the files
    y = []

    for i in list:
        if "opt/anaconda3" in i or "__pycache__" in i or ".vscode" in i:
            pass
        else:
            y.append(i)

    return y

def get_current_path():
    # oh it would be nice to get path for the folder i'm working in 
    return str(pathlib.Path().absolute())

def get_downloads():
    home = str(Path.home())
    return os.path.join(home, 'Downloads')

def get_file_extension(path: str):
    return Path(path).suffix

def get_file_basename(path: str):
    return  Path(path).stem

def get_home():
    # returns the home folder of the user
    return str(Path.home())

def get_root():
    # returns the base folder for the PC
    return get_parent_dir(get_parent_dir(get_home()))

def hygin(path, query):
    """
    Searches for files that match the given query within the specified path.

    Parameters:
    path (str): The path to start the search from.
    query (str): The file name or pattern to search for.

    Returns:
    list: A list of files that match the query.
    """
    # Check if the provided path is a directory, raise an error if not
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    # List all directories and files in the provided path and add the path itself to the list
    top_paths = os.listdir(path) + [path]
    # Initialize an empty list to store paths of files that match the query
    matching_files = []

    # Initialize a progress bar to track and display the search progress
    with tqdm(total=len(top_paths), desc="Initializing search") as pbar:
        # Iterate over each directory in the list of directories and files
        for top_dir in top_paths:
            # Update the progress bar description with the current directory being searched
            pbar.set_description(f"Searching in {top_dir}")
            # Construct the full path of the current directory
            full_top_path = os.path.join(path, top_dir)
            # Check if the current path is a directory
            if os.path.isdir(full_top_path):
                # Use os.walk to generate the file names in the directory tree by walking either top-down or bottom-up
                for root, dirs, files in os.walk(full_top_path):
                    # Iterate over each file in the current directory
                    for f in files:
                        # Check if the query is a substring of the file name
                        if query in f:
                            # If the file matches the query, append its full path to the list of matching files
                            matching_files.append(os.path.join(root, f))
            # Update the progress bar after each directory is searched
            pbar.update(1)
            
    # Return the list of matching files
    return matching_files

def get_paths(path, query):
    return hygin(path, query)

def get_local_path(query):
    """ Search for files in the local directory and below

    Args:
        query (str): a string to be contain-searched with

    Returns:
        list / str: a path or a list of paths
    """
    path = get_current_path()

    paths = hygin(path = path, query = query)

    return paths 

def find_highest_file(directory, name):
    matches = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if name in filename:
                matches[filename] = root + "\\" + filename
    
    test_val = 0
    test_path = ""
    for i in matches.keys():
        x = re.findall(r"\(\d*\)",i)
        if len(x) > 0:
            y = re.findall(r"\d+",x[0])
            y = int(y[0])
            if test_val <= y:
                test_path = matches[i]
    return test_path

def update_file(file_name, check = True):
    # Updating all the files in the directory, with the same name as the file we are using, with the
    # latest edited file.
    
    if check == True:
        assertion = assert_runtime()
    else:
        assertion = True
    if assertion == True:
        #oh it would be nice to update all files, with a similar name based on the last edited file, in the directory

        #lets start by getting the directory
        base = get_home()
        base = os.path.dirname(base) # this gets the directory for the file, so one level over
        # now it would be nice with all the files, which match our file name
        paths = hygin(base, file_name)  # This returns a list of them with the directory from before

        # now we just have to check, we do not get anything we do not want
            
        y = [] # oh it would be nice to have a list of all the utilities folders on all of the files

        for i in paths:
            if "opt/anaconda3" in i or "__pycache__" in i or ".vscode" in i or "ithub" in i:
                print(i)
            else:
                if get_file_basename(i) == file_name.split('.')[0]:
                    y.append(i)


        # lets make sure that we take the latest file
        my_dict = {} 
        for i in y: # this runs through all the files we found
            my_dict[len(my_dict)] = {
                "path": i,  # this returns the path
                "time": os.stat(i).st_mtime # this get the modified time
            }   

        y1 = 0 # this will be needed to check the latest modified time
        y2 = ''
        for key in my_dict:
        #for i in my_dict: # this iterates through the files
            if y1 <= my_dict[key]['time']: #if the test value is less or equal to the current file, then replace the values 
                y1 = my_dict[key]['time']  # this is the time
                y2 = my_dict[key]['path'] # this is the path

        # oh now we actual got some data that we like - yay! 
        text_file = open(y2, "r") # now we opening the latest edited file for reading 
        base_data = text_file.read()# now we are reading it to a string
        base_lines = re.findall("(.*)\n",base_data) # now we are creating a list of all the lines 

        # lets make sure that we do not delete everything by accident
        test_value = 20 # this is the minimum amount of functions in the file

        # lets check that the file that we got, has some functions
        functions = [] #lets read it to a list
        for i in base_lines:
            if "def" in i:
                functions.append(i)
                

        # here we make the actual test
        if len(functions) > test_value:
            #print(str(len(functions)) + " " + str(test_value))
            for j in my_dict:
                x = my_dict[j]['path']
                write_new_file(x,base_lines)
            print("File updated complete, from this folder " + get_parent_name(y2))
        else:
            "You maybe deleting some functions is that on purpose"
        
        # ts stores the time in seconds
        ts = time.time()
        #ts = ts.split('.')
        ts = str(int(ts))
        repo_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\00_First Rotation\Admin_tasks\util repo'
        repo_path = os.path.join(repo_path, f'util_back_up_{ts}.py')  
        #print(repo_path)
        write_new_file(repo_path,base_lines)
    else:
        print('Runtime assertion failed, this should mean that the update has been done today')
        
def get_parent_name(path):
    x1 = str(Path(path).parent)
    
    return get_file_basename(x1)

def insert_time_stamp(path):
    time = os.stat(path).st_mtime
    time = str(time).split(".")
    time = time[0]
    return time

def get_latest(input):
    """
    It takes a list of file paths, and returns the path of the file with the latest modification time
    
    :param input: a list of file paths
    :return: The path of the latest file.
    """
    if isinstance(input,list):    
        last_path = ""
        last_time = 0.0
        for i in input:
            if os.stat(i).st_mtime >= last_time:
                last_path = i
                last_time = os.stat(i).st_mtime
            else:
                pass
        print(datetime.fromtimestamp(last_time))
        return last_path
    else:
        print(datetime.fromtimestamp(os.stat(input).st_mtime))
        return input

def get_parent_dir(path):
    path = Path(path)
    return str(path.parent)

def get_download_files(que):
    out = ""

    downloads = hygin(get_downloads(), que)

    for download in downloads:
        move_to_folder(download)
        out += get_file_basename(download) + "\n"
        
    print(out)

def get_function_names():
    file_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\utilities.py'
    with open(file_path, "r") as source:
        tree = ast.parse(source.read())
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return functions

def find_util_function(function_name):
    file_name = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\utilities.py'
    with open(file_name, 'r') as file:
        content = file.read()
        
    pattern = r"(def {}.*?)(?=def |\Z)".format(function_name)
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    return matches

def concatenate_code_cells(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code = ""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code += ''.join(cell['source']) + '\n'
    return code

def find_module_functions(s, module_name='my'):
    pattern = fr'{module_name}\.(\w+)'
    matches = re.findall(pattern, s)
    return matches

def find_all_util_calls(notebook_path, copy_to_clipboard=True):
    code = concatenate_code_cells(notebook_path)
    function_names = find_module_functions(code)
    function_codes = [find_util_function(name) for name in function_names]
    output = ""
    # Getting the functions that i care about
    for function in function_codes:
        output += ''.join(function) + '\n'
  
    relying_functions = get_function_names()
    
    # Getting all the function that the code i care about relies on
    for func in relying_functions:
        pattern = f'(?<!def ){func}'
        matches = re.findall(pattern, output)
        if len(matches) > 0:
            output += ''.join(find_util_function(func)) + '\n'

    # copying it 
    if copy_to_clipboard:
        pyperclip.copy(output)
        
    
    return output

## These are functions to manipulate the clipboard
##
##

def set_clipboard(text):
    pyperclip.copy(text)

##
## These are functions to create template notebooks and folders
##

def get_demi_god():
    """Reading a csv file, getting a string from a column, then adding a value to a column, and then writing the file back to the same location.
    Returns:
        str: a demi god from greek mythology
    """
    path = get_latest(hygin(r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\data_lake", 'demi_gods.csv'))
    df = pd.read_csv(path, index_col=0)
    df.sort_values(['uses', 'demi_god'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    max_var = df[df.uses == df.uses.min()].index.max()
    rand_var = randint(0, max_var)
    df.at[rand_var, 'uses'] += 1
    output = df[df.index == rand_var].demi_god.values[0]
    df.sort_values('demi_god', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path)
    return output

def get_city():
    france_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\Admin_tasks\french_cities.csv'

    df = pd.read_csv(france_path)

    index = df[df.used != 1].index.min()
    city = df.at[index, 'Commune']
    df.at[index, 'used'] = 1

    city = city.lower()

    df.to_csv(france_path, 
            index=False)

    return city

def get_national_name():
    """Reading a csv file, getting a string from a column, then adding a value to a column, and then writing the file back to the same location.
    Returns:
        str: a demi god from greek mythology
    """
    path = get_latest(hygin(r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\Data Cleaning", 'landsholds_navne'))
    df = pd.read_csv(path, index_col=0)
    df = df.sort_values(['used'])
    df = df.reset_index(drop=True)
    max_var = df[df.used == df.used.min()].index.max()
    rand_var = randint(0, max_var)
    df.at[rand_var, 'used'] += 1
    output = df[df.index == rand_var].last_name.values[0]
    df = df.sort_values('name_')
    df = df.reset_index(drop=True)
    df.to_csv(path)
    return output

def create_notebook(filename):
    if filename == "":
        filename = get_demi_god()

    src = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\00_First Rotation\template_folder\template.ipynb"
    dest = get_current_path() + "/" + filename + ".ipynb" 
    if os.path.isfile(dest):
        identical_files = len(hygin(get_current_path(), filename))
        dest = get_current_path() + "/" + filename + "(" + str(identical_files) + ").ipynb"
        shutil.copy(src,dest)
    else:
        shutil.copy(src,dest)
    print(filename + ".ipynb created")

def create_folder(foldername):
    if foldername == "":
        foldername = get_demi_god()
        
    folder_src = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\00_First Rotation\template_folder"
    folder_dest = os.path.join(get_current_path(), foldername)
    shutil.copytree(folder_src,folder_dest)

def move_to_folder(path: str):
    """
    > Move a file to the current directory, and rename it with a timestamp
    
    :param path: the path to the file you want to move
    """
    base = get_current_path()
    time = insert_time_stamp(path)
    name = get_file_basename(path)
    ext  = get_file_extension(path)
    #print(each)
    destname = name + "_" + time + ext
    dest = os.path.join(base, destname)
    os.rename(path,dest)

def check_files_modified_today(file1_path, file2_path):
    """
    Checks if the provided Excel files have been modified today.
    Returns True if both have been modified today, else returns False.
    """
    # Get the current date
    current_date = datetime.date.today()

    # Get the last modified time for each file and convert it to a date
    file1_modified_time = datetime.fromtimestamp(os.path.getmtime(file1_path)).date()
    file2_modified_time = datetime.fromtimestamp(os.path.getmtime(file2_path)).date()

    # Check if both files have been modified today
    return file1_modified_time == current_date and file2_modified_time == current_date

def create_french_material(brute_force=False, verbose = False):
    # Example usage:
    file1 = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\french\french_practice.xlsx"
    file2 = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\french\english_to_french.xlsx"

    python_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\french\daily_runner.py'
    
    if check_files_modified_today(file1, file2):
        
        print("Je pense que tu as déjà fait ton material de francais aujourd'hui")
    elif brute_force:
        print('Bonjour - je veux créer ton material de francais')
        exec(open(python_path).read())
    else:
            print('Bonjour - je veux créer ton material de francais')
            exec(open(python_path).read())

def read_keys_file():
    """
    Reads the keys.txt file and extracts the API key, Azure endpoint, and API version.
    It only reads the file if the variables are not already set.
    """
    global api_key, azure_endpoint, api_version
    try:
        api_key
    except NameError:
        api_key = None

    try:
        azure_endpoint
    except NameError:
        azure_endpoint = None

    try:
        api_version
    except NameError:
        api_version = None
    # Only read the file if the variables have not been set yet
    if api_key is None or azure_endpoint is None or api_version is None:
        # Construct the file path to the keys.txt file
        file_path = os.path.join(get_home(), "Desktop", "keys.txt")

        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read()

        # Split the content into lines
        lines = content.split(',')

        # Extract the API key from the first line
        api_key = lines[0].split('=')[1].strip().replace("'", '')

        # Extract the Azure endpoint from the second line
        azure_endpoint = lines[1].split('=')[1].strip().replace("'", '')

        # Extract the API version from the third line
        api_version = lines[2].split('=')[1].strip().replace("'", '')

def get_client():
    """
    Returns the AzureOpenAI client object.
    If the client object does not exist, it creates a new one using the specified API key, Azure endpoint, and API version.
    """
    global client

    # Ensure the keys are read before creating the client
    read_keys_file()

    # Create the client if it doesn't exist
    if 'client' not in globals():
        client = AzureOpenAI(api_key=api_key,
                             azure_endpoint=azure_endpoint,
                             api_version=api_version)
    return client


# now we want to write a function which reques
#ts a completion from the chat endpoint
def get_completion(messages, model='gpt-4'):
    """
    Generates a completion using the specified model and messages.

    Parameters:
        messages (list): A list of messages exchanged in the conversation.
        model (str): The name of the model to use for completion. Defaults to 'gpt-4'.

    Returns:
        completion: The generated completion object.
    """
    client = get_client()
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion

# now we want to write a function which creates a message object, it should have an input var so that if it is said to "chat" or "greece" or "italy" then it returns "is greece bigger than italy" 
def create_message(content = None, system = 'default'):
    """
    Creates a message for a chatbot conversation.

    Parameters:
    content (str): The content of the user's message.
    role (str): The role for the system message to be displayed.

    Returns:
    list: A list of dictionaries representing the message, with each dictionary containing the role ('system' or 'user') and the content of the message.
    """
    if system != 'default':
        system = get_system_message(role = system)
    else:
        system = system
    
    if content == 'chat' or content == 'greece' or content == 'italy':
        message = [{'role':'system', 'content': system},
                   {'role':'user', 'content': 'Is greece bigger than Italy?'}]
    else:
        message = [{'role':'system', 'content': system },
                   {'role':'user', 'content': content}]
        
    return message

def get_system_message(role = 'default'):
    """
    Returns a system message based on the role.

    Parameters:
        role (str): The role of the system message. Defaults to 'default'.

    Returns:
        str: The system message.
    """
    if role == 'default':
        system_message = 'You are a super nice assistant'
    elif role == 'assistant':
        system_message = """You are a personal executive assistant, providing high-level administrative support
                                You manage schedules, organize meetings, and handle communication
                                You aid in preparing meeting agendas, gathering information, and distributing materials
                                You collaborate to create engaging meeting titles
                                You coordinate logistics and ensure resources are in place
                                You take meeting minutes and follow up on action items
                                You prioritize tasks, handle confidential information, and demonstrate strong communication and organizational skills
                                Your proactive and detail-oriented approach supports the executive in their day-to-day responsibilities
                                
                            You will be working for Jens Bay, and will in all instances be writing in the first person, write it as if you are Jens Bay.
                        """
    return system_message

def df_to_excel_table(df, file_name):
    """
    Converts a DataFrame into an Excel table and saves it to an Excel file.
    
    Args:
    df (DataFrame): The pandas DataFrame to save to Excel.
    file_name (str): The base name of the file to save the Excel document.
    
    Returns:
    None
    """
    df = df.applymap(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

    wb = Workbook()
    ws = wb.active

    # Assuming humanise_df is a function you've defined to format your DataFrame
    # If not defined, it should be or remove this transformation.
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    # Creating a table at the appropriate dimensions
    table_ref = f"A1:{ws.dimensions.split(':')[1]}"
    tab = Table(displayName="Table1", ref=table_ref)
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                           showLastColumn=False, showRowStripes=True, showColumnStripes=True)
    tab.tableStyleInfo = style
    ws.add_table(tab)

    # Saving the workbook
    wb.save(f"{file_name}.xlsx")
    print(f'Successfully saved DataFrame as a table in {file_name}.xlsx')

def humanise_df(local_df):
    # This function takes a DataFrame and returns a DataFrame with humanised column names
    # So there is a lack of underscores and everything is titel cased
    for col in local_df.columns.tolist():
        new_col = transform_column_name(col=col)
        
        local_df.rename(columns = {col:new_col}, inplace = True)
    return local_df

def transform_column_name(col):
    """
    Transforms a column name by replacing underscores with spaces and modifying each word.
    Words with 2 or fewer characters are converted to uppercase, while longer words are converted to title case.

    Parameters:
    col (str): The original column name.

    Returns:
    str: The transformed column name.
    """
    # Replace underscores with spaces
    new_col = col.replace('_', ' ')
    
    # Split the column name into words and transform each word
    for word in new_col.split(' '):
        if len(word) <= 2:
            new_word = word.upper()
        else:
            new_word = word.title()
        
        # Print the original and transformed word (for debugging purposes)
        print(word, new_word)
        
        # Replace the original word with the transformed word in the column name
        new_col = new_col.replace(word, new_word)
    
    return new_col

def sort_files():
    """Sort the files in the current directory based on their file extensions.

    Returns:
        _type_: _description_
    """
    try:
        # Get the current working directory
        parent_dir = get_current_path()
                # Get the parent directory of the current directory
        parent_of_parent = os.path.dirname(parent_dir)

        # Get the second-to-last folder name
        second_to_last_folder = os.path.basename(parent_of_parent)
        # Validate the directory to ensure it's a subfolder of the specified folders
        valid_folders = ['Workspace', '00_First Rotation', '01_Second Rotation']
        error_message = f"The folder is not a subfolder to either 'workspace', '00_First Rotation', or '01_Second Rotation' {parent_dir}"
        assert second_to_last_folder in valid_folders, error_message

        # List all directories and files in the parent directory
        dirs = os.listdir(parent_dir)

        # Create an empty DataFrame
        df = pd.DataFrame()

        # Add a column 'dirs' to the DataFrame containing the list of directories and files
        df['dirs'] = dirs

        # Add a column 'full_path' to the DataFrame, which contains the full path of each directory or file
        df['full_path'] = df['dirs'].apply(lambda x: os.path.join(parent_dir, x))

        # Add a column 'file_ext' to the DataFrame, which extracts the file extension from each full path
        df['file_ext'] = df['full_path'].apply(lambda x: os.path.splitext(x)[1])

        # Define a dictionary to categorize files based on their extensions
        sorting = {
            'csv_files': ['.csv', '.xlsx', '.xml', '.json', '.xls', '.tsv'],
            'image_files': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
            'text_files': ['.txt', '.doc', '.docx', '.pdf'],
        }

        # Define a function to get the category key from the sorting dictionary based on the file extension
        def get_key(my_dict, val):
            for key, value in my_dict.items():
                if val in value:
                    return key
            return "key doesn't exist"


        # Map the file extensions to their respective categories using the get_key function
        df['file_type'] = df['file_ext'].apply(lambda x: get_key(sorting, x))

        # Ensure that there is a directory for each file type category in the sorting dictionary only if relevant files exist
        for key in sorting.keys():
            sorting_path = os.path.join(parent_dir, key)
            # Check if there are any entries in the DataFrame for this file type
            if df['file_type'].str.contains(key).any():
                if not os.path.exists(sorting_path):
                    os.mkdir(sorting_path)
                    
        
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            if row['file_type'] != "key doesn't exist":
                # Construct a new path for the file based on its type
                new_path = os.path.join(parent_dir, row['file_type'], row['dirs'])
                try:
                    # Move the file to the new path if it does not already exist
                    if not os.path.exists(new_path):
                        os.rename(row['full_path'], new_path)
                    else:
                        print(f"File {new_path} already exists. Skipping.")
                except Exception as e:
                    print(f"Error occurred while renaming file {row['full_path']} to {new_path}. Error: {e}")
    except AssertionError as error:
        print(error)
