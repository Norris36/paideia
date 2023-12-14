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

from datetime import datetime
import matplotlib.pyplot as plt
from random import randint
from pathlib import Path
from tqdm import tqdm 
import pandas as pd
import time as time
import regex as re
import pyperclip
import pathlib
import shutil
import numpy as np
import json
import ast
import os

def copy_extras():
    source_path = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\util_extras.py"
    destination_path = "./util_extras.py"
    shutil.copyfile(source_path, destination_path)

## these are typical use, such as importing and regular cleaning
##
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
    repo_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\Admin_tasks\util repo'

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
    It takes a path and a query as input and returns a list of all the paths which matches the query
    
    :param path: the path to the directory you want to search
    :param query: the string you want to search for
    :return: A list of all the paths which matches the query
    """
    my_list = []
    for root, dirs, files in os.walk(path):
        for name in files:
                if query in name:
                    path = os.path.join(root,name)
                    my_list.append(path)
    
    
    if len(my_list) == 1:
        return my_list[0]
    else:   
        return my_list

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
                pass
            elif get_filename(i) == 'utilities.py':
                pass
            else:
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
        repo_path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\Admin_tasks\util repo'
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

    src = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\template_folder\template.ipynb"
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
        
    folder_src = r"C:\Users\jbay\OneDrive - GN Store Nord\Workspace\0_First Rotation\template_folder"
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

