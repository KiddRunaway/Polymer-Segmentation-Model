import numpy as np
import pandas as pd
import scipy as sp

def excel_reader(file_name, lower_range, upper_range, sheet_name="Merged"): 
    """
    Reads an Excel file and processes data from a specific sheet, focusing on a retention time range.

    Parameters:
    - file_name (str): The name of the Excel file.
    - lower_range (float): The lower bound of the retention time range.
    - upper_range (float): The upper bound of the retention time range.
    - sheet_name (str): The name of the sheet containing the data (default is "Merged").

    Returns:
    - pd.DataFrame: A processed DataFrame with data within the specified retention time range.
    """
    file = pd.read_excel(file_name, sheet_name=sheet_name).iloc[1:, :]
    
    locationax, locationay, locationbx, locationby = 0, 1, 2, 3
    N = len(file.iloc[:, 0])
    for k in range(0, len(file.iloc[0, :]), 5):
        temp_df = file.iloc[:, k+locationay]
        temp_array = temp_df.to_numpy().astype(float) # array since smoothing function doesnt like dataframes
        file.iloc[:, k+locationay] = sp.ndimage.gaussian_filter(temp_array, sigma=20) # smoothing function
        temp_df = file.iloc[:, k+locationby]
        temp_array = temp_df.to_numpy().astype(float)
        file.iloc[:, k+locationby] = sp.ndimage.gaussian_filter(temp_array, sigma=20)
        minindex, maxindex = np.argmin(np.abs(file.iloc[:, k+locationbx]-lower_range)), np.argmin(np.abs(file.iloc[:, k+locationbx]-upper_range))

        #retention time range function
        temp_arrayx, temp_arrayy = file.iloc[minindex:maxindex, k+locationax], file.iloc[minindex:maxindex, k+locationay]
        temp_arrayx = np.concatenate((np.full((minindex,), np.nan), temp_arrayx, np.full(((N-maxindex),), np.nan)))
        temp_arrayy = np.concatenate((np.full((minindex,), np.nan), temp_arrayy, np.full(((N-maxindex),), np.nan)))
        file.iloc[:, k+locationax], file.iloc[:, k+locationay] = temp_arrayx, temp_arrayy
        temp_arrayx, temp_arrayy = file.iloc[minindex:maxindex, k+locationbx], file.iloc[minindex:maxindex, k+locationby]
        temp_arrayx = np.concatenate((np.full((minindex,), np.nan), temp_arrayx, np.full(((N-maxindex),), np.nan)))
        temp_arrayy = np.concatenate((np.full((minindex,), np.nan), temp_arrayy, np.full(((N-maxindex),), np.nan)))
        file.iloc[:, k+locationbx], file.iloc[:, k+locationby] = temp_arrayx, temp_arrayy

    return file

def delete_nan(file, format=pd.DataFrame): 
    """
    Removes NaN values from arrays, lists, or dataframes, with the option to choose the output format.
    
    Parameters:
    - file: pandas.DataFrame, list, or numpy.ndarray
      The input data structure containing NaN values.
    - format: type, optional (default=pd.DataFrame)
      The desired output format. Can be pd.DataFrame, list, or np.ndarray.

    Returns:
    - output: pd.DataFrame, list, or np.ndarray
      The data structure with NaN values removed, formatted as specified.

    Raises:
    - TypeError: If the input file is not of type pandas.DataFrame, list, or numpy.ndarray.

    Example:
    >>> data_frame = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    >>> cleaned_data = delete_nan(data_frame, format=list)
    >>> print(cleaned_data)
    [[1.0, 4.0], [2.0, 6.0]]
    """
    if isinstance(file, pd.DataFrame) == True:
        file = file.to_numpy().astype(float)
    elif isinstance(file, list) == True:
        file = np.array(file)
    elif isinstance(file, np.ndarray) == False:
        raise TypeError("file needs to be a pandas.Dataframe, list or numpy.ndarray type object")
    storage = []

    if len(np.shape(file)) == 1:
        for i in range(len(file)):
            storage.append(file[i])
        output = np.array(storage)
    else:
        lenrow, lencol = np.shape(file)[0], np.shape(file)[1]
        FirstRun = False
        for i in range(lencol):
            storage = []
            for j in range(lenrow):
                if np.isnan(file[j, i]) == False:
                    storage.append(file[j, i])
            if FirstRun == False:
                output = np.array(storage)
                len_storage = len(storage)
                FirstRun = True
            elif FirstRun == True:
                if len(storage) == 0:
                    output = np.vstack((output, np.zeros(len_storage)))
                else:
                  output = np.vstack((output, np.array(storage)))
        output = output.T
    if format == pd.DataFrame:
        output = pd.DataFrame(output)
    elif format == list:
        output = list(output)
    elif format == np.ndarray:
        pass 
    return output

def fractionater(file): 
    """
    Converts GPC graph into fractions and removes any values less than zero.

    Parameters:
    - file (pd.DataFrame): Pandas DataFrame representing the GPC graph.

    Returns:
    - pd.DataFrame: Modified DataFrame with fractions and non-negative values.
    """
    if isinstance(file, pd.DataFrame) == False:
        raise TypeError("file object needs to be a pandas DataFrame.")
    else:
        locationay, locationby = 1, 3
        for k in range(0, len(file.iloc[0, :]), 5): 
            for l in range(len(file.iloc[:, 0])): #set any values less than zero to zero
                if file.iloc[l, k+locationay] < 0:
                    file.iloc[l, k+locationay] = 0
                if file.iloc[l, k+locationby] < 0:
                    file.iloc[l, k+locationby] = 0
            suma= np.sum(file.iloc[:, k+locationay])
            sumb = np.sum(file.iloc[:, k+locationby])
            file.iloc[:, k+locationay] = file.iloc[:, k+locationay] / suma #convert into fractions
            file.iloc[:, k+locationby] = file.iloc[:, k+locationby] / sumb
    return file

def calibration(ret_time, standard="PMMA"):
    """
    Converts retention time to molecular weight (Da) based on the chosen calibration standard ("PMMA" or "PS").

    This function takes a retention time as input and, based on the specified calibration standard ("PMMA" or "PS"), converts it to molecular weight (Da) using pre-determined calibration data. The calibration data, stored in an Excel file, is loaded for the selected standard, and mathematical transformations are applied to derive molecular weights for two calibration standards ('A' and 'B'). The resulting molecular weights are returned as a dictionary.

    Parameters:
    - rettime (float): Retention time to be converted.
    - standard (str): Calibration standard, either "PMMA" or "PS".

    Returns:
    - dict: Dictionary containing molecular weights for standards 'A' and 'B'.
    """
    calPM = pd.read_excel("Calibration2.xlsx", "PMMA", header=None, skiprows=1)
    calPSA = pd.read_excel("Calibration2.xlsx", "PS_A", header=None, skiprows=1)
    calPSB = pd.read_excel("Calibration2.xlsx", "PS_B", header=None, skiprows=1)
    mw_array = {}
    if standard == "PMMA":
        Ax, Ay = calPSA.iloc[:, 0].to_numpy().astype(np.float64), calPSA.iloc[:, 1].to_numpy().astype(np.float64) #PS STANDARDS, 5th order polynomial
        Ay = -np.log10(Ay)
        calA = np.poly1d(np.polyfit(Ax, Ay, 5))
        mw_array["A"] = 10**(-calA(ret_time))
        Bx, By = calPM.iloc[:, 0].to_numpy().astype(np.float64), calPM.iloc[:, 1].to_numpy().astype(np.float64)
        By = -np.log10(By)
        calB = sp.interpolate.interp1d(Bx, By, kind="linear")
        B = calB(ret_time)
        mw_array["B"] = 10**(-B)
    elif standard == "PS":
        Ax, Ay = calPSA.iloc[:, 0].to_numpy().astype(np.float64), calPSA.iloc[:, 1].to_numpy().astype(np.float64)
        Ay = -np.log10(Ay)
        calA = sp.interpolate.interp1d(Ax, Ay, kind="linear")
        mw_array["A"] = 10**(-calA(ret_time))
        Bx, By = calPSB.iloc[:, 0].to_numpy().astype(np.float64), calPSB.iloc[:, 1].to_numpy().astype(np.float64)
        By = -np.log10(By)
        calB = sp.interpolate.interp1d(Bx, By, kind="linear")
        mw_array["B"] = 10**(-calB(ret_time))
    return mw_array

def ret_to_logmw_file(file, standard):
    """
    Converts the retention time axis of a standard file type (assumed to be in a pandas DataFrame) into the molecular weight.
    
    Parameters:
    - file: pandas DataFrame
        The input file with retention time data.
    - standard: str
        The calibration standard to use ("PMMA" or "PS").

    Returns:
    - file: pandas DataFrame
        The modified file with the retention time axis converted to molecular weight.
    """
    if isinstance(file, pd.DataFrame) == False:
        raise TypeError("file object needs to be a pandas DataFrame.")
    locationax, locationay, locationbx, locationby = 0, 1, 2, 3
    lenrow, lencol = np.shape(file)[0], np.shape(file)[1]
    for k in range(0, lencol, 5):
        temp_arrayax, temp_arraybx = file.iloc[:, k+locationax], file.iloc[:, k+locationbx]
        file.iloc[:, k+locationax], file.iloc[:, k+locationbx] = np.log10(calibration(temp_arrayax, standard)["A"]), np.log10(calibration(temp_arraybx, standard)["B"])
    return file

def interpolator(file, mw_array, N_max): 
    """
    Convert GPC graph into equal length of the N array using interpolation.

    Parameters:
    - file (pd.DataFrame): GPC data as a DataFrame.
    - mw_array (dict): Dictionary containing molecular weight arrays for standards A and B.
    - N_max (int): Desired length of the output arrays.

    Returns:
    - pd.DataFrame: Interpolated GPC data with equal-length N arrays.
    """
    if isinstance(file, pd.DataFrame) == False:
        raise TypeError("file object needs to be a pandas DataFrame.")
    
    discrete_mw_array = {}
    discrete_mw_array["A"] = np.linspace(np.min(mw_array["A"]), np.max(mw_array["A"]), N_max)
    discrete_mw_array["B"] = np.linspace(np.min(mw_array["B"]), np.max(mw_array["B"]), N_max)
    N = np.arange(1, N_max+1, 1)
    new_file = pd.DataFrame()

    locationax, locationay, locationbx, locationby = 0, 1, 2, 3
    lenrow, lencol = np.shape(file)[0], np.shape(file)[1]

    lambdaAx = lambda nr: f'Ax{nr}'
    lambdaAy = lambda nr: f'Ay{nr}'
    lambdaBx = lambda nr: f'Bx{nr}'
    lambdaBy = lambda nr: f'By{nr}'
    lambdaC = lambda nr: f'C{nr}'

    for k in range(0, lencol, 5):
        temp_arrayx, temp_arrayy = file.iloc[:, k+locationax], file.iloc[:, k+locationay]
        func = sp.interpolate.interp1d(x=temp_arrayx, y=temp_arrayy, kind='quadratic')
        temp_arrayy = func(discrete_mw_array["A"])
        namex, namey = lambdaAx(int(k/5 + 1)), lambdaAy(int(k/5 + 1))
        new_file[namex], new_file[namey] = N, temp_arrayy / np.sum(temp_arrayy)
        temp_arrayx, temp_arrayy = file.iloc[:, k+locationbx], file.iloc[:, k+locationby]
        func = sp.interpolate.interp1d(x=temp_arrayx, y=temp_arrayy, kind='quadratic')
        temp_arrayy = func(discrete_mw_array["B"])
        namex, namey = lambdaBx(int(k/5 + 1)), lambdaBy(int(k/5 + 1))
        new_file[namex], new_file[namey] = N, temp_arrayy / np.sum(temp_arrayy)
        new_file[lambdaC(int(k/5 + 1))] = np.zeros(N_max)

    return new_file






















