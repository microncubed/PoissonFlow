import csv

def build_thermal_dictionaries():
    '''
    Builds up the dictionaries of thermal parameters

    Returns
    -------
    k np.ndarray: thermal conductivity array
    c np.ndarray: thermal conductivity array
    rho np.ndarray: thermal conductivity array
    '''
    k = {}
    c = {}
    rho = {}

    filename = "thermal_properties.csv"

    with open(filename,'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            k[row['material']]= float(row['thermal_conductivity'])
            c[row['material']]= float(row['specific_heat'])
            rho[row['material']]= float(row['density'])

        return k,c,rho