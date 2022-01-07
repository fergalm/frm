from frm.support import npmap
import numpy as np 

LEVEL_1 = {
    0: "Agriculture, Forestry, Fishing",
    1: "Mining & Construction",
    2: "Manufacturing",
    3: "Manufacturing",
    4: "Trans, Comms, Energy, Sanitation",
    5: "Trade (Wholesale/Retail)",
    6: "Finance, Insurance, Real Estate",
    7: "Services",
    8: "Services",
    9: "Public Adminstration",
}

LEVEL_2 = {
    1: "Agricultural Production - Crops",
    2: "Agricultural Production - Livestock",
    7: "Agricultural Services",
    8: "Forestry",
    9: "Fishing, Hunting and Trapping",

    10: "Metal Mining",
    12: "Bituminous Coal and Lignite Mining",
    13: "Oil and Gas Extraction",
    14: "Mining and Quarrying of Nonmetallic Minerals, except Fuel",

    15: 'Building Construction General Contractors and Operative Builders',
    16: 'Heavy Construction other than Building Construction Contractors',
    17: 'Construction Special Trade Contractor',

    20: 'Food and Kindred Products',
    21: 'Tobacco Products',
    22: 'Textile Mill Products',
    23: 'Apparel and other Finished Products Made from Fabrics and Similar Materials',
    24: 'Lumber and Wood Products, except Furniture',
    25: 'Furniture and Fixtures',
    26: 'Paper and Allied Products',
    27: 'Printing, Publishing, and Allied Industries',
    28: 'Chemicals and Allied Products',
    29: 'Petroleum Refining and Related Industries',

    30: 'Rubber and Miscellaneous Plastics Products',
    31: 'Leather and Leather Products',
    32: 'Stone, Clay, Glass, and Concrete Products',
    33: 'Primary Metal Industries',
    34: 'Fabricated Metal Products, except Machinery and Transportation Equipment',
    35: 'Industrial and Commercial Machinery and Computer Equipment',
    36: 'Electronic and other Electrical Equipment and Components, except Computer Equipment',
    37: 'Transportation Equipment',
    38: 'Measuring, Analyzing, and Controlling Instruments; Photographic, Medical and Optical Goods; Watches and Clocks',
    39: 'Miscellaneous Manufacturing Industries',

    40: "Railroad Transportation",
    41: "Local and Suburban Transit and Interurban Highway Passenger Transportation",
    42: "Motor Freight Transportation and Warehousing",
    43: "United States Postal Service",
    44: "Water Transportation",
    45: "Transportation by Air",
    46: "Pipelines, except Natural Gas",
    47: "Transportation Services",
    48: "Communications",
    49: "Electric, Gas and Sanitary Services",

    50: "Wholesale Trade-Durable Goods",
    51: "Wholesale Trade-Nondurable Goods",

    52: "Building Materials, Hardware, Garden Supply, and Mobile Home Dealers",
    53: "General Merchandise Stores",
    54: "Food Stores",
    55: "Automotive Dealers and Gasoline Service Stations",
    56: "Apparel and Accessory Stores",
    57: "Home Furniture, Furnishings, and Equipment Stores",
    58: "Eating and Drinking Places",
    59: "Miscellaneous Retail",

    60: "Depository Institutions",
    61: "Non-Depository Credit Institutions",
    62: "Security and Commodity Brokers, Dealers, Exchanges, and Services",
    63: "Insurance Carriers",
    64: "Insurance Agents, Brokers and Service",
    65: "Real Estate",
    67: "Holding and other Investment Offices",

    70: "Hotels, Rooming Houses, Camps, and other Lodging Places",
    72: "Personal Services",
    73: "Business Services",
    75: "Automotive Repair, Services, and Parking",
    76: "Miscellaneous Repair Services",
    78: "Motion Pictures",
    79: "Amusement and Recreation Services",
    80: "Health Services",
    81: "Legal Services",
    82: "Educational Services",
    83: "Social Services",
    84: "Museums, Art Galleries, and Botanical and Zoological Gardens",
    86: "Membership Organizations",
    87: "Engineering, Accounting, Research, Management, and Related Services",
    88: "Private Households",
    89: "Miscellaneous Services",

    91: "Executive, Legislative, and General Government, except Finance",
    92: "Justice, Public Order, and Safety",
    93: "Public Finance, Taxation, and Monetary Policy",
    94: "Administration of Human Resource Programs",
    95: "Administration of Environmental Quality and Housing Programs",
    96: "Administration of Economic Programs",
    97: "National Security and International Affairs",
    99: "Nonclassifiable Establishments",
}


def sic_code_to_category(code, level=1, labels=False):
    
    code = np.array(code) 
    if np.any(code > 9999):
        raise ValueError("All sic codes must be less than 10000")

    if level > 2:
        raise ValueError("Max level implemented is 2")

    level = 10**(4-level)
    code = np.floor(code / level).astype(int)

    dtype = int
    if labels:
        dtype="|S32"
    
    classifications = np.zeros_like(code, dtype=dtype)
    class_boundaries = [10, 15, 20, 40, 50, 52, 60, 70, 90, 100]
    for i in range(len(class_boundaries)-1):
        lwr = class_boundaries[i]
        upr = class_boundaries[i+1]
        idx = (lwr <= code) & (code < upr)

        print(i, np.sum(idx))
        if labels:
            classifications[idx] = SIC_CATEGORIES[i] 
        else:
            classifications[idx] = i 


    return classifications


def sic_code_to_string(code, level=2, numeric=False):
    """Translate a SIC code to a human readable string.
    
    The Standard Industrial Classification is a New Deal era approach to classifying US industry
    using a 4 letter code. The code is obsolete, having been replaced by NAICS, but is still
    in common usage.

    The code is hierarchical, with similar industries sharing the same first digit of the code,
    and very similar industries sharing the first 3 digits. 

    See https://en.wikipedia.org/wiki/Standard_Industrial_Classification

    Inputs
    -----------
    code
        (int, or array) A 4 digit number of the company
    level
        (int) How specific an industrial classification is needed. For example 2000s are all
        manufacturing industires, while 32 is glass, 321 is flat glass, and 322 is glassware,
        and 3229 is "Pressed and blown glass, not elsewhere classfied".
        Currently, only levels 1 and 2 are implemented in this function.

    Optional Inputs
    -----------------
    numeric
        (bool) If True return the truncated SIC code instead of the string. For example, 4321 might get truncated to 43

    Returns
    ---------
    A numpy array of strings of length len(code)
    """
    code = np.atleast_1d(code) 
    if np.any(code > 9999):
        raise ValueError("All sic codes must be less than 10000")

    if level < 1:
        raise ValueError("Min level is 1")
    if level > 2:
        raise ValueError("Max level implemented is 2. Max possible level is 4")

    categories = [None, LEVEL_1, LEVEL_2][level]
    level = 10**(4-level)
    code = np.floor(code / level).astype(int)

    if numeric:
        return code 


    labels = npmap(lambda x: categories[x], code)
    return labels

