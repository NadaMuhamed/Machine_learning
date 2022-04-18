import ast
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re


UNITS = {"s":"seconds", "m":"minutes", "h":"hours", "d":"days", "w":"weeks"}
def convert_to_seconds1(s):
    count = int(s[:-1])
    unit = UNITS[s[-1]]
    td = timedelta(**{unit: count})
    return td.seconds + 60 * 60 * 24 * td.days

def convert_to_seconds2(s):
    return int(timedelta(**{
        UNITS.get(m.group('unit').lower(), 'seconds'): float(m.group('val'))
        for m in re.finditer(r'(?P<val>\d+(\.\d+)?)(?P<unit>[smhdw]?)', s, flags=re.I)
    }).total_seconds())

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def dictionary_to_columns(X, colmn):
    temp = X[colmn].values
    X.drop(colmn, inplace=True, axis=1)
    s = [ast.literal_eval(d)['source'] for d in temp]
    d = [ast.literal_eval(d)['destination'] for d in temp]
    X['source'] = s
    X['destination'] = d
    return X

