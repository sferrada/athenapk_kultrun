import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import operator as opr
import statsmodels as sm
import matplotlib.pyplot as plt

class DataBase(object):
    def __init__(self, fname, engine='sas') -> None:
        self.name = fname
        self.dbms = engine
        self.data = self.fopen()

    def fopen(self):
        switcher = {
            'sas': pd.read_sas(self.name),
            # 'csv': pd.read_csv(self.name),
            # 'xslx': pd.read_excel(self.name)
        }
        df = switcher.get(self.dbms, None)
        return df

    def subset(self, filt, field):
        """
        Example
        -------
        To get the values of the field 'Name' where 'Age' is 14,
        i.e., `Df.loc[Df['Age'] == 14.0]['Name']`:

        >>> Df.subset('Age == 14.0', 'Name')
        """
        opers = {
            '+': opr.add,
            '-': opr.sub,
            '*': opr.mul,
            '/': opr.truediv,
            '%': opr.mod,
            '^': opr.xor,
            '<': opr.lt,
            '>': opr.gt,
            '<=': opr.le,
            '>=': opr.ge,
            '==': opr.eq,
            }

        key, op, val = filt.split()
        print(self.data.loc[opers[op](self.data[key], float(val))][field])

    def format(self) -> None:
        pass

    def sort(self) -> None:
        pass

    def barplot(self, x, y, theme='viridis', savefig=False) -> None:
        sns.set_theme(style='white', context='talk')

        # Set up the matplotlib figure
        width = len(self.data[x]) * 0.85
        f, ax = plt.subplots(1, 1, figsize=(width, 5))

        # Actual plot after setup
        sns.barplot(x=self.data[x], y=self.data[y], palette=theme, ax=ax)

        ax.set_xticklabels([l.decode('utf-8') for l in Df.data['Name']], rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

class Library(object):
    def __init__(self, name, path, engine='sas') -> None:
        print(f'{name, path, engine}')

Df = DataBase('/home/simonfch/usampy/examples/class_birthdate.sas7bdat')

Df.barplot('Name', 'Age', theme='magma')