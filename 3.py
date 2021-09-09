import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import akshare as ak
import plotly_express as px

import csv

mydict = {}
with open('fund_name.csv', mode='r',encoding='utf_8') as inp:
    reader = csv.reader(inp)
    dict_from_csv = {rows[0]:rows[1] for rows in reader}

print(dict_from_csv)