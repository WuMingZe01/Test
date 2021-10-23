import pandas as pd
import datetime
import numpy as np
import tushare as ts

import akshare as ak

fund_em_open_fund_rank_df = ak.fund_em_open_fund_rank(symbol="全部")
fund_em_open_fund_rank_df = pd.DataFrame(fund_em_open_fund_rank_df)
fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.sort_values(by='近1年', ascending=False)
fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.reset_index(drop=True)
print(fund_em_open_fund_rank_df)