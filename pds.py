import pandas as pd
import numpy as np
t2 = pd.Series([1, 23, 2, 2, 1], index=list("abcde"))
t2

# %%
temp_dict = {"name": "xiaohong", "age": 30, "tel": 10086}
t3=pd.Series(temp_dict)
t3

# %%

t=pd.DataFrame(np.arange(12).reshape(3,4),index=list("abc"),columns=list("WXYZ"))
t

#%%
d1={
    "name": ["xiaoming","xiaogang"],
    "age":[20,32],
    "tel": [10086,10010]
}

ta=pd.DataFrame(d1)
ta
