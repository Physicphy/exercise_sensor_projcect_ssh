# %%
seed = 7 # 42
import os
os.environ['PYTHONHASHSEED'] = '0'
# %%
print(hash("keras"))