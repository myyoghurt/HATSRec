## Model implementation for the HATS mechanism

Code for my master thesis: Time Lag Aware Sequential Recommendation
with Hierarchical Self-Attention Network

# Requirements

Python 3.7

Pytorch1.4 for Python 3.7 (preferrably with GPU support)

numpy

pandas

Pickle

## Process

```
In process.py you can specify the SESSIONS_Timedlt that is the limit deciding whether two interactions belong to the same session or two different ones.
Max_SESSION_LEN and Min_SESSION_LEN are the maximum session length and the minimum session length respectively.
Min_SESSIONS and WINDOWS_size are the maximum number of sessions and the minimum number of sessions in a sample respectively.

After running process.py on a dataset, the resulting train ,valid , test data
and global_timedlt data are stored in a pickle file, 5train_test_split.pickle.
global_timedlt data is session time interval data
```

