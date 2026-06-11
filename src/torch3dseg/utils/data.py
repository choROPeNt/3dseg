import h5py as h5



def readH5(file_path, **kwargs):
    """
    Generic function to read an H5 file with all groups and datasets
    and return a nested dictionary.
    Prints what was loaded: key, shape, dtype.
    """
    out = {}

    with h5.File(file_path, 'r') as f:
        for key in f.keys():

            item = f[key]

            # ------------------------------------------------------------------
            # Case: GROUP
            # ------------------------------------------------------------------
            if isinstance(item, h5.Group):
                print(f"[GROUP]  '{key}'")

                out[key] = {}
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    data = sub_item[...]

                    print(f"  ├─ [DATASET] '{key}/{sub_key}' "
                          f"shape={data.shape}, dtype={data.dtype}")

                    out[key][sub_key] = data

            # ------------------------------------------------------------------
            # Case: DATASET
            # ------------------------------------------------------------------
            else:
                data = item[...]

                print(f"[DATASET] '{key}' "
                      f"shape={data.shape}, dtype={data.dtype}")

                out[key] = data

    return out