import pandas as pd


# Standard dimension order: time-like dims first, then spatial
_STANDARD_DIM_ORDER = ["init_time", "time", "member", "step", "lat", "lon"]

def dim_order_fmt(ds):
    """
    Transpose all variables in ds so their dimensions follow the standard order:
        (init_time, time, member, step, lat, lon)

    Only the dimensions actually present in each variable are used — missing ones
    are skipped.  For example, a variable with dims (lat, lon, init_time, step)
    becomes (init_time, step, lat, lon).

    Any dimension not listed in _STANDARD_DIM_ORDER is appended at the end in
    their original relative order, so unknown dims are never dropped.
    """
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        current_dims = list(da.dims)

        # Build target order: known dims in standard order, then any unknown dims
        ordered = [d for d in _STANDARD_DIM_ORDER if d in current_dims]
        remainder = [d for d in current_dims if d not in _STANDARD_DIM_ORDER]
        target_dims = ordered + remainder

        if target_dims != current_dims:
            da = da.transpose(*target_dims)

        new_vars[var] = da

    return ds.assign(new_vars)


def dim_fmt(ds):
    """Standardize dimension names"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        #print("lon NOT in coords  --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()][0]

        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

#    if "time" not in coord_list:
#        print("time NOT in coords --> ")  # , model_name)
#        time_coords = [variable for variable in coord_list if "TIME" in variable][0]
#        ds = ds.rename({time_coords: "time"})

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        keywords = ["time", 'date']
        time_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({time_coords: "time"})

    #return ds
    return dim_order_fmt(ds)


def dim_fmt_model(ds):
    """Standardize dimension names for deterministic reforecast model data"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        #print("lon NOT in coords  --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()][0]

        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

    if "init_time" not in coord_list:
        #print("init_time NOT in coords --> ")  # , model_name)
        time_coords = [variable for variable in coord_list if "time" in variable.lower()][0]
        ds = ds.rename({time_coords: "init_time"})

    if "step" not in coord_list:
        keywords = ["day", "prediction_timedelta"]
        step_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ][0]
        ds = ds.rename({step_coords: "step"})

    # convert TimedeltaIndex to integer (days)
    if isinstance(ds.indexes["step"], pd.TimedeltaIndex):
        ds = ds.assign_coords(step=ds.step.dt.days)

    #return ds
    return dim_order_fmt(ds)


def dim_fmt_model_ensemble(ds):
    """Standardize dimension names for probabilistic reforecast model data"""

    ds = dim_fmt_model(ds)

    # Look at coordinates AND dimensions: a member dimension may exist without a
    # coordinate variable (e.g. after the coords were dropped or a squeeze).
    names = list(dict.fromkeys(list(ds.coords.keys()) + list(ds.dims)))

    if "member" not in names:
        keywords = ["number", "sample", "member", "ensemble", "realization", "realisation"]
        candidates = [
            name for name in names
            if any(keyword in name.lower() for keyword in keywords)
        ]
        if not candidates:
            raise ValueError(
                "dim_fmt_model_ensemble: no ensemble-member dimension found. "
                f"coords={list(ds.coords)}, dims={dict(ds.sizes)}, "
                f"variables={list(ds.data_vars)}. A probabilistic dataset needs a "
                "member dimension named like 'number', 'sample', 'member' or 'ensemble'."
            )
        ds = ds.rename({candidates[0]: "member"})

    return dim_order_fmt(ds)