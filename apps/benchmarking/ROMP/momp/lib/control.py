import os
#import inspect
from dataclasses import fields
from typing import Union, Tuple, List, Optional
from pathlib import Path

from momp.lib.convention import Case, Setting
from momp.utils.printing import combi_to_str
#from momp.io.input import set_dir
from momp.utils.practical import set_dir
from dataclasses import asdict, is_dataclass
import copy


def init_dataclass(dc, dic):
    keys = dc.__annotations__.keys()
    subset = {key: dic[key] for key in keys if key in dic}
    return dc(**subset)


def modify_list_keys(dictionary):
    modified_keys = []
    for key, value in dictionary.items():
        if key.endswith("_list"):
            modified_key = key[:-5]  # Remove the last 5 characters "_list"
            modified_keys.append(modified_key)
    return modified_keys


def case_across_list(item, list1, list2):
    """find the corresponding item in list2 for a given item in list1"""
    if item in list1:
        index = list1.index(item)
        if index < len(list2):
            return list2[index]
    return None


def iter_list(dic, ext="_list"):
    layout_pool = []
    for field in dic["layout"]:
        lst = dic.get(field + ext)  # .copy()
        layout_pool.append(lst)
    return layout_pool


def years_tuple_clim(year_start: int, year_end: int) -> tuple[int, ...]:
    """
    Create a tuple of integers from year_start to year_end inclusive.
    """
    return tuple(range(year_start, year_end + 1))


def years_tuple_model(start_date: tuple[int,int,int], end_date: tuple[int,int,int]) -> tuple[int, ...]:
    """
    Create a tuple of years from start_date to end_date inclusive.
    Each date is a tuple like (year, month, day)
    """
    start_year = start_date[0]
    end_year = end_date[0]
    return tuple(range(start_year, end_year + 1))


#def take_ensemble_members(
#    members: Union[List[int], str]
#) -> List[int]:
#    """
#    Normalize members into a list of ints.
#
#    Accepts:
#    - list of ints → returned as-is
#    - string 'start-end' → expanded to list
#    """
#    # Case 1: already a list of ints
#    if isinstance(members, list):
#        return list(members)  # return a copy
#
#    # Case 2: string range "1-5"
#    if isinstance(members, str) and "-" in members:
#        start, end = map(int, members.split("-"))
#        return list(range(start, end + 1))
#
#    raise TypeError("members must be list[int] or 'start-end' string")


def take_ensemble_members(
    members: Optional[Union[tuple[int, ...], list[int], str]]
) -> list[int]:
    """
    Normalize members into a list of ints.

    Accepts:
    - list[int] or tuple[int, ...] → returned as a list
    - string 'start-end' → expanded to list
    - None → empty list
    """
    # Case 0: None
    if not members or members == 'All':
        return None

    # Case 1: list or tuple of ints
    if isinstance(members, (list, tuple)):
        return tuple(members)  # normalize to list

    # Case 2: string range "1-5"
    if isinstance(members, str) and "-" in members:
        start, end = map(int, members.split("-", 1))
        return tuple(range(start, end + 1))

    raise TypeError(
        "members must be list[int], tuple[int, ...], 'start-end' string, or None"
    )


#def restore_args(func, kwargs, bound_args):
#    """
#    Restore keyword-only parameters of `func` back into kwargs.
#    """
#    sig = inspect.signature(func)
#    new_kwargs = dict(kwargs)
#
#    for name, param in sig.parameters.items():
#        if (
#            param.kind is param.KEYWORD_ONLY
#            and name in bound_args
#            and name not in new_kwargs
#        ):
#            new_kwargs[name] = bound_args[name]
#
#    return new_kwargs


def make_case(dataclass, combi, dic):

    layout = dic["layout"]

    layout_dict = {key: None for key in layout}
    layout_dict.update(zip(layout_dict.keys(), combi))

    case_keys = [
        field.name for field in fields(dataclass)
    ]  # get defined keys in dataclass

    dic_list_keys = modify_list_keys(dic)  # return keys endwith _list, removing _list
    dic_list_keys = [
        item for item in dic_list_keys if item not in list(layout_dict.keys())
    ]
    list_keys = list(set(case_keys).intersection(set(dic_list_keys)))
    list_keys.remove("tolerance_days") # tolerance_days is paired with verification_window


    case = init_dataclass(dataclass, dic)
    case.update(layout_dict)

    value_list = []

    for key in list_keys:
        value = case_across_list(case.model, dic["model_list"], dic[key + "_list"])

        if key == "model_dir":
            if not Path(value).is_absolute():
                value = set_dir(value)

        value_list.append(value)

        #if key == "model_dir":
        #    case.fn_var = value

    #case_name = "{}".format("_".join(combi_to_str(combi)))
    case_name = combi_to_str(combi)
    case.case_name = case_name.replace(" ", "_")

    #if not dic['years']:
    if dic.get('years') == 'All' or not dic.get('years'):
        case.years = years_tuple_model(dic['start_date'], dic['end_date'])

    #if not dic['years_clim']:
    if dic.get('years_clim') == 'All' or not dic.get('years_clim'):
        case.years_clim = years_tuple_clim(dic['start_year_clim'], dic['end_year_clim'])

    case.members = take_ensemble_members(dic['members'])

    value_dic = dict(zip(list_keys, value_list))
    case.update(value_dic)

    case.tolerance_days = case_across_list(case.verification_window, 
                                           dic["verification_window_list"], dic["tolerance_days_list"])

    return case



def ref_cfg_layout(cfg, ref_model=None, verification_window=None):

    cfg_ref = copy.copy(cfg)

    if ref_model:
        cfg_ref.ref_model = ref_model

    if cfg.ref_model == 'climatology':
        cfg_ref.probabilistic = False

    cfg_ref.model_list = (cfg.ref_model,)

    if verification_window:
        cfg_ref.verification_window_list = (verification_window,)

    layout_pool = iter_list(vars(cfg_ref))

    return cfg_ref, layout_pool


def ref_model_case(case_orig, setting):

    case = copy.copy(case_orig)

    if isinstance(setting, Setting):
        case_ref = {'model_dir': setting.ref_model_dir,
                    'model_var': case.ref_model_var,
                    'file_pattern': setting.ref_model_file_pattern,
                    'unit_cvt': setting.ref_model_unit_cvt
                    }
    elif isinstance(setting, dict):
        case_cfg = setting
        case_ref = {'model_dir': case_cfg['ref_model_dir'],
                    'model_var': case_cfg['ref_model_var'],
                    'file_pattern': case_cfg['ref_model_file_pattern'],
                    'unit_cvt': case_cfg['ref_model_unit_cvt']
                    }

    case.update(case_ref)

    #if case.model == 'climatology':
    #    case.years = case.years_clim

    case_cfg_ref = {**asdict(case), **asdict(setting)}


    return case, case_cfg_ref



def filter_bins_in_window(day_bins, verification_window):
    """
    Returns only the day bins that are completely or partially within the verification window.

    Args:
        day_bins: tuple of tuples, e.g. ((1,5), (6,10), ...)
        verification_window: tuple of (start, end), e.g. (6, 20)

    Returns:
        tuple of tuples containing only the bins inside the window
    """
    start_win, end_win = verification_window

    #filtered = []
    #for bin_start, bin_end in day_bins:
    #    # Check if the bin overlaps with or is inside the window
    #    # Condition: bin is not completely before window AND not completely after window
    #    if bin_end >= start_win and bin_start <= end_win:
    #        filtered.append((bin_start, bin_end))
    #return tuple(filtered)

    return tuple(
        bin_range
        for bin_range in day_bins
        if start_win <= bin_range[0] and bin_range[1] <= end_win
    )
