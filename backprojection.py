"""
Back projection module for LabDataSet objects.
"""

import lab_data_set as lds
import matplotlib.pyplot as plt
import obspy
import numpy as np
from scipy import signal  # for filtering
import glob
import time


def apply(ds: lds.LabDataSet, BP_func, tag, trace_num, **kwargs):
    """Apply a BP_func (taking kwargs, returning a stack) to (tag,trcnum) in the LDS.
    Save resulting stack in aux data under 'BP/BP_func/tag/trcnum/timestamp', with input parameters in the aux data parameters dict.
    """
    stack = BP_func(ds, tag, trace_num, **kwargs)
    aux_path = "{}/{}/tr{}/{}".format(BP_func.__name__, tag, trace_num, timestamp())
    ds.add_auxiliary_data(data=stack, data_type="BP", path=aux_path, parameters=kwargs)


def makeframes(
    ds: lds.LabDataSet, aux_path, tag, trace_num, dt, win_len, view_to, pre=0
):
    stack = ds.auxiliary_data["BP/" + aux_path][tag][f"tr{trace_num}"].data
    base_rg = np.arange(0, view_to + dt + pre, dt)
    win_list = zip(base_rg, base_rg + win_len)

    frames = {}
    for window in win_list:
        win_key = (window[0] - pre, window[1] - pre)
        frames[win_key] = np.sum(stack[:, :, slice(*window)], axis=-1)
    return frames


def frames_by_quarters(
    ds: lds.LabDataSet, aux_path, tag, trace_num, base_pts: int, pre=0
):
    """Make and return BP frame images for a BP stack, setting frame parameters
    based on base_pts."""
    win_len = int(base_pts / 4)
    step_len = int(win_len / 4)
    view_to = int(base_pts * 1.5)
    return makeframes(ds, aux_path, tag, trace_num, step_len, win_len, view_to, pre)


def get_frames_range(frames):
    """Return global max and min from set of frames as (min,max)."""
    glob_min, glob_max = [100, -100]
    for frm in frames.values():
        glob_min = np.min((glob_min, frm.min()))
        glob_max = np.max((glob_max, frm.max()))
    return (glob_min, glob_max)


def show_frames(frames, minmax):
    """Make a 5-column plot of frames with a global colorscale.
    Plots with pyplot and returns figure."""
    nr = int(np.ceil(len(frames) / 5))
    fig = plt.figure(figsize=(15, 12))
    axs = [fig.add_subplot(nr, 5, n + 1, xticks=[], yticks=[]) for n in range(nr * 5)]

    for i, (win, frm) in enumerate(frames.items()):
        axs[i].imshow(frm, origin="lower")
        axs[i].set_title(str(win))
        img = axs[i].get_images()[0]
        img.set_clim(*minmax)
    return fig


######## BP_funcs ########
def BP_xcorr(ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    pre=0,
    vp=0.272,
    vel=0,
    filt=False,
    ):
    """Use cross correlation to define weighting and polarity."""
    # set up from BP_core
    x_pts, y_pts, grid_stack, orgxy, picked_stns, trc_dict = BP_core(
        ds, tag, trace_num, dxy, grid_rg_x, grid_rg_y
    )
    # loop through points
    for i, x in enumerate(x_pts):
        for j, y in enumerate(y_pts):
            # loop through stations
            for stn in picked_stns:
                loc = ds.stat_locs[stn][:2]
                dt = (
                    np.sqrt(np.sum((loc - (orgxy + [x, y])) ** 2) + 3.85 ** 2) / vp
                )  # us travel time
                stack_start = int(dt * 40)
                # stack from P wave
                if vel:
                    grid_stack[i, j, :] += np.diff(
                        trc_dict[stn][stack_start : stack_start + 2049]
                    )
                else:
                    grid_stack[i, j, :] += trc_dict[stn][
                        stack_start : stack_start + 2048
                    ]

            if filt:  # apply optional filter
                grid_stack[i, j, :] = signal.filtfilt(*filt, grid_stack[i, j, :])
    # return grid stack for processing
    return grid_stack


def BP_withpre(
    ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    pre=0,
    vp=0.272,
    vel=0,
    filt=False,
):
    """First BP function defined and improved for testing on ball drop and capillary calibration tests. Returns a BP_stack.
    :param dxy: Pixel width (mm)
    :param grid_rg_x: Origin-centered x-range (mm)
    :param grid_rg_y: Origin-centered y-range (mm)
    :param pre: Amount of signal to include before origin time (samples)
    :param vp: P-wave velocity (cm/s)
    :param vel: 0 to run with displacement traces, 1 to use np.diff for velocity traces (filtering recommended for vel=1)
    :param filt: Optional (b,a) coefficients for acausal filter
    """
    # set up from BP_core
    x_pts, y_pts, grid_stack, orgxy, picked_stns, trc_dict = BP_core(
        ds, tag, trace_num, dxy, grid_rg_x, grid_rg_y
    )
    # loop through points
    for i, x in enumerate(x_pts):
        for j, y in enumerate(y_pts):
            # loop through stations
            for stn in picked_stns:
                loc = ds.stat_locs[stn][:2]
                dt = (
                    np.sqrt(np.sum((loc - (orgxy + [x, y])) ** 2) + 3.85 ** 2) / vp
                )  # us travel time
                stack_start = int(dt * 40)
                # stack from P wave
                if vel:
                    grid_stack[i, j, :] += np.diff(
                        trc_dict[stn][stack_start : stack_start + 2049]
                    )
                else:
                    grid_stack[i, j, :] += trc_dict[stn][
                        stack_start : stack_start + 2048
                    ]

            if filt:  # apply optional filter
                grid_stack[i, j, :] = signal.filtfilt(*filt, grid_stack[i, j, :])
    # return grid stack for processing
    return grid_stack


def BP_core(
    ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    stack_len=2048,
    orgxy=False,
    trc_len=40000,
):
    """Simplify definition of new BP functions by distilling the core setup steps that are part of every BP.
    Creates a grid around an origin, sets up the matching empty_grid_stack, and sets up the dict of waveforms to stack from.
    :param stack_len: Total length of waveforms to stack
    :param orgxy: Optional alternate origin to center grid. Default False auto-selects the event origin.
    :param trc_len: Length of waveforms added to trc_dict
    """
    # get origin and time, remove this as well as travel time from each trace
    o_ind = ds.auxiliary_data.Origins[tag][f"tr{trace_num}"].parameters["o_ind"][0]
    if not orgxy:
        orgxy = ds.auxiliary_data.Origins[tag][f"tr{trace_num}"].data[:2]

    # make cm grid, ranges are inclusive
    x_pts = np.arange(grid_rg_x[0], grid_rg_x[1] + dxy, dxy) / 10
    y_pts = np.arange(grid_rg_y[0], grid_rg_y[1] + dxy, dxy) / 10
    empty_grid_stack = np.zeros((len(x_pts), len(y_pts), stack_len))

    # get picks
    picks = ds.auxiliary_data.LabPicks[tag][f"tr{trace_num}"].parameters
    # check for old or new stn format in picks keys
    if "L0" in list(picks.keys())[0]:
        picked_stns = [s for s in ds.stns if picks["L0." + s][0] > 0]
    else:
        picked_stns = [s for s in ds.stns if picks[s][0] > 0]
    # get waveforms, very slow operation so run once!
    trc_dict = {
        stn: ds.waveforms["L0_" + stn][tag][trace_num].data[o_ind : o_ind + trc_len]
        for stn in picked_stns
    }  # reduce size of waveforms read in by starting at o_ind
    return x_pts, y_pts, empty_grid_stack, orgxy, picked_stns, trc_dict


######## misc ########
def timestamp() -> str:
    return time.strftime("%Y%m%d_%X", time.localtime())
