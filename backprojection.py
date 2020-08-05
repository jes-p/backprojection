"""
Back projection module for LabDataSet objects.
"""

import matplotlib.pyplot as plt
import obspy
from obspy.signal import cross_correlation
import numpy as np
from scipy import signal  # for filtering
import glob
import time
from numba import jit

# Add paths to my other modules
import sys
import os

_home = os.path.expanduser("~") + "/"
for d in os.listdir(_home + "git_repos"):
    sys.path.append(_home + "git_repos/" + d)

import lab_data_set as lds


def apply(ds: lds.LabDataSet, BP_func, tag, trace_num, dont_save=[], **kwargs):
    """Apply a BP_func (taking kwargs, returning a stack) to (tag,trcnum) in the LDS.
    Save resulting stack in aux data under 'BP/BP_func/tag/trcnum/timestamp', with input parameters in the aux data parameters dict.
    Report elapsed time.
    Added option to return additional parameters/info from each BP_func as a dict, saved with kwarg params.
    Added dont_save list of keys to drop from the saved kwarg params.
    """
    start_timer = time.perf_counter()
    stack, addtl_params = BP_func(ds, tag, trace_num, **kwargs)
    aux_path = "{}/{}/tr{}/{}".format(BP_func.__name__, tag, trace_num, timestamp())
    for k in dont_save:
        del kwargs[k]
    ds.add_auxiliary_data(
        data=stack, data_type="BP", path=aux_path, parameters={**kwargs, **addtl_params}
    )
    end_timer = time.perf_counter()
    print(f"BP grid computed in {end_timer-start_timer} seconds")
    return aux_path


def makeframes(ds: lds.LabDataSet, aux_path, tstamp, dt, win_len, view_to, pre=0):
    stack = ds.auxiliary_data["BP/" + aux_path][tstamp].data
    base_rg = np.arange(0, view_to + dt + pre, dt)
    win_list = zip(base_rg, base_rg + win_len)

    frames = {}
    for window in win_list:
        win_key = (window[0] - pre, window[1] - pre)
        frames[win_key] = np.sum(stack[:, :, slice(*window)], axis=-1)
    return frames


def frames_by_quarters(ds: lds.LabDataSet, aux_path, tstamp, base_pts: int, pre=0):
    """Make and return BP frame images for a BP stack, setting frame parameters
    based on base_pts."""
    win_len = int(base_pts / 4)
    step_len = int(win_len / 4)
    view_to = int(base_pts * 1.5)
    return makeframes(ds, aux_path, tstamp, step_len, win_len, view_to, pre)


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
def BP_xcorr(
    ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    pre=0,
    vp=0.272,
    no_shift=False,
    no_weights=False,
    **kwargs,
):
    """Use cross correlation to define weighting and polarity."""
    # set up from BP_core
    x_pts, y_pts, grid_stack, orgxy, picked_stns, trc_dict = BP_core(
        ds, tag, trace_num, dxy, grid_rg_x, grid_rg_y, **kwargs
    )

    # find the best xcorr window for this data
    post_lens = np.arange(150, 400, 50)
    wgts, sfts, best_post = run_xcorr(ds, tag, trace_num, post_lens)

    # exploratory option to ignore weights or shifts
    if no_shift:
        sfts = {stn: 0 for stn in picked_stns}
    if no_weights:
        wgts = {stn: 1 for stn in picked_stns}

    w_norm = sum(wgts.values())

    stn_locs, trc_list = [[], []]
    for stn in picked_stns:
        stn_locs.append(ds.stat_locs[stn])
        trc_list.append(trc_dict[stn])

    # loop through points
    for i, x in enumerate(x_pts):
        for j, y in enumerate(y_pts):
            # loop through stations
            for s, stn in enumerate(picked_stns):
                loc = stn_locs[s][:2]
                dt = (
                    np.sqrt(np.sum((loc - (orgxy + [x, y])) ** 2) + 3.85 ** 2) / vp
                )  # us travel time
                stack_start = (
                    int(dt * 40) - sfts[stn]
                )  # added shift correction from xcorr
                # stack from P wave
                grid_stack[i, j, :] += (
                    wgts[stn] * trc_dict[stn][stack_start : stack_start + 2048]
                )

    # return grid stack for processing
    return grid_stack * (1 / w_norm), {"best_post": best_post}


# functions for running xcorr and checking spread
def run_xcorr(ds, tag, trace_num, post_lens, ncomp=0, plot_best_post=True):
    """Runs cross correlation on short traces focused around the first pick for a trc_num and returns the weights and shifts for the best window.
    :param ds: LDS object in use
    :param tag: tag of the ds
    :param trace_num: trace number within the tag
    :param post_lens: list of window lengths beyond the pick to test for the best xcorr result
    :param ncomp: arrival-sorted stn to use as reference trace
    :param plot_best_post: bool to plot and save the shifted traces for the auto-selected window
    """
    # cross-correlate to first arrival, get weight and shift for each trace
    # get very short, pick-windowed traces for this part
    trs_pre, xcorr_start = [
        200,
        100,
    ]  # the extra 100 leave room for shifting more easily
    trs = ds.get_traces(
        tag, trace_num, pre=trs_pre, tot_len=600
    )  # TODO: add params to the trc_dict so pre is documented?
    pks = ds.get_picks(tag, trace_num)
    good_pks = {s: pks[s] for s in pks if pks[s] > 0}
    seq = sorted(good_pks.keys(), key=lambda s: good_pks[s][0])
    astn = seq.pop(ncomp)  # get stn to autocorrelate
    weights, shifts, spreads, shifted_trs = [{}, {}, {}, {}]
    for post in post_lens:
        cut = slice(xcorr_start, trs_pre + post)
        azd = trs[astn][cut] - trs[astn][xcorr_start]  # zero the trace
        acorr = signal.correlate(azd, azd, mode="same")
        sh, aval = cross_correlation.xcorr_max(acorr)
        weights[post] = {astn: 1}
        shifts[post] = {astn: 0}
        shifted_trs[post] = {astn: trs[astn][cut]}
        for stn in seq:
            szd = trs[stn][cut] - trs[stn][xcorr_start]
            xcorr = signal.correlate(azd, szd, mode="same")
            sh, cval = cross_correlation.xcorr_max(xcorr)
            shifts[post][stn] = int(sh)
            weights[post][stn] = aval / cval
            # calculate spread for this post
            sh_cut = slice(max(cut.start - int(sh), 0), cut.stop - int(sh))
            adj = weights[post][stn] * trs[stn][sh_cut]
            shifted_trs[post][stn] = adj - adj[0]
        # check spread of adjusted traces if more than one post_len was provided
        if len(post_lens) > 1:
            # get polarity of shifted_trs[post] (they all match astn)
            is_up_first = abs(max(shifted_trs[post][astn])) > abs(
                min(shifted_trs[post][astn])
            )  # better to sample ~20pts past pick?
            # set spread check point at half the max amplitude # TODO: probably multiple points is better
            if is_up_first:
                check = np.max(shifted_trs[post][astn]) / 2
                hits = [np.argwhere(shifted_trs[post][astn] > check)[0][0]]
            else:
                check = np.min(shifted_trs[post][astn]) / 2
                hits = [np.argwhere(shifted_trs[post][astn] < check)[0][0]]
                # TODO: now works for any polarity but seems misplaced. Probably just forcing positive polarity would be better
            # now add check points for remaining stations
            for stn in seq:
                hits.append(np.argwhere(shifted_trs[post][stn] < check)[0][0])
            hits.sort()
            spreads[post] = hits[-1] - hits[0]

    if len(post_lens) > 1:
        # get best post_len based on shifted trace spreads
        best_post = min(spreads, key=spreads.get)
    else:
        best_post = post_lens[0]

    # plot shifted traces from best post for visual confirmation of no funny business
    if plot_best_post:
        plt.figure()
        plt.plot(shifted_trs[best_post][astn])
        for stn in seq:
            plt.plot(shifted_trs[best_post][stn])
        plt.show()

    return weights[best_post], shifts[best_post], best_post


def BP_coherency(
    ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    pre=0,
    vp=0.272,
    xcorr_stack=[],
    best_post=None,
    Tpts=(0, 200),
    **kwargs,
):
    """Use Ishi 2011 coherency to bring out small source features after BP_xcorr."""
    # set up from BP_core still needed for original traces
    x_pts, y_pts, grid_stack, orgxy, picked_stns, trc_dict = BP_core(
        ds, tag, trace_num, dxy, grid_rg_x, grid_rg_y, **kwargs
    )

    # get stack from BP_xcorr as input or by running
    if len(np.shape(xcorr_stack)) != 3:
        xcorr_stack, xcorr_info = BP_xcorr(
            ds, tag, trace_num, dxy, grid_rg_x, grid_rg_y, pre, vp
        )
        best_post = xcorr_info["best_post"]
    stn_locs, trc_list = [[], []]
    for stn in picked_stns:
        stn_locs.append(ds.stat_locs[stn])
        trc_list.append(trc_dict[stn])

    # get shifts to align traces better from run_xcorr
    _, sfts, _ = run_xcorr(ds, tag, trace_num, [best_post], plot_best_post=False)

    # loop through points
    for i, x in enumerate(x_pts):
        for j, y in enumerate(y_pts):
            # loop through stations
            for s in range(len(picked_stns)):
                loc = stn_locs[s][:2]
                dt = (
                    np.sqrt(np.sum((loc - (orgxy + [x, y])) ** 2) + 3.85 ** 2) / vp
                )  # us travel time
                stack_start = int(dt * 40) - sfts[stn]  # shift correction from xcorr
                # loop over tau to build coherency stack for this station
                base_trs = trc_list[s][stack_start : stack_start + 2048]
                num_sum = np.zeros_like(base_trs)
                for tau in range(*Tpts):
                    # numerator sum
                    num_sum += base_trs * xcorr_stack[i, j, tau]
                # left sqrt sum
                left_sqrt = np.sqrt(np.square(base_trs) * (Tpts[1] - Tpts[0]))
                # right sqrt sum
                right_sqrt = np.sqrt(np.sum(np.square(xcorr_stack[i, j, slice(*Tpts)])))
                # grid_stack += num/sqrt*sqrt to complete this station
                grid_stack[i, j, :] += num_sum / (left_sqrt * right_sqrt)

    # return grid stack for processing
    return grid_stack * (1 / len(picked_stns)), {}


def BP_withpre(
    ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    pre=0,
    vp=0.272,
    **kwargs,
):
    """First BP function defined and improved for testing on ball drop and capillary calibration tests. Returns a BP_stack.
    :param dxy: Pixel width (mm)
    :param grid_rg_x: Origin-centered x-range (mm)
    :param grid_rg_y: Origin-centered y-range (mm)
    :param pre: Amount of signal to include before origin time (samples)
    :param vp: P-wave velocity (cm/s)
    """
    # set up from BP_core
    x_pts, y_pts, grid_stack, orgxy, picked_stns, trc_dict = BP_core(
        ds, tag, trace_num, dxy, grid_rg_x, grid_rg_y, **kwargs
    )
    stn_locs, trc_list = [[], []]
    for stn in picked_stns:
        stn_locs.append(ds.stat_locs[stn])
        trc_list.append(trc_dict[stn])
    # grid_stack = BP_looper(x_pts,y_pts,picked_stns,stn_locs,orgxy,grid_stack,trc_list)
    # return grid_stack

    # @jit(nopython=True,parallel=True)
    # def BP_looper(x_pts,y_pts,picked_stns,stn_locs,orgxy,grid_stack,trc_list,vp=.272):
    # loop through points
    for i, x in enumerate(x_pts):
        for j, y in enumerate(y_pts):
            # loop through stations
            for s in range(len(picked_stns)):
                # loc = ds.stat_locs[stn][:2] # SO SLOW
                loc = stn_locs[s][:2]
                dt = (
                    np.sqrt(np.sum((loc - (orgxy + [x, y])) ** 2) + 3.85 ** 2) / vp
                )  # us travel time
                stack_start = int(dt * 40)
                # stack from P wave
                grid_stack[i, j] += np.array(
                    trc_list[s][stack_start : stack_start + 2048]
                )

    # return grid stack for processing
    return grid_stack, {}


def BP_core(
    ds: lds.LabDataSet,
    tag,
    trace_num,
    dxy,
    grid_rg_x,
    grid_rg_y,
    stack_len=2048,
    vel=False,
    orgxy=False,
    trc_len=40000,
    filt=False,
):
    """Simplify definition of new BP functions by distilling the core setup steps that are part of every BP.
    Creates a grid around an origin, sets up the matching empty_grid_stack, and sets up the dict of waveforms to stack from.
    :param stack_len: Total length of waveforms to stack
    :param vel: 0 to run with displacement traces, 1 to use np.diff for velocity traces (filtering recommended for vel=1)
    :param orgxy: Optional alternate origin to center grid. Default False auto-selects the event origin.
    :param trc_len: Length of waveforms added to trc_dict
    :param filt: Optional (b,a) coefficients for acausal filter
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
    if vel:
        for stn in trc_dict.keys():
            trc_dict[stn] = np.diff(trc_dict[stn])
    if filt:
        for stn in trc_dict.keys():
            trc_dict[stn] = signal.filtfilt(*filt, trc_dict[stn])
    return x_pts, y_pts, empty_grid_stack, orgxy, picked_stns, trc_dict


######## misc ########
def timestamp() -> str:
    return time.strftime("%Y%m%d_%X", time.localtime())
