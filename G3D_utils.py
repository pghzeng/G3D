import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys
import random
import tabix
import time
from functools import wraps
import threading
# import multiprocessing
# from torch.multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from queue import Queue
import pkg_resources
import pyfaidx

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from selene_sdk.sequences import Genome

import cooler
import hicmatrix
from hicmatrix.HiCMatrix import hiCMatrix
from hicexplorer.hicFindTADs import HicFindTads


def to_np(arr):
    if isinstance(arr, torch.Tensor):
        return arr.cpu().detach().numpy()
    return arr

def call_TAD_by_matrix(
    matrix,
    chrom_name,
    start,
    end,
    num_processors=8,
    max_depth=200000,
    min_depth=80000,
    step=40000,
    delta=0.01,
    min_boundary_distance=80000,
    use_zscore=True,
    p_correct_for_multiple_testing="fdr",
    p_threshold_comparisons=0.01,
    save_result = False,
    save_prefix = "TAD_result"
    ):

    matrix = np.squeeze(to_np(matrix))
    sparse_matrix = csr_matrix(matrix)
    assert matrix.shape[0] == matrix.shape[1]
    bin_size = (end - start) / matrix.shape[0]
    bin_starts = np.arange(start, end, bin_size)
    intervals = [(chrom_name, s, s + bin_size, None) for s in bin_starts] # None is dummy for extra needed by hiCMatrix
    hic_matrix = hiCMatrix(pMatrixFile=None)
    hic_matrix.setMatrix(matrix=sparse_matrix, cut_intervals=intervals)

    ft = HicFindTads(
    matrix=hic_matrix,
    num_processors=num_processors,
    max_depth=max_depth,
    min_depth=min_depth,
    step=step,
    delta=delta,
    min_boundary_distance=min_boundary_distance,
    use_zscore=use_zscore,
    p_correct_for_multiple_testing=p_correct_for_multiple_testing,
    p_threshold_comparisons=p_threshold_comparisons,
    )

    ft.compute_spectra_matrix()
    insulation_scores = ft.bedgraph_matrix["matrix"]
    insulation_scores = insulation_scores.mean(axis=1)

    if save_result:
        ft.find_boundaries()
        # boundary_dict = ft.boundaries
        # min_idx_list = boundary_dict['min_idx']
        # delta_dict   = boundary_dict['delta']
        # pval_dict    = boundary_dict['pvalues']
        # boundaries = []
        # for b in min_idx_list:
        #     delta = delta_dict[b]
        #     pval = pval_dict[b]
        #     chrom = ft.bedgraph_matrix['chrom'][b]
        #     start = ft.bedgraph_matrix['chr_start'][b]
        #     end   = ft.bedgraph_matrix['chr_end'][b]
        #     boundaries.append((chrom, start, end, pval))
        ft.save_domains_and_boundaries(prefix=save_prefix)

    return insulation_scores


def figshow(
    matrices,
    lines=None,
    colorbar_mode="single",
    cmap="viridis",
    vmin=0,
    vmax=None,
    show_ticks=False,
    show=False,
    save_file=None,
    fig_size=(5, 3),
    dpi=100,
    wspace = 0.1,
    hspace = 0,
    mat_titles = None,
    line_titles = None,
    cbar_title = None,
    height_ratios = [10, 1]
):
    if not matrices:
        print("No matrices to plot.")
        return
    num_mats = len(matrices)

    if not lines:
        lines = []
    if len(lines) not in (0, num_mats):
        raise ValueError("lines should have same length as matrices")
    have_linecharts = (len(lines) == num_mats)

    nrows = 2 if have_linecharts else 1
    ncols = num_mats

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if have_linecharts:
        height_ratios = height_ratios
    else:
        height_ratios = None

    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        width_ratios=[1]*ncols,
        wspace=wspace,
        hspace=hspace
    )

    ax_mats = []
    image_handles = []

    mat_row = 0

    for i in range(num_mats):
        ax_mat = fig.add_subplot(gs[mat_row, i])
        ax_mats.append(ax_mat)

        arr = np.squeeze(to_np(matrices[i]))
        im = ax_mat.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        image_handles.append(im)
        if mat_titles:
            ax_mat.set_title(mat_titles[i])
        else:
            ax_mat.set_title("", pad=0)

        if not show_ticks:
            ax_mat.set_xticks([])
            ax_mat.set_xlabel("")
            ax_mat.set_yticks([])
            ax_mat.set_ylabel("")

    ax_lines = []

    if have_linecharts:
        line_row = 1
        line_max, line_min = np.max(np.concatenate(lines)), np.min(np.concatenate(lines))
        for i in range(num_mats):
            ax_line = fig.add_subplot(gs[line_row, i])
            ax_lines.append(ax_line)
            linedata = to_np(lines[i])
            ax_line.plot(linedata, color='red')
            ax_line.set_xlim([0, len(linedata)])
            ax_line.set_ylim(line_min, line_max)
            if line_titles:
                ax_line.set_title(line_titles[i])
            else:
                ax_line.set_title("", pad=0)
            if not show_ticks:
                ax_line.set_xticks([])
                ax_line.set_xlabel("")
                ax_line.set_yticks([])
                ax_line.set_ylabel("")

    if colorbar_mode != "none":
        fig.canvas.draw()
        def add_cbar_for(im):
            ax_ = im.axes
            image_extent = im.get_window_extent(ax_)
            fig_extent = image_extent.transformed(fig.transFigure.inverted())
            cbar_width = 0.02
            x0 = fig_extent.x1 + 0.02
            y0 = fig_extent.y0
            h  = fig_extent.height
            cbar_ax = fig.add_axes([x0, y0, cbar_width, h])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
            if cbar_title:
                cbar.ax.set_title(cbar_title)
            return cbar_ax, cbar

        if colorbar_mode == "single":
            im_last = image_handles[-1]
            add_cbar_for(im_last)

        elif colorbar_mode == "each":
            for im in image_handles:
                add_cbar_for(im)

    if save_file:
        plt.savefig(save_file, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def array32(*args, **kwargs):
    return np.array(*args, dtype=np.float32, **kwargs)

def sample_regions(chrom_dict, num_samples, window_size, bin_size):
    total_length = sum(chrom_dict.values())
    weights = {chrom: length / total_length for chrom, length in chrom_dict.items()}
    chrom_list = list(chrom_dict.keys())
    weight_list = list(weights.values())
    sampled_regions = []
    for _ in range(num_samples):
        chrom = random.choices(chrom_list, weights=weight_list, k=1)[0]
        chrom_length = chrom_dict[chrom]
        start = random.randint(0, chrom_length - window_size)
        start -= start % bin_size
        end = start + window_size
        sampled_regions.append((chrom, start, end))
    return sampled_regions

def extract_gene_exp(df, sample_name, gene_name_list=None):
    if sample_name not in df.columns:
        raise ValueError(f"sample {sample_name} not found")
    if gene_name_list:
        filtered_df = df[df['Name'].isin(gene_name_list)].set_index('Name').reindex(gene_name_list)
        exp_list = filtered_df[sample_name].tolist()
    else:
        exp_list = df[sample_name].tolist()
    return exp_list

def load_exps(args):
    sample_name, gene_expression_df = args
    exp = torch.tensor(extract_gene_exp(gene_expression_df, sample_name))
    return exp

def load_seqs(args):
    # print("loading seqs...")
    reference_genome, region = args
    # print(reference_genome.initialized)
    seq = torch.tensor(reference_genome.get_encoding_from_coords(*region).transpose(-1, -2))
    # print("seq shape", seq.shape)
    return seq

# def load_targets(args):
#     sample_name, cool_dir, sampled_regions = args
#     cool_path = f"{cool_dir}/{sample_name}.sumnorm.mcool::/"
#     # print(cool_path)
#     targets = []
#     cooler_file = cooler.Cooler(cool_path)
#     # print(cooler_file)
#     for region in sampled_regions:
#         matrix = cooler_file.matrix(balance=False).fetch(region)
#         targets.append(torch.tensor(matrix))
#     targets = torch.stack(targets)
#     return targets

def load_targets(args):
    # print("loading targets...")
    sample_name, cool_dir, bin_size, sampled_regions = args
    cool_path = f"{cool_dir}/{sample_name}.sumnorm.mcool::/resolutions/{bin_size}"
    # print(cool_path)
    targets = []
    cooler_file = cooler.Cooler(cool_path)
    # print(cooler_file)
    for region in sampled_regions:
        # print("fetching matrix...")
        matrix = cooler_file.matrix(balance=False).fetch(region)
        targets.append(torch.tensor(matrix))
        # print("targets appended...")
    targets = torch.stack(targets)
    # print("targets shape", targets.shape)
    return targets

class G3DMultiCoolDataset(Dataset):
    def __init__(self,
        sample_names,
        cool_dir,
        gene_expression_df,
        genome_size_dict,
        reference_genome,
        window_size=2000000,
        bin_size=4000,
        holdout_chroms=["chr8"],
        regions_per_sample=1,
        gene_name_list=None,
        processes=os.cpu_count(),
        blacklist_regions="hg38"):

        self.sample_names = sample_names
        self.cool_dir = cool_dir
        self.gene_expression_df = gene_expression_df
        self.genome_size_dict = genome_size_dict
        self.reference_genome = reference_genome
        self.window_size = window_size
        self.bin_size = bin_size
        self.regions_per_sample = regions_per_sample
        self.gene_name_list = gene_name_list
        self.processes = processes

        self.total_samples = regions_per_sample * len(sample_names)
        self.train_chroms_dict = {k:v for k,v in genome_size_dict.items() if k not in holdout_chroms}
        self.sampled_regions = sample_regions(self.train_chroms_dict, regions_per_sample, window_size, bin_size)

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.load_data)
        self.thread.start()
        self.thread.join()

    def load_data(self):
        with Pool(self.processes) as pool:
            print("start multiprocess...")
            exps = pool.map(load_exps, [(sample_name, self.gene_expression_df) for sample_name in self.sample_names])
            # print(len(exps))
            # print(reference_genome)
            # print(reference_genome.initialized)
            seqs = pool.map(load_seqs, [(self.reference_genome, region) for region in self.sampled_regions])
            # print(len(seqs))
            targets = pool.map(load_targets, [(sample_name, self.cool_dir, self.bin_size, self.sampled_regions) for sample_name in self.sample_names])
            # print(len(targets))

        exps = torch.stack(exps)
        self.exps = exps
        # print("exps shape", exps.shape)
        seqs = torch.stack(seqs)
        self.seqs = seqs
        # print("seqs shape", seqs.shape)
        targets = torch.cat(targets)
        # print("targets shape", targets.shape)

        exp_indices = torch.arange(len(self.sample_names)).repeat_interleave(self.regions_per_sample)
        seq_indices = torch.arange(self.regions_per_sample).repeat(len(self.sample_names))
        perm = torch.randperm(self.total_samples)
        # print(perm)
        self.exp_indices = exp_indices[perm]
        self.seq_indices = seq_indices[perm]
        self.targets = targets[perm]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return idx, self.exps[self.exp_indices[idx]], self.seqs[self.seq_indices[idx]], self.targets[idx]


# Something went wrong with this manager. It slows down training speed, do not use it until debugged
class G3DDatasetManager:
    def __init__(self,
        sample_names,
        cool_dir,
        gene_expression_df,
        genome_size_dict,
        reference_genome,
        window_size=2000000,
        bin_size=4000,
        holdout_chroms=["chr8"],
        regions_per_sample=1,
        gene_name_list=None,
        queue_size=2,
        processes=os.cpu_count()):

        self.sample_names = sample_names
        self.cool_dir = cool_dir
        self.gene_expression_df = gene_expression_df
        self.genome_size_dict = genome_size_dict
        self.reference_genome = reference_genome
        self.window_size=window_size
        self.bin_size=bin_size
        self.holdout_chroms=holdout_chroms
        self.regions_per_sample=regions_per_sample
        self.gene_name_list=gene_name_list
        self.queue = Queue(maxsize=queue_size)
        self.processes = processes
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.dataset_count=0
        self.thread = threading.Thread(target=self.dataset_thread)
        self.thread.start()

    def dataset_thread(self):
        while not self.stop_event.is_set():
            if self.queue.full():
                continue  # Skip if the queue is full, otherwise block on get.
            print(f"Preparing Dataset {self.dataset_count}...")
            start_time = time.time()
            dataset = G3DMultiCoolDataset(
                sample_names=self.sample_names,
                cool_dir=self.cool_dir,
                gene_expression_df=self.gene_expression_df,
                genome_size_dict=self.genome_size_dict,
                reference_genome=self.reference_genome,
                window_size=self.window_size,
                bin_size=self.bin_size,
                holdout_chroms=self.holdout_chroms,
                regions_per_sample=self.regions_per_sample,
                processes = self.processes
            )
            end_time = time.time()
            self.queue.put(dataset)
            with self.lock:
                print(f"Dataset {self.dataset_count} put, used time: {end_time - start_time}")
                self.dataset_count += 1
            time.sleep(5)  # Allow a small delay to avoid constant checking


    def get_next_dataset(self):
        return self.queue.get()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
