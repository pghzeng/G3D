import os
limit_devices = True
devices_number = 1
if limit_devices:
    devices = [2]
    devices_number = len(devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(list(map(str, devices)))

import sys
G3D_dir = "/public/home/hpyu/ZPGH/G3D"
sys.path.append(G3D_dir)
import gc
import psutil
import pathlib
import glob
from collections import ChainMap
from collections import defaultdict
import itertools
from itertools import chain
import random
import time
import multiprocessing
import subprocess
import warnings
warnings.filterwarnings('ignore')

import torch
print(torch.cuda.is_available())
# print(torch.cuda.device_count())
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
fm.fontManager.addfont('/public/home/hpyu/ZPGH/arial.ttf')
matplotlib.rcParams['font.family'] = 'Arial'

import cooler
import selene_sdk
from G3D_selene_utils import SamplerDataLoader, MemmapGenome, Genomic2DFeatures, RandomPositionsSamplerHiC
# from G3D_utils import call_TAD_by_matrix, figshow # G3DMultiCoolDataset, G3DDatasetManager
from G3D_modules_round2_v1 import Decoder, Encoder_wo_checkpoint, Encoder2

torch.set_default_dtype(torch.float32)
names = locals()

net0 = Encoder_wo_checkpoint().cuda()
net = Encoder2().cuda()
denet_2 = Decoder().cuda()
net0.eval()
net.eval()
denet_2.eval()

model_dicts = {}

def load_sample_model(sample):
    MODELA_PATH = "/public/home/hpyu/ZPGH/G3D/retrain_orca/final_models/%s.encoder.checkpoint" % sample
    MODEL2M_PATH = "/public/home/hpyu/ZPGH/G3D/retrain_orca/final_models/%s.encoder2.checkpoint" % sample
    DENET2_PATH = "/public/home/hpyu/ZPGH/G3D/retrain_orca/final_models/%s.decoder2.checkpoint" % sample
    model_dicts[f"{sample}_net0"] = {k.replace("module.", ""): v for k, v in torch.load(MODELA_PATH).items()}
    model_dicts[f"{sample}_net"] = {k.replace("module.", ""): v for k, v in torch.load(MODEL2M_PATH).items()}
    model_dicts[f"{sample}_denet_2"] = {k.replace("module.", ""): v for k, v in torch.load(DENET2_PATH).items()}


def set_sample_model(sample):
    net0.load_state_dict(model_dicts["%s_net0" % sample])
    net.load_state_dict(model_dicts["%s_net" % sample])
    denet_2.load_state_dict(model_dicts["%s_denet_2" % sample])

def predict(seq):
    with torch.no_grad():
        encoding0 = net0(seq)
        encoding1, encoding2 = net(encoding0)
        pred = denet_2.forward(encoding2).squeeze()
    return pred

reference_genome=MemmapGenome(
    input_path="%s/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa" % G3D_dir,
    init_unpicklable=False,
    memmapfile="%s/resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap" % G3D_dir,
    blacklist_regions="hg38")

chromosomes = dict(reference_genome.get_chr_lens())
chromosomes.pop("chrX")
chromosomes.pop("chrY")

window_size, step_size = 2000000, 2000000
samples = ['AoTCPCs', 'ADAC418', 'ESCs_day0', 'ESCs_day15', 'ESCs_day5', 'GM23248', 'IMR90_MboI', 'MCF10_ERPR']
motifs = [i.strip() for i in os.popen("cat /public/home/hpyu/ZPGH/G3D/resources/motif/latest_human_mouse_motif_names.txt").readlines()]

for sample in samples:
    load_sample_model(sample)

chrom = "chr1"
chrom_len = chromosomes[chrom]
os.makedirs("./%s_results/" % chrom, exist_ok=True)

chrom_seq = torch.tensor(reference_genome.get_encoding_from_coords(chrom, 0, chrom_len).transpose(-1, -2)).cuda()

for motif in motifs:
    names["d_%s" % motif] = defaultdict(list)
    if os.path.exists("/public/home/hpyu/ZPGH/G3D/resources/motif/split_motif_human/%s/%s.motif_pos.bed" % (chrom, motif)):
        with open("/public/home/hpyu/ZPGH/G3D/resources/motif/split_motif_human/%s/%s.motif_pos.bed" % (chrom, motif),"r") as bedf:
            motif_intervals = bedf.readlines()
            for i in motif_intervals:
                start, end = i.strip().split("\t")[1:]
                start, end = int(start), int(end)
                window_idx = start // window_size
                names["d_%s" % motif][window_idx].append((max(0, start-1) % window_size, min((window_idx+1)*window_size-1, end) % window_size)) # subtract 1 to get right motif start

window_starts = np.arange(0, chrom_len-window_size, step_size)
for window_start in window_starts:
    if os.path.exists("./%s_results/%s:%s-%s.motifs_KO.diffs.npy" % (chrom, chrom, window_start, window_start+window_size)):
        continue
    # print(window_start)
    sequence = chrom_seq[:,window_start:window_start+window_size].unsqueeze(0)
    window_idx = window_start // window_size
    window_motif_bases_num = np.zeros(len(motifs))
    d_motif_pos, d_motif_bases = dict(), dict()
    for (motif_idx, motif) in enumerate(motifs):
        intervals = names["d_%s" % motif][window_idx]
        if not intervals:
            continue
        motif_mark = np.zeros(window_size)
        for (start, end) in intervals:
            motif_mark[start : end] = 1
        motif_positions = np.where(motif_mark == 1)[0]
        motif_bases_num = motif_positions.size
        d_motif_pos[motif_idx], d_motif_bases[motif_idx] = motif_positions, motif_bases_num
        window_motif_bases_num[motif_idx] = motif_bases_num
    np.save("./%s_results/%s:%s-%s.motif_bases_num.npy" % (chrom, chrom, window_start, window_start+window_size), window_motif_bases_num)

    samples_diffs, samples_diff_ratios, samples_diff_by_bases = np.zeros((len(samples), len(motifs))), np.zeros((len(samples), len(motifs))), np.zeros((len(samples), len(motifs)))
    for (sample_idx, sample) in enumerate(samples):
        set_sample_model(sample)
        base_pred = predict(sequence).detach().cpu().numpy()
        base_pred_sum = np.sum(base_pred)
        window_diffs, window_diff_ratios, window_diff_by_bases = np.zeros(len(motifs)), np.zeros(len(motifs)), np.zeros(len(motifs))
        for (motif_idx, motif) in enumerate(motifs):
            intervals = names["d_%s" % motif][window_idx]
            if not intervals:
                continue
            sequence_mut = sequence.clone()
            motif_bases_num = d_motif_bases[motif_idx]
            random_idx = torch.randint(low=0, high=4, size=(motif_bases_num,), dtype=torch.long)
            one_hot_bases = F.one_hot(random_idx, num_classes=4).float().T.cuda()
            motif_positions = d_motif_pos[motif_idx]
            sequence_mut[0, :, motif_positions] = one_hot_bases
            mut_pred = predict(sequence_mut).detach().cpu().numpy()
            diff = np.sum(np.abs(mut_pred - base_pred))
            diff_ratio = diff / base_pred_sum
            diff_by_bases = diff / motif_bases_num
            window_diffs[motif_idx], window_diff_ratios[motif_idx], window_diff_by_bases[motif_idx] = diff, diff_ratio, diff_by_bases
        samples_diffs[sample_idx, :], samples_diff_ratios[sample_idx, :], samples_diff_by_bases[sample_idx, :] = window_diffs, window_diff_ratios, window_diff_by_bases
    np.save("./%s_results/%s:%s-%s.motifs_KO.diffs.npy" % (chrom, chrom, window_start, window_start+window_size), samples_diffs)
    np.save("./%s_results/%s:%s-%s.motifs_KO.diff_ratios.npy" % (chrom, chrom, window_start, window_start+window_size), samples_diff_ratios)
    np.save("./%s_results/%s:%s-%s.motifs_KO.diff_by_bases.npy" % (chrom, chrom, window_start, window_start+window_size), samples_diff_by_bases)
