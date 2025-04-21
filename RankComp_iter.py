#! /usr/bin/env python3
import argparse
import scipy.stats as stats
import numpy as np
import math
import os
from collections import defaultdict
from collections import Counter
from itertools import combinations
from multiprocessing import Pool

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="""Get DEG based on RankComp. Input contains expression file separated by tab, the 1st columns is the ID, the rest columns as samples and rows as genes.
                                 Also you need to set cutoff and determine which columns are control (the rest will be regarded as treated).
                                 Please try your best to avoid a super big expression matrix, e.g. first discard genes with extremelly low expression.
                                 Exp: RankComp_iter.py -i TPM.txt -c 0.99 0.9 -s SampleSheet.txt -g Treat~Control -p 10""")
parser.add_argument("-i", "--input", nargs=1, help="input expression file", required=True)
parser.add_argument("-c", "--cutoffs", nargs="+", help="cut-off for determine stable gene pairs", required=True)
parser.add_argument("-s", "--sample", nargs=1, help="use a sample sheet to identify samples, first col is name, second col is group", required=True)
parser.add_argument("-g", "--group", nargs=1, help="groups or samples used for DEG detecting. format: a~b, a is treat and b is control, multiple groups separated by commas", default=["all"], required=True)
parser.add_argument("-pre", "--prefix", nargs=1, default=["prefix"])


args = parser.parse_args()
input_file, cutoffs, sample_sheet, group, prog, prefix = args.input[0], list(map(float, args.cutoffs)), args.sample[0], args.group[0], args.progress[0], args.prefix[0]
names = locals()

d_group_sample = defaultdict(list)
with open(sample_sheet)as f:
    lines = f.readlines()
    for line in lines:
        d_group_sample[line.strip().split("\t")[1]].append(line.strip().split("\t")[0])
        d_group_sample[line.strip().split("\t")[0]].append(line.strip().split("\t")[0])

# groups = group.split(",")
# Get the rank array, no problem...
d_col = dict()
Genes, data_list = list(), list()
with open(input_file)as f:
    header = f.readline()
    samples = header.rstrip().split("\t")[1:]
    for i in enumerate(samples):
        d_col[i[1]] = i[0]
    lines = f.readlines()
    for line in lines:
        Gene = line.split("\t")[0]
        Genes.append(Gene)
        res = list(map(float, line.strip().split("\t")[1:]))
        data_list.append(res)
data_array = np.transpose(np.array(data_list))
rank_array = np.argsort(np.argsort(-data_array))
len_Genes = len(Genes)

def contingency_for_gene(g, rank_t, pair_array, cutoff, ctrl_n, deg_mask):
    C_up = C_down = T_up = T_down = up_rev = down_rev = 0
    for j in range(len_Genes):
        if deg_mask[j]:
            continue
        v = pair_array[g, j] / ctrl_n
        if   v >= cutoff:
            C_up += 1
            if rank_t[g] > rank_t[j]:
                T_down += 1; up_rev += 1
            elif rank_t[g] < rank_t[j]:
                T_up += 1
        elif v <= -cutoff:
            C_down += 1
            if rank_t[g] < rank_t[j]:
                T_up += 1; down_rev += 1
            elif rank_t[g] > rank_t[j]:
                T_down += 1
    return C_up, C_down, T_up, T_down, up_rev, down_rev

from statsmodels.stats.multitest import multipletests

def rankcomp_iter(treat_col, cutoff, pair_array, prefix):
    sample_name = samples[treat_col]
    rank_t      = rank_array[treat_col]
    ctrl_n      = len(col_controls)
    max_iter    = 30
    deg_mask    = np.zeros(len_Genes, dtype=bool)   # False = background

    for it in range(1, max_iter+1):
        C_up_l, C_down_l, T_up_l, T_down_l = [], [], [], []
        odds_l, pval_l, type_l = [], [], []

        for g in range(len_Genes):
            C_up, C_down, T_up, T_down, urev, drev = \
                contingency_for_gene(g, rank_t, pair_array, cutoff,
                                     ctrl_n, deg_mask)
            if C_up + C_down == 0:
                odds, p = 1.0, 1.0
                gtype   = 'none'
            else:
                odds = ((T_up+1)/(T_down+1)) / ((C_up+1)/(C_down+1))
                p    = stats.fisher_exact([[C_up, C_down], [T_up, T_down]])[1]
                if   odds > 1: gtype = 'up'
                elif odds < 1: gtype = 'down'
                else:          gtype = 'none'

            C_up_l.append(C_up);   C_down_l.append(C_down)
            T_up_l.append(T_up);   T_down_l.append(T_down)
            odds_l.append(odds);   pval_l.append(p);   type_l.append(gtype)

        fdr_l = multipletests(pval_l, method='fdr_bh')[1]
        new_deg_mask = np.array(fdr_l < 0.05)

        print(f'  Iter {it:2d}:  DEG={new_deg_mask.sum()}')

        if np.array_equal(new_deg_mask, deg_mask):
            break
        deg_mask = new_deg_mask

    out = '%s.RankComp.txt' % prefix
    with open(out, 'w') as fo:
        fo.write('ID\tType\tCtrlUp\tCtrlDown\tTreatUp\tTreatDown\tOdds\tlog2Odds\tPval\tFDR\n')
        for i,gene in enumerate(Genes):
            fo.write('\t'.join(map(str, [
                gene, type_l[i], C_up_l[i], C_down_l[i],
                T_up_l[i], T_down_l[i], round(odds_l[i],3), round(math.log2(odds_l[i]),3),
                "{:.3g}".format(pval_l[i]), "{:.3g}".format(fdr_l[i])
            ]))+'\n')

def run_control(col):
    print("start processing control %s..." % samples[col])
    names["%s_pair_array" % col] = np.zeros((len_Genes, len_Genes), np.int0)
    counter = 0
    comb = combinations(list(range(len(Genes))), 2)  # An iterator generates M*(M-1)/2 non-repeated gene-pairs.
    for i in comb:
        counter += 1
        ID1, ID2 = i[0], i[1]
        if counter % 40000000 == 0:
            print("processing the %sth pair of %s..." % (counter, samples[col]))
        rank1, rank2 = rank_array[col][ID1], rank_array[col][ID2]
        if rank1 > rank2:  # means that expression of ID1 < ID2, ID1 vs ID2 is down
            names["%s_pair_array" % col][ID1][ID2] = -1
            names["%s_pair_array" % col][ID2][ID1] = 1
        elif rank1 < rank2:  # vice versa
            names["%s_pair_array" % col][ID1][ID2] = 1
            names["%s_pair_array" % col][ID2][ID1] = -1
    print("end processing control %s!" % samples[col])
    return names["%s_pair_array" % col]
# Now all the gene status in control samples are recorded in 2 arrays

def control_pool(prog, ctrls):
    p = Pool(prog)
    results = list()
    for col in ctrls:
        results.append(p.apply_async(run_control, args=(col,)))
    p.close()
    p.join()
    print("start adding control pair array...")
    pair_arrays = list()
    for i in results:
        pair_arrays.append(i.get())
    control_pair_array = np.sum(pair_arrays, axis=0)
    print("end adding control pair array!")
    return control_pair_array

def treat_pool_iter(prog, pair_array, treats, prefix):
    p = Pool(prog)
    for col in treats:
        for cutoff in cutoffs:
            p.apply_async(rankcomp_iter, args=(col, cutoff, pair_array, prefix))
    p.close()
    p.join()


# for group in groups:
col_treats, col_controls = list(), list()
current_treat, current_control = group.split("~")
current_treat, current_control = current_treat.split(","), current_control.split(",")
for treat in current_treat:
    for sample in d_group_sample[treat]:
        col_treats.append(d_col[sample])
for control in current_control:
    for sample in d_group_sample[control]:
        col_controls.append(d_col[sample])
pair_array = control_pool(prog, col_controls)
treat_pool_iter(prog, pair_array, col_treats, prefix)
