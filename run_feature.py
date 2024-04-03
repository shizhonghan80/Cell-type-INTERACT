from collections import OrderedDict
from tokenization import Tokenizer
from scipy.stats import pearsonr, spearmanr
import numpy as np
import random 
import os
import sys
import json

cell_type = sys.argv[1]
##input_chrom = sys.argv[2]

window = 2001
chr_len = {"chr1":249250621,"chr2":243199373,"chr3":198022430,"chr4":191154276,"chr5":180915260,"chr6":171115067,\
           "chr7":159138663,"chr8":146364022,"chr9":141213431,"chr10":135534747,"chr11":135006516,"chr12":133851895,\
           "chr13":115169878,"chr14":107349540,"chr15":102531392,"chr16":90354753,"chr17":81195210,"chr18":78077248,\
           "chr19":59128983,"chr20":63025520,"chr21":48129895,"chr22":51304566}


def label_sequence(sequence, MAX_SEQ_LEN):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(sequence):
        X[i] = nucleotide_ind[ch]
    return X


def read_type_chrom(chrom):
    chrom_data = {}
    for type_name in [cell_type]:
        print(type_name)
        infile = open("./datasets/small_eval/"+type_name+"/"+chrom+".tsv","r")
        line = infile.readline()
        for line in infile:
            line = line.strip("\n").split()
            pos, strand, mc_class, mc_count, total_count = int(line[2]), line[3], line[4], float(line[5]), float(line[6])
            key = chrom + "_" + str(pos)
            if key not in chrom_data.keys():
                chrom_data[key] = [chrom, pos, strand, mc_count, total_count, mc_class]
            else:
                chrom_data[key][3] = chrom_data[key][3] + mc_count
                chrom_data[key][4] = chrom_data[key][4] + total_count
    return chrom_data

        
def run_cell_type(chrom):
    nucleotide = {"A":0, "T":1, "C":2, "G":3, "N":4}
    methylation_data, length, num_pos, num_neg = [], int(window/2), 0, 0
    print("processing " + chrom)
    chrom_data = read_type_chrom(chrom)
    for cpg_key in chrom_data.keys():
        cpg_site = chrom_data[cpg_key]
        chrom, pos, strand, mc_count, total_count, mc_class = cpg_site[0], cpg_site[1], cpg_site[2], cpg_site[3], cpg_site[4], cpg_site[5]
        methyl_level = (mc_count > 0)
        item = {"cell_type": cell_type, "chrom": chrom, "pos": pos, "strand": strand, "target": methyl_level}
        methylation_data.append(item)
    print("The number of samples is " + str(len(methylation_data)))
    return methylation_data


def read_neuron_chrom(chrom):
    chrom_data = {}
    for type_name in ["L23","L4","L5","L6","Ndnf","Pvalb","Sst","Vip"]:
        print(type_name)
        infile = open("./datasets/small_eval/"+type_name+"/"+chrom+".tsv","r")
        line = infile.readline()
        for line in infile:
            line = line.strip("\n").split()
            pos, strand, mc_class, mc_count, total_count = int(line[2]), line[3], line[4], float(line[5]), float(line[6])
            if strand == "-":
                pos = pos - 1
                strand = "+"
            key = chrom + "_" + str(pos)
            if key not in chrom_data.keys():
                chrom_data[key] = [chrom, pos, strand, mc_count, total_count, mc_class]
            else:
                chrom_data[key][3] = chrom_data[key][3] + mc_count
                chrom_data[key][4] = chrom_data[key][4] + total_count
    return chrom_data


def run_neuron():
    nucleotide = {"A":0, "T":1, "C":2, "G":3, "N":4}
    methylation_data, length = [], int(window/2)
    print("processing " + chrom)
    chrom_data = read_chrom(chrom)
    for cpg_key in chrom_data.keys():
        cpg_site = chrom_data[cpg_key]
        chrom, pos, strand, mc_count, total_count, mc_class = cpg_site[0], cpg_site[1], cpg_site[2], cpg_site[3], cpg_site[4], cpg_site[5]
        methyl_level = mc_count / total_count
        item = {"chrom": chrom, "pos": pos, "strand": strand, "target": methyl_level}
        methylation_data.append(item)
    print("The number of samples is " + str(len(methylation_data)))
    return methylation_data


def read_glia_chrom(chrom):
    chrom_data = {}
    for type_name in ["Astro","Endo","MG","ODC","OPC"]:
        print(type_name)
        infile = open("./datasets/small_eval/"+type_name+"/"+chrom+".tsv","r")
        line = infile.readline()
        for line in infile:
            line = line.strip("\n").split()
            pos, strand, mc_class, mc_count, total_count = int(line[2]), line[3], line[4], float(line[5]), float(line[6])
            if strand == "-":
                pos = pos - 1
                strand = "+"
            key = chrom + "_" + str(pos)
            if key not in chrom_data.keys():
                chrom_data[key] = [chrom, pos, strand, mc_count, total_count, mc_class]
            else:
                chrom_data[key][3] = chrom_data[key][3] + mc_count
                chrom_data[key][4] = chrom_data[key][4] + total_count
    return chrom_data


def run_glia():
    nucleotide = {"A":0, "T":1, "C":2, "G":3, "N":4}
    methylation_data, length = [], int(window/2)
    print("processing " + chrom)
    chrom_data = read_glia_chrom(chrom)
    for cpg_key in chrom_data.keys():
        cpg_site = chrom_data[cpg_key]
        chrom, pos, strand, mc_count, total_count, mc_class = cpg_site[0], cpg_site[1], cpg_site[2], cpg_site[3], cpg_site[4], cpg_site[5]
        methyl_level = mc_count / total_count
        item = {"chrom": chrom, "pos": pos, "strand": strand, "target": methyl_level}
        methylation_data.append(item)
    print("The number of samples is " + str(len(methylation_data)))
    return methylation_data
     

def read_GENOME():
    genome, size = {}, {}
    chromo, current_chr = list(), ""
    DNA_file = open("./datasets/hg19.fa")
    for line in DNA_file:
        line = line.strip("\t\r\n")
        if ">chr" in line:
            print(line)
            if current_chr == "":
                line = line.split()
                current_chr = line[0][1:]
            else:
                genome[current_chr],size[current_chr] = chromo,len(chromo)
                chromo,line = list(),line.split()
                current_chr = line[0][1:]
        elif ">" in line:
            genome[current_chr],size[current_chr] = chromo,len(chromo)
            break
        else:
            #if current_chr == input_chrom:
            sequence = label_sequence(line,len(line))
            sequence = list(sequence)
            chromo.extend(sequence) 
    for i in range(1,23):
        print("the length of chr %d is %d " % (i,size["chr"+str(i)]))
    print("the length of chrX is %d" % (size["chrX"]))
    print("the length of chrY is %d" % (size["chrY"]))
    print("the length of chrM is %d" % (size["chrM"]))
    np.save("./datasets/genome", genome, allow_pickle=True)
    return genome


if __name__=="__main__":
    for idx in range(1,23):
        input_chrom = "chr"+str(idx)
        print(input_chrom)
        #Constructs training data, validation data and test_data for each cell type to finetune INTERACT model
        methylation_data = run_cell_type(input_chrom)
        #Constructs training data and validation data for glia to prerain INTERACT model
        #methylation_data = run_glia()
        #Constructs training data and validation data for neuron to prerain INTERACT model
        #methylation_data = run_neuron()
        json.dump(methylation_data, open("./datasets/small_eval/"+cell_type+"/"+input_chrom+".json","w"))
