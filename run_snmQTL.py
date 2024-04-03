import sys
import random
import gzip
import numpy as np
import scipy.stats as stats
import math
import os

cell_type = sys.argv[1]
target_chrom = sys.argv[2]

whole_chroms = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]

def chrom_length():
    chr_len = {}
    chr_len["chr1"] = 249250621
    chr_len["chr2"] = 243199373
    chr_len["chr3"] = 198022430
    chr_len["chr4"] = 191154276
    chr_len["chr5"] = 180915260
    chr_len["chr6"] = 171115067
    chr_len["chr7"] = 159138663
    chr_len["chr8"] = 146364022
    chr_len["chr9"] = 141213431
    chr_len["chr10"] = 135534747
    chr_len["chr11"] = 135006516
    chr_len["chr12"] = 133851895
    chr_len["chr13"] = 115169878
    chr_len["chr14"] = 107349540
    chr_len["chr15"] = 102531392
    chr_len["chr16"] = 90354753
    chr_len["chr17"] = 81195210
    chr_len["chr18"] = 78077248
    chr_len["chr19"] = 59128983
    chr_len["chr20"] = 63025520
    chr_len["chr21"] = 48129895
    chr_len["chr22"] = 51304566
    chr_len["chrX"] = 155270560
    chr_len["chrY"] = 59373566
    return chr_len

def read_proportion():
    infile = open("/dcs04/lieber/statsgen/jiyunzhou/Achive/scmQTL/output/Finemap/"+cell_type+"/difference/"+target_chrom+".txt","r")
    ref_low_number, ref_high_number = 0, 0
    var_low_number, var_high_number = 0, 0
    line = infile.readline()
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        ref_value, var_value = float(line[4]), float(line[6])
        if ref_value < 0.5: ref_low_number = ref_low_number + 1
        elif ref_value > 0.5: ref_high_number = ref_high_number + 1

        if var_value < 0.5: var_low_number = var_low_number + 1
        elif var_value > 0.5: var_high_number = var_high_number + 1
        if (idx+1) % 1000 == 0:
            ref_proportion = ref_low_number / (ref_low_number + ref_high_number)
            var_proportion = var_low_number / (var_low_number + var_high_number)
            print(str(idx) + "\t" + str(ref_proportion) + "\t" + str(var_proportion))
    return ref_proportion, var_proportion

def read_cpg_annot():
    cpg_annot = {}
    cpg_file = open("./datasets/cpg.annotation.csv")
    line = cpg_file.readline()
    for line in cpg_file:
        line = line.split(",")
        number = line[0][1:-1]
        ID = line[1][1:-1]
        chrom = "chr"+line[3][1:-1]
        pos = int(line[4])
        cpg_annot[chrom+"_"+str(pos)] = ID
    return cpg_annot

def get_pos():
    genome = np.load("./datasets/backup/genome.npy",allow_pickle=True).item()
    infile = open("./datasets/1KGCEU_qc.bim","r")
    pos2id = {}
    for idx,line in enumerate(infile):
        line = line.strip("\n").split("\t")
        snp_chr,snp_pos,snp_id = "chr"+line[0],int(line[3]),line[1]
        A1,A2 = line[4],line[5]
        if genome[snp_chr][snp_pos-1] == A1:
            REF,ALT = A1,A2
        else:
            REF,ALT = A2,A1
        pos2id[snp_chr+"_"+str(snp_pos)] = [snp_id,REF,ALT]
    return pos2id

def read_SNP():
    infile = open("./data/snp_map.txt","r")
    SNP = {}
    line = infile.readline()
    for line in infile:
        line = line.strip("\n").split("\t")
        snp_chr,snp_pos,snp_id = "chr"+line[0],int(line[2]),line[1]
        A1,A2 = line[3],line[4]
        element = [snp_chr,snp_pos,snp_id,A1,A2]
        SNP[snp_id] = element
    return SNP

def max_pos_diff():
    chr_len = chrom_length()
    chrom = target_chrom
    ref_mqtl,var_mqtl = {},{}
    pos2id = get_pos()
    cpg_annot = read_cpg_annot()
    input_files = os.listdir("./outputs/merge_genome/"+cell_type+"/reference/"+chrom+"/")
    for input_file in input_files:
        if ".txt" not in input_file: continue
        print("reading reference " + input_file)
        ref_file = open("./outputs/merge_genome/"+cell_type+"/reference/"+chrom+"/"+input_file, "r")
        for ref_line in ref_file:
            line = ref_line.strip("\n").split("\t")
            item = [int(line[0]),int(line[1])] + line[2:]
            ref_mqtl[line[0]+"_"+line[1]] = item
    input_files = os.listdir("./outputs/merge_genome/"+cell_type+"/variation/"+chrom+"/")
    for input_file in input_files:
        if ".txt" not in input_file: continue
        print("reading variation " + input_file)
        var_file = open("./outputs/merge_genome/"+cell_type+"/variation/"+chrom+"/"+input_file, "r")
        for index,var_line in enumerate(var_file):
            line = var_line.strip("\n").split("\t")
            try: item = [int(line[0]), int(line[1])] + line[2:]
            except: continue
            var_mqtl[line[0]+"_"+line[1]] = item
    map_mqtl = {}
    for k in ref_mqtl.keys():
        if k not in var_mqtl.keys(): continue
        ref_level,var_level = float(ref_mqtl[k][2]),float(var_mqtl[k][2])
        diff,ab_diff = var_level-ref_level,np.abs(var_level-ref_level)
        cpg_pos,snp_pos = ref_mqtl[k][0],ref_mqtl[k][1] 
        if chrom+"_"+str(snp_pos) not in pos2id.keys(): continue
        var_id, REF, ALT = pos2id[chrom+"_"+str(snp_pos)]
        cpg_id = chrom+"_"+str(cpg_pos)
        distance = abs(cpg_pos - snp_pos)
        element = [chrom,var_id,snp_pos,REF,ALT,cpg_id,cpg_pos,ref_level,var_level,diff,ab_diff]
        try: 
            if map_mqtl[var_id][10] < ab_diff: map_mqtl[var_id] = element
        except: map_mqtl[var_id] = element
    mqtl = map_mqtl.values()
    mqtl = sorted(mqtl,key=lambda element: element[10],reverse=True)
    outfile = open("./outputs/merge_genome/"+cell_type+"/difference/"+chrom+".txt","w")
    for k in range(0,len(mqtl)):
        element = mqtl[k]
        outline = element[0]+"\t"+str(element[6])+"\t"+str(element[2])+"\t"+str(element[3])+"\t"+str(element[7])+"\t"+\
        str(element[4])+"\t"+str(element[8])+"\t"+str(element[10])+"\n"
        outfile.write(outline)
    outfile.close()


def combine_chrom():
    genome_snps = []
    output_file = open("./outputs/merge_genome/"+cell_type+"/difference/genome.txt","w")
    for chrom in whole_chroms:
        input_file = "./outputs/merge_genome/"+cell_type+"/difference/"+chrom+".txt"
        if os.path.exists(input_file) == False: continue
        infile = open(input_file,"r")
        print(chrom)
        for line in infile:
            line = line.strip("\n").split("\t")
            genome_snps.append([line[0],line[1],line[2],line[3],line[4],line[5],line[6],float(line[7])])
    genome_snps = sorted(genome_snps,key=lambda element: element[7],reverse=True)
    output_file.write("chrom\tcpg_pos\tsnp_pos\tref_allele\tref_value\tvar_allele\tvar_value\tdifference\n")
    for index in range(0,len(genome_snps)):
        element = genome_snps[index]
        line = str(element[0])+"\t"+str(element[1])+"\t"+str(element[2])+"\t"+str(element[3])+"\t"+str(element[4])+"\t"+str(element[5])\
               +"\t"+str(element[6])+"\t"+str(element[7])+"\n"
        output_file.write(line)
    output_file.close()


if __name__ == "__main__":
    #read_proportion()
    max_pos_diff()
    #combine_chrom()
