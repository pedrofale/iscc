#!/bin/bash
# Seed data for stCNASim (not timed): align a public Visium run with STARsolo, call heterozygous
# SNPs, phase them at random, and write the seed, feature, clone and CNA inputs.
#   stcnasim_prep.sh <seed dir> <stcnasim env> <scripts dir> <gencode gtf.gz> <threads>
# <seed dir> holds SRR21230125_{1,2}.fastq.gz, GRCh38.primary_assembly.genome.fa.gz, geo/spatial/
# spatial/tissue_positions_list.csv and inputs/cna_profile.tsv (stCNASim's own 1n2t_aleloh benchmark
# profile, hxj5/bcd f5ef0e2).
set -euo pipefail
S=$1; E=$2; SCR=$3; GTF=$4; T=$5
export PATH=$E/bin:$PATH
cd "$S"
POS=geo/spatial/spatial/tissue_positions_list.csv
[ -f genome.fa ] || { gunzip -c GRCh38.primary_assembly.genome.fa.gz > genome.fa; samtools faidx genome.fa; }
[ -f genes.gtf ] || gunzip -c "$GTF" > genes.gtf
if [ ! -f star_index/SA ]; then
  mkdir -p star_index
  STAR --runMode genomeGenerate --runThreadN "$T" --genomeDir star_index --genomeFastaFiles genome.fa \
       --sjdbGTFfile genes.gtf --sjdbOverhang 100 --outFileNamePrefix star_index/
fi
python "$SCR/stcnasim_inputs.py" whitelist "$POS" whitelist.txt
if [ ! -f solo/Aligned.sortedByCoord.out.bam ]; then
  mkdir -p solo
  STAR --runThreadN "$T" --genomeDir star_index --readFilesIn SRR21230125_2.fastq.gz SRR21230125_1.fastq.gz \
       --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist whitelist.txt \
       --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 \
       --soloStrand Forward --soloFeatures Gene --outSAMtype BAM SortedByCoordinate \
       --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM --limitBAMsortRAM 60000000000 \
       --outFileNamePrefix solo/
  samtools index -@ "$T" solo/Aligned.sortedByCoord.out.bam
fi
python "$SCR/stcnasim_inputs.py" seed "$POS" 600 seed_cell_anno.tsv
python "$SCR/stcnasim_inputs.py" features "$GTF" features.tsv
if [ ! -f snps.het.vcf.gz ]; then
  mkdir -p snp
  seq 1 22 | xargs -P "$T" -I{} sh -c "bcftools mpileup -f genome.fa -r chr{} -a AD,DP -q 20 -Q 20 -d 100000 \
      solo/Aligned.sortedByCoord.out.bam 2>/dev/null | bcftools call -mv -Oz -o snp/chr{}.vcf.gz 2>/dev/null"
  bcftools concat -Oz -o snps.all.vcf.gz $(for i in $(seq 1 22); do echo snp/chr$i.vcf.gz; done)
  bcftools view -v snps -m2 -M2 -g het -i 'QUAL>=20 && INFO/DP>=10' -Oz -o snps.het.vcf.gz snps.all.vcf.gz
fi
python "$SCR/stcnasim_inputs.py" phase snps.het.vcf.gz 0 phased_snps.vcf
cp inputs/cna_profile.tsv cna_profile.tsv
echo "seed spots: $(wc -l < seed_cell_anno.tsv); features: $(wc -l < features.tsv); phased het SNPs: $(grep -vc "^#" phased_snps.vcf)"
echo PREP_DONE
