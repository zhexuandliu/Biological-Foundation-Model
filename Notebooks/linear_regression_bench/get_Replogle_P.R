library(SingleCellExperiment)
library(tidyverse)
library(glmGamPoi)



# Load data
folder <- "../data/GEARS/"
sce <- zellkonverter::readH5AD(file.path("../data/GEARS/perturb_processed.h5ad"))

# Clean up the colData(sce) a bit
sce$condition <- droplevels(sce$condition)
sce$clean_condition <- stringr::str_remove(sce$condition, "\\+ctrl")

gene_names <- rowData(sce)[["gene_name"]]
rownames(sce) <- gene_names

baseline <- MatrixGenerics::rowMeans2(assay(sce, "X")[,sce$condition == "ctrl",drop=FALSE])

# Pseudobulk everything
psce <- glmGamPoi::pseudobulk(sce, group_by = vars(condition, clean_condition))
assay(psce, "change") <- assay(psce, "X") - baseline


pca <- irlba::prcomp_irlba(t(as.matrix(assay(psce, "X"))), n = 10)
pert_emb <- t(pca$x)
colnames(pert_emb) <- psce$clean_condition


# Store output
as.data.frame(pert_emb) |>
  write_tsv(out_file)


#### Session Info
sessionInfo()
