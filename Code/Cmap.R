#Sublab_new /home/xdhan/project/drug/cmap2/
library(cmapR)
trt = parse_gctx("data/level5_beta_trt_cp_n720216x12328.gctx")
Cmap_data<-trt@mat
dim(Cmap_data)
Cmap_data[1:3,1:3]


gene_meta = data.table::fread("data/geneinfo_beta.txt")
pp <- match(row.names(Cmap_data),gene_meta$gene_id)
table(duplicated(pp))
row.names(Cmap_data) <- gene_meta$gene_symbol[pp]
Cmap_data[1:5,1:5]

max(Cmap_data[,1])
min(Cmap_data[,1])
dim(Cmap_data)
saveRDS(Cmap_data,"outdata/Cmap_data.rds")

Cmap_data <- readRDS("outdata/Cmap_data.rds")
Cmap_data_test <- Cmap_data[,1:600]
saveRDS(Cmap_data_test,"outdata/Cmap_data_test.rds")
