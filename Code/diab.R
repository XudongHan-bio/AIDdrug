load("D:/S_drug/Drug/diabetes2x/outdata/train_diab_drug.Rdata")
load("D:/S_drug/Cmap_data_proce/R1/gbm/gene_names.Rdata")
att <- read.csv("diab/attention_map_diab.csv") #行是疾病 列是药物
save(att,file = "diab/att_diab.Rdata")
siginfo = data.table::fread("D:/S_drug/data/cmap/siginfo_beta.txt",data.table=FALSE)

gene_ann <- data.frame(Gene=gene_names,Regulate=c(rep('up',length(gene_names)),rep('down',length(gene_names))))

att_m <- apply(att, 1, mean)
which.max(att_m)
gene_ann[12729,] #LRRC8C
which(gene_ann$Gene=="SCD1")

train_diab_drug[length(train_diab_drug)] #"REP.A001_A375_24H:N07" #MK-8245 10uM 24 h

gene_ann$attn_socre <- att_m
gene_ann$attn_socre <- (gene_ann$attn_socre-min(gene_ann$attn_socre))/(max(gene_ann$attn_socre)-min(gene_ann$attn_socre))
gene_ann$attn_socre[12729]

library(pheatmap)
# 提取分数矩阵（热图数据）
score_matrix <- as.matrix(gene_ann$attn_socre)
rownames(score_matrix) <- paste0(gene_ann$Gene,"_",gene_ann$Regulate)  # 用基因名作为行名

# 设置行注释（up/down 分类）
annotation_row <- data.frame(diabetes_gene = gene_ann$Regulate)
rownames(annotation_row) <- paste0(gene_ann$Gene,"_",gene_ann$Regulate)

# 设置颜色（up=红色，down=蓝色）
ann_colors <- list(diabetes_gene = c(up = "red", down = "blue"))

labels_row <- rep("", nrow(score_matrix))  # 全部初始化为空
labels_row[12729] <- gene_ann$Gene[12729]         # 第200位显示基因名

pheatmap(
  score_matrix,scale="none",
  cluster_rows = FALSE,          # 不聚类行
  cluster_cols = FALSE,          # 不聚类列
  show_rownames = TRUE,          # 必须为TRUE才能显示自定义标签
  labels_row = labels_row,       # 指定自定义行标签
  annotation_row = annotation_row,
  annotation_colors = ann_colors,
  fontsize_row = 8,              # 调整行名字体大小
  gaps_row = 12728,                # 在第199行后加白线（即第200行上方）
  main = "MK-8245 (10uM_24h))"
)
