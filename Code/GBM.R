load("D:/S_drug/Cmap_data_proce/R1/gbm/train_gbm_drug.Rdata")
gene_names <- row.names(up_feature_matrix_gbm)
save(gene_names,file = "outdata/gene_names.Rdata")
load("D:/S_drug/Cmap_data_proce/R1/gbm/gene_names.Rdata")

att <- read.csv("gbm/attention_map_GBM.csv") #行是疾病 列是药物
save(att,file = "gbm/att_gbm.Rdata")
dim(att)
gene_ann <- data.frame(Gene=gene_names,Regulate=c(rep('up',length(gene_names)),rep('down',length(gene_names))))

max_index <- which(att == max(att), arr.ind = TRUE)
rindex <- c(3225:3235)
cindex <- c(6479:6489)

att <- att[rindex,cindex]
gene_ann_row <- gene_ann[rindex,]
gene_ann_col <- gene_ann[cindex,]
row.names(att) <- paste0(gene_ann_row$Gene,"_",gene_ann_row$Regulate)
colnames(att) <- paste0(gene_ann_col$Gene,"_",gene_ann_col$Regulate)

#pilaralisib
gene_ann[3230,] #GNAL
gene_ann[6484,] #ENPP3       
train_gbm_drug[length(train_gbm_drug)]

att <- apply(att, 1, mean)
which.max(att)
gene_ann[10199,]
gene_ann$attn_socre <- att
gene_ann$attn_socre <- (gene_ann$attn_socre-min(gene_ann$attn_socre))/(max(gene_ann$attn_socre)-min(gene_ann$attn_socre))



library(pheatmap)
# 提取分数矩阵（热图数据）
score_matrix <- as.matrix(gene_ann$attn_socre)
rownames(score_matrix) <- paste0(gene_ann$Gene,"_",gene_ann$Regulate)  # 用基因名作为行名

# 设置行注释（up/down 分类）
annotation_row <- data.frame(GBM_gene = gene_ann$Regulate)
rownames(annotation_row) <- paste0(gene_ann$Gene,"_",gene_ann$Regulate)

# 设置颜色（up=红色，down=蓝色）
ann_colors <- list(GBM_gene = c(up = "red", down = "blue"))

labels_row <- rep("", nrow(score_matrix))  # 全部初始化为空
labels_row[10199] <- gene_ann$Gene[10199]         # 第200位显示基因名

pheatmap(
  score_matrix,
  cluster_rows = FALSE,          # 不聚类行
  cluster_cols = FALSE,          # 不聚类列
  show_rownames = TRUE,          # 必须为TRUE才能显示自定义标签
  labels_row = labels_row,       # 指定自定义行标签
  annotation_row = annotation_row,
  annotation_colors = ann_colors,
  fontsize_row = 8,              # 调整行名字体大小
  gaps_row = 10198,                # 在第199行后加白线（即第200行上方）
  main = "Pilaralisib (0.01uM_4h))"
)
p@name <- "Attn_socre"
leg_label <- textGrob("Temperature [°C]",x=0,y=0.9,hjust=0,vjust=0,gp=gpar(fontsize=10,fontface="bold"))
####
library(ggplot2)
library(dplyr)
library(tibble)

# 准备数据
gene_ann <- gene_ann %>% 
  mutate(
    row_id = row_number(),
    row_name = ifelse(row_id == 10199, Gene, ""),  # 只显示第10199行的基因名
    row_group = factor(Regulate, levels = c("up", "down"))
  )

# 创建热图
ggplot(gene_ann, aes(x = 1, y = reorder(row_id, -row_id), fill = attn_socre)) +
  # 热图主体
  geom_tile() +
  
  # 添加行分组颜色条（左侧）
  geom_tile(aes(x = 0, fill = NULL, color = row_group), 
            width = 0.1, height = 0.9, size = 0.5) +
  scale_color_manual(values = c("up" = "red", "down" = "blue"), name = "Regulate") +
  
  # 设置热图颜色
  scale_fill_gradient(low = "white", high = "black", name = "attn_socre") +
  
  # 添加特定行标签和指示线
  geom_text(aes(x = 1.05, label = row_name), 
            hjust = 0, size = 3, check_overlap = TRUE) +
  geom_segment(
    aes(x = 1.01, xend = 1.04, y = row_id, yend = row_id),
    data = gene_ann %>% filter(row_id == 10199),
    linetype = "solid", color = "black"
  ) +
  
  # 在第10198行后添加分隔线
  geom_hline(yintercept = 10198.5, color = "white", linewidth = 1) +
  
  # 调整主题和坐标
  labs(title = "Pilaralisib (0.01uM_4h)", x = "", y = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_blank(),
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  ) +
  coord_cartesian(clip = "off")  # 防止标签被裁剪  
