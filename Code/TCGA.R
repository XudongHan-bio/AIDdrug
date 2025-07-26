exp <- read.table("data/HiSeqV2",sep = "\t",header = T,row.names = 1)
phen <- read.table("data/phen.txt",sep = "\t",header = T)
phen$sampleID <- gsub("-",".",phen$sampleID)
pp <- match(colnames(exp),phen$sampleID)
phen <- phen[pp,]
all(colnames(exp)==phen$sampleID)

pp <- which(row.names(exp)=="ADRB3")
data <- data.frame(v=as.numeric(exp[pp,]),p=phen$sample_type)
boxplot(v~p,data)

length(which(data$v>0))/nrow(data)

zero_ratio <- rowMeans(exp == 0)
filtered_matrix <- expr_matrix[zero_ratio <= 0.25, ]


#整理疾病数据####
path <- "G:/Drug/TCGA_data/output/"
df <- list.files(path)

disease_list <- list()
for(i in 1:length(df)){
  f <- list.files(paste0(path,df[i]))
  index <- grep("FC_result",f)
  d <- read.csv(paste0(path,df[i],"/",f[index]))
  disease_list[[i]] <- d
}
disease_df <- Reduce(function(x, y) merge(x, y, by = "Gene"), disease_list)
dim(disease_df)
row.names(disease_df) <- disease_df[,1]
saveRDS(disease_df,file = "outdata/disease_df.rds")




#构建测试数据####
disease_df <- readRDS("G:/Drug/R_p/r1/outdata/disease_df.rds")
Cmap_data_test <- readRDS("G:/Drug/R_p/r2/outdata/Cmap_data_test.rds")

disease_df <- disease_df[,-1]

jj <- intersect(row.names(disease_df),row.names(Cmap_data_test))
disease_df <- disease_df[jj,]
Cmap_data_test <- Cmap_data_test[jj,]
all(row.names(disease_df)==row.names(Cmap_data_test))

min(disease_df)
min(Cmap_data_test)
#disease_df1 <- disease_df
for(i in 1:ncol(disease_df)){
  pp <- which(disease_df[,i]>=0)
  u <- disease_df[pp,i]
  u <- (u-min(u))/(max(u)-min(u))
  disease_df[pp,i] <- u

  pp <- which(disease_df[,i]<0)
  d <- abs(disease_df[pp,i])
  d <- -((d-min(d))/(max(d)-min(d)))
  disease_df[pp,i] <- d
}
range(disease_df[,1])

for(i in 1:ncol(Cmap_data_test)){
  pp <- which(Cmap_data_test[,i]>=0)
  u <- Cmap_data_test[pp,i]
  u <- (u-min(u))/(max(u)-min(u))
  Cmap_data_test[pp,i] <- u

  pp <- which(Cmap_data_test[,i]<0)
  d <- abs(Cmap_data_test[pp,i])
  d <- -((d-min(d))/(max(d)-min(d)))
  Cmap_data_test[pp,i] <- d
}
range(Cmap_data_test[,1])

dim(disease_df) #9262  196
dim(Cmap_data_test) #9262  600

write.table(disease_df,file = "outdata/disease_matrix.txt",quote = F,sep = "\t",col.names = F,row.names = F)
write.table(Cmap_data_test,file = "outdata/drug_matrix.txt",quote = F,sep = "\t",col.names = F,row.names = F)
