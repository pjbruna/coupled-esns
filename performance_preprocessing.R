library(tidyverse)

data <- data.frame(
  X0 = numeric(),
  X1 = numeric(),
  X2 = numeric(),
  X3 = numeric(),
  X4 = numeric(),
  X5 = numeric(),
  X6 = numeric(),
  X7 = numeric(),
  X8 = numeric(),
  TX0 = numeric(),
  TX1 = numeric(),
  TX2 = numeric(),
  TX3 = numeric(),
  TX4 = numeric(),
  TX5 = numeric(),
  TX6 = numeric(),
  TX7 = numeric(),
  TX8 = numeric(),
  model = character(),
  coupling = numeric(),
  run_idx = numeric(),
  signal_idx = numeric(),
  timestep = numeric()
)

y1 <- read.csv(paste0("data/test_06_18/Y1_n=0.3_nnodes=300.csv")) %>%
 mutate(model = "1")

y2 <- read.csv(paste0("data/test_06_18/Y2_n=0.3_nnodes=300.csv")) %>%
 mutate(model = "2")

yjoint <- data.frame("X0" = rowMeans(cbind(y1$X0, y2$X0)),
                     "X1" = rowMeans(cbind(y1$X1, y2$X1)),
                     "X2" = rowMeans(cbind(y1$X2, y2$X2)),
                     "X3" = rowMeans(cbind(y1$X3, y2$X3)),
                     "X4" = rowMeans(cbind(y1$X4, y2$X4)),
                     "X5" = rowMeans(cbind(y1$X5, y2$X5)),
                     "X6" = rowMeans(cbind(y1$X6, y2$X6)),
                     "X7" = rowMeans(cbind(y1$X7, y2$X7)),
                     "X8" = rowMeans(cbind(y1$X8, y2$X8)),
                     "model" = "joint")

y_data <- rbind(y1, y2, yjoint)

meta_data <- read.csv(paste0("data/test_06_18/meta_data_n=0.3_nnodes=300.csv"))

df <- cbind(y_data, meta_data) %>%
 group_by(model, coupling, run_idx, signal_idx) %>%
 mutate(timestep = row_number()) %>%
 slice_tail(n=1)

data <- rbind(data, df)


performance_data <- data %>%
rowwise() %>%
mutate(response = which.max(c(X0, X1, X2, X3, X4, X5, X6, X7, X8)),
      target = which.max(c(TX0, TX1, TX2, TX3, TX4, TX5, TX6, TX7, TX8)),
      correct = ifelse(response==target, 1, 0))

write.csv(performance_data, file = "data/test_06_18/performance_n=0.3_nnodes=300.csv")
