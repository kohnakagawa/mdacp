library(hash)

root_dir <- "./cpu_profile/mdunit512"
## root_dir <- "./gpu_profile/mdunit512"

kinds <- c("force", "comm", "pair", "all")
suffix <- "00000.dat"
summed_time <- hash()
for (k in kinds) {
  summed_time[k] <- 0.0
}

for (k in kinds) {
  fname <- paste(k, suffix, sep = "")
  fname <- paste(root_dir, fname, sep = "/")
  dat <- data.matrix(read.table(fname))
  summed_time[k] <- sum(dat)
}

print(summed_time)

tot_time = summed_time[["all"]]
summed_time["all"] <- NULL
summed_time["other"] <- tot_time - summed_time[["force"]] - summed_time[["comm"]] - summed_time[["pair"]]

for (k in c("force", "comm", "pair", "other")) {
  summed_time[[k]] <- summed_time[[k]] / tot_time
}

print(summed_time)
