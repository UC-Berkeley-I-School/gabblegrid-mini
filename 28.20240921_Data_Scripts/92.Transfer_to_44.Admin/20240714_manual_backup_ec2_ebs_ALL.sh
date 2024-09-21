#!/bin/bash

# Set variables
SOURCE_DIR="/home/ubuntu/efs-w210-capstone-ebs"
BUCKET_NAME="w210-capstone"
BACKUP_PREFIX="01.Backups/01.Manual"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
S3_TARGET="s3://$BUCKET_NAME/$BACKUP_PREFIX/$TIMESTAMP/"

# List of directories to backup
DIRS=(
    "00.GabbleGrid"
    "01.EDA"
    "02A.Pre_Baseline_Models"
    "02B.Baseline_Models"
    "03.Final_Models"
    "04.AgentGroup3_Hist_Weather_TBD"
    "06.Inference_and_Deployment"
    "06A.Inference_Data_Prep"
    "07.Scripts"
    "08.GIT_Repos_Anomaly_Detection"
    "09.Models"
    "11.Data"
    "12.Agents_TOI_AutoGen"
    "13.Agents_TOI_MetaGPT_ChatDev_etc"
    "14.Agent_POCs"
    "15.Agents_Source_Code"
    "16.Hadoop_Spark_Kafka"
    "17.w261_ML_Scale_Deliverables__REF__"
    "18.w266_NLP"
    "30.Templates"
    "31.Admin_Tasks_Backup_etc"
    "32.Papers"
    "33.Essential_Reading"
    "97.Archive"
)

# Perform the backup for each directory
for dir in "${DIRS[@]}"; do
    echo "Backing up $SOURCE_DIR/$dir to $S3_TARGET/$dir"
    aws s3 cp --recursive "$SOURCE_DIR/$dir" "$S3_TARGET/$dir" --exclude ".*" --exclude "*/.*"

    # Check if the backup was successful
    if [ $? -eq 0 ]; then
        echo "Backup of $dir completed successfully."
    else
        echo "Backup of $dir failed."
    fi
done

