#!/bin/bash

# Set variables
SOURCE_DIR="/home/sagemaker-user"
BUCKET_NAME="w210-capstone"
BACKUP_PREFIX="01.Backups/01.Manual"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
S3_TARGET="s3://$BUCKET_NAME/$BACKUP_PREFIX/$TIMESTAMP/"

# Perform the backup
echo "Backing up $SOURCE_DIR to $S3_TARGET"
aws s3 cp --recursive $SOURCE_DIR $S3_TARGET --exclude ".*" --exclude "*/.*"

# Check if the backup was successful
if [ $? -eq 0 ]; then
  echo "Backup completed successfully."
else
  echo "Backup failed."
fi