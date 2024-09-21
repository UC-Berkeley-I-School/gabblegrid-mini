#!/bin/bash

# Set AWS credentials
export AWS_ACCESS_KEY_ID="AKIARTUOVM5EHAQZLXIB"
export AWS_SECRET_ACCESS_KEY="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"

# Set variables
SOURCE_DIR="/home/sagemaker-user"
BUCKET_NAME="w210-capstone"
BACKUP_PREFIX="01.Backups/01.Automated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
S3_TARGET="s3://$BUCKET_NAME/$BACKUP_PREFIX/$TIMESTAMP/"
LOG_FILE="/home/sagemaker-user/cron_backup.log"

# Log start time
echo "$(date): Starting backup of $SOURCE_DIR to $S3_TARGET, excluding files larger than 25MB" >> $LOG_FILE

# Perform the backup, excluding files larger than 25MB and preserving relative paths
cd $SOURCE_DIR
find . -type f -size -25M -exec aws s3 cp --only-show-errors {} $S3_TARGET{} \; >> $LOG_FILE 2>&1

# Check if the backup was successful
if [ $? -eq 0 ]; then
  echo "$(date): Backup completed successfully." >> $LOG_FILE
else
  echo "$(date): Backup failed." >> $LOG_FILE
fi
