# 需要配置kaggle api
# https://github.com/Kaggle/kaggle-api

pip install kaggle > /dev/null

DOWNLOAD_DIR=/home/jupyter/download
EXTRACT_DIR=/home/jupyter/rsna/source_data

#https://stackoverflow.com/questions/59838/check-if-a-directory-exists-in-a-shell-script
if [ ! -d "$DOWNLOAD_DIR" ]; then
  mkdir -p $DOWNLOAD_DIR && echo 'mkdir $DOWNLOAD_DIR'
else
    echo ${DOWNLOAD_DIR}" exists"
fi

cd $DOWNLOAD_DIR && echo "cd "$DOWNLOAD_DIR


kaggle competitions download -c rsna-intracranial-hemorrhage-detection -o


if [ ! -d "$EXTRACT_DIR" ]; then
  mkdir -p $EXTRACT_DIR && echo 'mkdir $EXTRACT_DIR'
else
    echo ${EXTRACT_DIR}" exists"
fi

echo '=> Start extract to '$EXTRACT_DIR
unzip -qo $DOWNLOAD_DIR/rsna-intracranial-hemorrhage-detection.zip -d $EXTRACT_DIR 
