DOWNLOAD_DIR=/home/jupyter/download
EXTRACT_DIR=/home/jupyter/rsna/source_data

#https://stackoverflow.com/questions/59838/check-if-a-directory-exists-in-a-shell-script
if [ ! -d "$DOWNLOAD_DIR" ]; then
  mkdir $DOWNLOAD_DIR && echo 'mkdir $DOWNLOAD_DIR'
else
    echo ${DOWNLOAD_DIR}" exists"
fi

cd $DOWNLOAD_DIR && echo "cd "$DOWNLOAD_DIR


kaggle competitions download -c rsna-intracranial-hemorrhage-detection

unzip $DOWNLOAD_DIR/rsna-intracranial-hemorrhage-detection.zip -d $EXTRACT_DIR -q