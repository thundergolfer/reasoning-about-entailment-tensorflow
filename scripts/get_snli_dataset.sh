SNLI_ZIP=snli_1.0.zip
SNLI_FOLDER=snli_1.0
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f $SNLI_ZIP ]; then
  echo Downloading the SNLI dataset from nlp.stanford.edu
  curl https://nlp.stanford.edu/projects/snli/snli_1.0.zip > ./$SNLI_ZIP
fi

if [ ! -d $SNLI_FOLDER ]; then
  echo Extracting the SNLI dataset .zip file
  unzip $SNLI_ZIP
fi

echo Done!
