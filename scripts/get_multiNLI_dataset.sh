MULTI_NLI_ZIP=multinli_0.9.zip
MULTI_NLI_FOLDER=multinli_0.9
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f $SCRIPT_DIR/../data/$MULTI_NLI_ZIP ]; then
  echo Downloading the SNLI dataset from nlp.stanford.edu
  curl https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip > $SCRIPT_DIR/../data/$MULTI_NLI_ZIP
fi

if [ ! -d $SCRIPT_DIR/../data/$MULTI_NLI_FOLDER ]; then
  echo Extracting the SNLI dataset .zip file
  unzip $SCRIPT_DIR/../data/$MULTI_NLI_ZIP -d $SCRIPT_DIR/../data/$MULTI_NLI_FOLDER
fi

echo Done!
