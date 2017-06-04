URL=http://clic.cimec.unitn.it/composes/materials/SICK.zip

SICK_ZIP=SICK.zip
SICK_FOLDER=SICK
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f $SCRIPT_DIR/../data/$SICK_ZIP ]; then
  echo Downloading the SICK dataset from clic.cimec.unitn.it
  curl $URL > $SCRIPT_DIR/../data/$SICK_ZIP
fi

if [ ! -d $SICK_FOLDER ]; then
  echo Extracting the SICK dataset .zip file
  unzip $SCRIPT_DIR/../data/$SICK_ZIP -d $SCRIPT_DIR/../data/$SICK_FOLDER
fi

echo Done!
