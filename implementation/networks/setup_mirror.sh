source ../../run_in_environment.sh

JUPYTER_DIRECTORY="$(jupyter --config-dir)"
JUPYTER_CONFIG_FILE="$JUPYTER_DIRECTORY/jupyter_notebook_config.py"

# setup jupyter config
if [ ! -f $JUPYTER_CONFIG_FILE ]; then
    echo "settung up Jupyter config"
    jupyter notebook --generate-config
else
    echo "Jupyter config already exists. continuing..."
fi


# add save hook if required
if grep -q "Saving script " $JUPYTER_CONFIG_FILE
then
    echo "Jupyter save hook already added"
else
    echo "adding Jupyter save hook"
    cat "jupyter_save_hook.py" >> $JUPYTER_CONFIG_FILE
fi
