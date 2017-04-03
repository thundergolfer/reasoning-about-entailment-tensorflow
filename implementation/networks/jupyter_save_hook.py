import io
import os
from notebook.utils import to_api_path

_script_exporter = None

def script_post_save(model, os_path, contents_manager, **kwargs):
    MIRROR_DIRECTORY = "code_mirrors"
    from nbconvert.exporters.script import ScriptExporter

    if model['type'] != 'notebook':
        return
    global _script_exporter

    if _script_exporter is None:
        _script_exporter = ScriptExporter(parent=contents_manager)
    log = contents_manager.log

    base, ext = os.path.splitext(os_path)
    index_filename_start = base.rfind('/') + 1
    base = base[:index_filename_start] + 'code_mirrors/' + base[index_filename_start] # Save to subdirectory
    py_fname = base + '.py'
    script, resources = _script_exporter.from_filename(os_path)
    script_fname = base + resources.get('output_extension', '.txt')
    log.info("Saving script /%s", to_api_path(script_fname, contents_manager.root_dir))
    with io.open(os.path.join(MIRROR_DIRECTORY, script_fname), 'w', encoding='utf-8') as f:
        for line in script.split('\n'):
            if "# In" in line:
                continue
            f.write(line + '\n')

c.FileContentsManager.post_save_hook = script_post_save
