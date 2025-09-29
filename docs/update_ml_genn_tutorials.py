import os
import re
import sys
import pygenn

# **HACK** add PyGeNN docs to path
pygenn_docs_path = os.path.join(os.path.dirname(pygenn.__file__), os.pardir, "docs")
sys.path.append(pygenn_docs_path)

# Import process function
from update_tutorials import process_notebooks

# Comile regular expressions used to find GeNN installation code
_rm_regex = re.compile(r"!rm -rf /content/ml_genn-ml_genn_([_0-9a-zA-Z]+)")
_wget_regex = re.compile(r"!wget https://github.com/genn-team/ml_genn/archive/refs/tags/ml_genn_([_0-9a-zA-Z]+).zip")
_unzip_regex = re.compile(r"!unzip -q ml_genn_([_0-9a-zA-Z]+).zip")
_pip_regex = re.compile(r"!pip install ./ml_genn-ml_genn_([_0-9a-zA-Z]+)/ml_genn")


def update_ml_genn(source, ml_genn_tag):
    # Search for matches
    rm_match = _rm_regex.search(source)
    wget_match = _wget_regex.search(source)
    unzip_match = _unzip_regex.search(source)
    pip_match = _pip_regex.search(source)
    
    if rm_match and wget_match and unzip_match and pip_match:
        print(f"\tOld mlGeNN tag {rm_match[1]}")
        
        # Check all tags match
        assert rm_match[1] == wget_match[1]
        assert rm_match[1] == unzip_match[1]
        assert rm_match[1] == pip_match[1]
        
        # Update all lines with new tag
        source = _rm_regex.sub(f"!rm -rf /content/ml_genn-ml_genn_{ml_genn_tag}", source)
        source = _wget_regex.sub(f"!wget https://github.com/genn-team/ml_genn/archive/refs/tags/ml_genn_{ml_genn_tag}.zip", source)
        source = _unzip_regex.sub(f"!unzip -q ml_genn_{ml_genn_tag}.zip", source)
        source = _pip_regex.sub(f"!pip install ./ml_genn-ml_genn_{ml_genn_tag}/ml_genn", source)
        return source
    else:
        raise RuntimeError("mlGeNN installation not found in cell")

if __name__ == "__main__":
    assert len(sys.argv) > 5

    # Extract fixed arguments from end of list after potential wildcard expansion
    gdown_hash = sys.argv[-4]
    pygenn_ver = sys.argv[-3]
    python_tag = sys.argv[-2]
    ml_genn_tag = sys.argv[-1]
    
    notebooks = sys.argv[1:-4]
    
    # Process notebooks with additional callback to handle ml_genn_tag
    process_notebooks(notebooks, gdown_hash, pygenn_ver, python_tag, 
                      lambda source: update_ml_genn(source, ml_genn_tag))
