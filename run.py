import sys
import os

try:
    from streamlit.web import cli as stcli
except ImportError:
    from streamlit import cli as stcli

if __name__ == '__main__':
    # Make sure we're running from the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
