import sys
import os
import subprocess

def main():
    # Ensure we are in the script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Path to app.py
    app_path = os.path.join(current_dir, "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found.")
        sys.exit(1)
        
    print(f"Starting {os.path.basename(current_dir)} via Shiny...")
    
    # Use sys.executable to ensure we use the same environment's python
    # Calling 'python -m shiny run app.py'
    cmd = [sys.executable, "-m", "shiny", "run", "app.py", "--reload"]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping app...")
    except subprocess.CalledProcessError as e:
        print(f"\nError running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
