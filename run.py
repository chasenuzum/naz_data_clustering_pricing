import subprocess

def run_python_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

def main():
    # Run EDA script
    print("Running EDA script...")
    run_python_script("run_files/run_eda.py")

    # Run modeling script
    print("Running modeling script...")
    run_python_script("run_files/run_modeling_v3.py")

if __name__ == "__main__":
    main()