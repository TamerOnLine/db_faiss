import os

def get_scripts():
    """Retrieve all Python scripts in the current directory."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        str(index): os.path.join(scripts_dir, file)
        for index, file in enumerate(os.listdir(scripts_dir))
        if file.endswith('.py') and file != os.path.basename(__file__)  # Exclude the current script
    }

def run_script():
    """Display available scripts and execute the selected script."""
    scripts = get_scripts()

    if not scripts:
        print("\n❌ No Python scripts to run in this directory.")
        return

    print("\n📜 Select the script you want to run:\n")
    for key, script in scripts.items():
        print(f"{key}. {script}")

    choice = input("\n🔹 Enter the number of the script to execute: ").strip()

    if choice in scripts:
        script_to_run = scripts[choice]
        print(f"\n🚀 Running {script_to_run}...\n")
        try:
            os.system(f"python {script_to_run}")
        except Exception as e:
            print(f"\n❌ Error running the script: {e}")
    else:
        print("\n❌ Invalid choice. Please select a valid number.")

if __name__ == "__main__":
    run_script()
