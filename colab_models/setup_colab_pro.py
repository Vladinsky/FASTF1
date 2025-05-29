"""
Formula 1 Tire Change Prediction - Colab Pro Setup
Configurazione automatica ottimizzata per Google Colab Pro
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Stampa banner di benvenuto"""
    banner = """
    🏎️ ================================================================ 🏎️
    
         Formula 1 Tire Change Prediction - Colab Pro Edition
                     RNN Multi-task Learning System
    
    🏎️ ================================================================ 🏎️
    """
    print(banner)

def check_colab_environment():
    """Verifica che siamo in Google Colab"""
    try:
        import google.colab
        print("✅ Google Colab environment detected")
        return True
    except ImportError:
        print("❌ Not running in Google Colab!")
        print("This setup is optimized for Google Colab Pro")
        return False

def check_gpu_availability():
    """Verifica disponibilità GPU e tipo"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
            
            # Verifica se è GPU adatta per Pro
            if "T4" in gpu_name or "A100" in gpu_name or "V100" in gpu_name:
                print("🚀 Excellent GPU for training!")
            else:
                print("⚠️  GPU detected but may be slower for training")
        else:
            print("❌ No GPU detected!")
            print("Consider enabling GPU in Runtime > Change runtime type")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, will check GPU later")
    return True

def get_system_memory():
    """Ottiene informazioni sulla memoria di sistema"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1e9
        available_gb = memory.available / 1e9
        
        print(f"✅ System RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        
        if total_gb > 20:
            print("🚀 Excellent RAM for Colab Pro!")
        elif total_gb > 12:
            print("✅ Good RAM, should work well")
        else:
            print("⚠️  Limited RAM detected, consider Colab Pro")
        
        return total_gb
    except ImportError:
        print("⚠️  psutil not installed yet, will check memory later")
        return 0

def mount_google_drive():
    """Monta Google Drive"""
    try:
        from google.colab import drive
        print("🔗 Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Verifica che il mount sia avvenuto
        drive_path = "/content/drive/MyDrive"
        if os.path.exists(drive_path):
            print("✅ Google Drive mounted successfully")
            return drive_path
        else:
            print("❌ Google Drive mount failed")
            return None
            
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return None

def install_requirements():
    """Installa le dipendenze necessarie"""
    print("📦 Installing requirements...")
    
    # Installa requirements base
    req_file = Path("requirements_pro.txt")
    if req_file.exists():
        print("Installing from requirements_pro.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                      check=True, capture_output=True)
    else:
        print("Installing essential packages manually...")
        essential_packages = [
            "torch>=2.0.0",
            "pandas>=2.0.0", 
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "fastf1>=3.1.0",
            "pyarrow>=12.0.0",
            "tqdm>=4.65.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "ipywidgets>=8.0.0"
        ]
        
        for package in essential_packages:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
    
    print("✅ Requirements installed successfully")

def setup_colab_optimizations():
    """Configura ottimizzazioni specifiche per Colab"""
    print("⚙️ Setting up Colab optimizations...")
    
    # Configura matplotlib per Colab
    try:
        import matplotlib.pyplot as plt
        plt.style.use('default')
        print("✅ Matplotlib configured")
    except ImportError:
        pass
    
    # Configura pandas per output ottimizzato
    try:
        import pandas as pd
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)
        print("✅ Pandas display options configured")
    except ImportError:
        pass
    
    # Configura numpy per performance
    try:
        import numpy as np
        print("✅ NumPy configured")
    except ImportError:
        pass

def create_project_structure(drive_path):
    """Crea la struttura del progetto su Drive"""
    if not drive_path:
        print("⚠️  Cannot create project structure without Drive access")
        return None
        
    project_path = os.path.join(drive_path, "F1_TireChange_Project")
    
    # Crea cartelle principali
    folders = [
        "data/raw",
        "data/processed", 
        "data/unified",
        "models/checkpoints",
        "models/trained",
        "results/training_logs",
        "results/evaluations",
        "results/predictions"
    ]
    
    for folder in folders:
        folder_path = os.path.join(project_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"✅ Project structure created at: {project_path}")
    return project_path

def setup_environment_variables():
    """Configura variabili d'ambiente"""
    print("🔧 Setting up environment variables...")
    
    # Configura cache FastF1
    os.environ['FASTF1_CACHE'] = '/content/ff1_cache'
    
    # Crea directory cache
    os.makedirs('/content/ff1_cache', exist_ok=True)
    
    # Configura PyTorch per performance
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    print("✅ Environment variables configured")

def verify_installation():
    """Verifica che tutto sia installato correttamente"""
    print("🔍 Verifying installation...")
    
    essential_modules = {
        'torch': 'PyTorch',
        'pandas': 'Pandas', 
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'fastf1': 'FastF1',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM'
    }
    
    failed_imports = []
    
    for module, name in essential_modules.items():
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"⚠️  Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All essential modules verified")
        return True

def show_next_steps():
    """Mostra i prossimi step da seguire"""
    next_steps = """
    🎯 SETUP COMPLETED! Next Steps:
    
    1. 📊 Data Unification:
       from data.data_unifier_complete import CompleteDataUnifier
       unifier = CompleteDataUnifier()
       dataset = unifier.unify_all_data()
    
    2. 🏋️ Training:
       from training.train_from_scratch_pro import ProTrainer
       trainer = ProTrainer()
       model = trainer.train_complete()
    
    3. 📱 Quick Demo:
       %run notebooks/01_quick_start_pro.ipynb
    
    4. 🔍 Explore Data:
       %run notebooks/02_data_exploration.ipynb
    
    🏎️ Happy Racing! Check the README.md for detailed guide.
    """
    print(next_steps)

def main():
    """Funzione principale di setup"""
    start_time = time.time()
    
    print_banner()
    
    # Verifiche preliminari
    if not check_colab_environment():
        return False
    
    check_gpu_availability()
    get_system_memory()
    
    try:
        # Setup principale
        print("\n" + "="*60)
        print("STARTING COLAB PRO SETUP")
        print("="*60)
        
        # 1. Mount Drive
        drive_path = mount_google_drive()
        
        # 2. Install requirements  
        install_requirements()
        
        # 3. Setup ottimizzazioni
        setup_colab_optimizations()
        
        # 4. Crea struttura progetto
        project_path = create_project_structure(drive_path)
        
        # 5. Setup environment
        setup_environment_variables()
        
        # 6. Verifica installazione
        if not verify_installation():
            print("⚠️  Some modules failed to install properly")
            
        # 7. Final checks
        print("\n" + "="*60)
        print("FINAL SYSTEM CHECK")
        print("="*60)
        
        check_gpu_availability()
        get_system_memory()
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Setup completed in {elapsed_time:.1f} seconds")
        
        show_next_steps()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Setup completed successfully!")
    else:
        print("💥 Setup failed. Please check errors above.")
