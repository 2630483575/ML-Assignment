"""
Quick test script to verify dimensionality reduction module setup.
This script checks that all dependencies are installed correctly.
"""

import sys

def check_imports():
    """Check if all required packages are available."""
    print("Checking required packages...")
    
    required_packages = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'umap': 'UMAP (optional)',
    }
    
    missing = []
    optional_missing = []
    
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            if 'optional' in name.lower():
                optional_missing.append(name)
                print(f"⚠ {name} is NOT installed (optional)")
            else:
                missing.append(name)
                print(f"✗ {name} is NOT installed")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    if optional_missing:
        print(f"\n⚠ Optional packages not installed: {', '.join(optional_missing)}")
        print("Analysis will work but UMAP visualizations will be skipped.")
    
    print("\n✅ All required packages are installed!")
    return True


def check_module():
    """Check if our dimensionality reduction module can be imported."""
    print("\nChecking dimensionality reduction module...")
    
    try:
        sys.path.insert(0, '..')
        from utils import dimensionality_reduction
        print("✓ dimensionality_reduction module imported successfully")
        
        # Check key functions
        required_functions = [
            'perform_pca',
            'perform_tsne',
            'perform_kmeans',
            'plot_2d_embeddings',
            'plot_clustering_results',
        ]
        
        for func_name in required_functions:
            if hasattr(dimensionality_reduction, func_name):
                print(f"✓ {func_name} is available")
            else:
                print(f"✗ {func_name} is NOT available")
                return False
        
        print("\n✅ Dimensionality reduction module is ready!")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import module: {e}")
        return False


def main():
    print("="*60)
    print("Dimensionality Reduction Module Verification")
    print("="*60)
    print()
    
    step1 = check_imports()
    print()
    step2 = check_module() if step1 else False
    
    print()
    print("="*60)
    if step1 and step2:
        print("✅ VERIFICATION PASSED")
        print("You can now run: python main.py --mode analyze --model both")
    else:
        print("❌ VERIFICATION FAILED")
        print("Please install missing dependencies first.")
    print("="*60)


if __name__ == "__main__":
    main()
