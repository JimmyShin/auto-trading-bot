import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import main

if __name__ == "__main__":
    print("Calling log_exit test...")
    main.log_exit('TEST/USDT','long',123.45,0.78,'unit_test')
    print("OK")
