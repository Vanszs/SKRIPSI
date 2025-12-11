
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load data
    df = pd.read_csv(r'd:\SKRIPSI\gas-ml\data\blocks.csv')
    
    # Check first few values
    print("First 10 values:")
    print(df['baseFeePerGas'].head(10))
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(df['baseFeePerGas'].iloc[:500], label='Actual BaseFee')
    plt.title('BaseFee Raw Data Check')
    plt.xlabel('Block')
    plt.ylabel('BaseFee')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(r'd:\SKRIPSI\gas-ml\repro_plot.png')
    print("Plot saved to repro_plot.png")

if __name__ == "__main__":
    main()
