import pandas as pd
import os

def convert_excel_to_csv():
    """
    Convert Excel sheets to CSV files for machine learning modeling
    """
    excel_file = "HSI egg data_Poultry farm exp_Master file.xlsx"
    
    print("Converting Excel file to CSV format...")
    
    # Create data directory
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load and convert reference/metadata sheet
    ref_df = pd.read_excel(excel_file, sheet_name='Ref. and Morpho. parameters')
    ref_df.to_csv('data/reference_metadata.csv', index=False)
    print(f"Saved reference_metadata.csv: {ref_df.shape}")
    
    # Convert spectral data sheets
    spectral_sheets = ['Spectra (DN Value)_D0', 'D_1', 'D_2', 'D_3', 'D_4']
    day_mapping = {'Spectra (DN Value)_D0': 'D0', 'D_1': 'D1', 'D_2': 'D2', 'D_3': 'D3', 'D_4': 'D4'}
    
    for sheet_name in spectral_sheets:
        try:
            spectral_df = pd.read_excel(excel_file, sheet_name=sheet_name)
            day = day_mapping[sheet_name]
            
            # Save to CSV
            csv_filename = f'data/spectral_data_{day}.csv'
            spectral_df.to_csv(csv_filename, index=False)
            print(f"Saved {csv_filename}: {spectral_df.shape}")
            
        except Exception as e:
            print(f"Error processing {sheet_name}: {e}")
    
    print("Excel to CSV conversion complete!")

def main():
    """
    Main conversion function
    """
    print("HSI Data Conversion")
    print("="*30)
    convert_excel_to_csv()

if __name__ == "__main__":
    main() 