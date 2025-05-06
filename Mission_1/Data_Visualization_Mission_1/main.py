import Data_preprocess as dp
import Data_visualization as dv

def main():
    df = dp.data_preprocess_filter_data(2010, 2020, 'AUSPRAEGUNG', 'insgesamt', 'filtered_data.csv')
    dv.data_preprocess_visualizaiton(df)

if __name__ == "__main__":
    main()