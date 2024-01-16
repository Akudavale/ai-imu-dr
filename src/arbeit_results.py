import matplotlib.pyplot as plt
import pickle

def results():
    """
    import saved loss per epoch and plot the graph (epoch,loss)
    """

    file_path = 'layer_3_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\arbeit results\\pickel files\\avg_loss_per_epoch\\avg_loss_results.pkl'
    with open(file_path, 'rb') as file:
        avg_loss_per_epoch = pickle.load(file)
    
    print(f'avg_loss per epochafter loded in plot file {avg_loss_per_epoch}')

    #save the results in results folder before ploting graph, shoul be saved while running testfilter
    def plot_avg_loss(avg_loss_per_epoch):
        plt.plot(range(1, len(avg_loss_per_epoch) + 1), avg_loss_per_epoch, marker='o')
        plt.title('Average Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Training Loss')
        plt.savefig("layer_3_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\arbeit results\\Figures\\'Average_Training_Loss_per_Epoch.png")
        plt.show()
        
    plot_avg_loss(avg_loss_per_epoch)

    """def plot_RMSE_layers():
        # List of file names containing results
        file_names = ['file1.pkl', 'file2.pkl', 'file3.pkl']

        # Initialize an empty list to store combined results
        combined_results = []

        # Read results from each file and append to the combined_results list
        for file_name in file_names:
            with open(file_name, 'rb') as f:
                results = pickle.load(f)
                combined_results.extend(results)

        plt.plot(range(1, len(combined_results) + 1), combined_results, marker='o')
        plt.title('Variation in RMSE as layers increase in number')
        plt.xlabel('Network layers')
        plt.ylabel('RMSE')
        plt.savefig("layer_3_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\results\\arbeit_results_fig\\avg_loss_results.png")
        plt.show()

        # Save the combined results to a new .pkl file
        with open('combined_results.pkl', 'wb') as f:
            pickle.dump(combined_results, f)
        
        plot_RMSE_layers()"""