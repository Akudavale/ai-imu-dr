import matplotlib
import os
from termcolor import cprint
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from utils import *
from utils_torch_filter import TORCHIEKF
import pickle

def results_filter(args, dataset):
    RMSE_xy = []
    RMSE_z = []
    Trajectory_rmse_dict = {}
    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        file_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            continue

        print("\nResults for: " + dataset_name)

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = dataset.get_estimates(
            dataset_name)

        # get data
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        # get data for nets
        u_normalized = dataset.normalize(u).numpy()
        # shift for better viewing
        u_normalized[:, [0, 3]] += 5
        u_normalized[:, [2, 5]] -= 5

        t = (t - t[0]).numpy()
        u = u.cpu().numpy()
        ang_gt = ang_gt.cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ang = np.zeros((Rot.shape[0], 3))
        Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 0] = pitch.numpy()
            ang[j, 0] = yaw.numpy()
        # unwrap
            Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
                                        torch.Tensor([ang_gt[j, 1]]),
                                        torch.Tensor([ang_gt[j, 2]]))
            roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
            ang_gt[j, 0] = roll.numpy()
            ang_gt[j, 0] = pitch.numpy()
            ang_gt[j, 0] = yaw.numpy()

        Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align)
        v_norm = np.sqrt(np.sum(v_gt ** 2, 1))
        v_norm /= np.max(v_norm)

        #print(f"predicted: {p}, predicted shape {p.shape}") #for debug
        #print(f"GT: {p_gt}, GT shape {p_gt.shape}") #for debug

        # Compute various errors
        error_p = np.abs(p_gt - p)
        # MATE
        mate_xy = np.mean(error_p[:, :2], 1)
        mate_z = error_p[:, 2]

        # CATE
        cate_xy = np.cumsum(mate_xy)
        cate_z = np.cumsum(mate_z)

        # RMSE
        rmse_xy = 1 / 2 * np.sqrt(error_p[:, 0] ** 2 + error_p[:, 1] ** 2)
        rmse_z = error_p[:, 2]
        RMSE_xy.append(rmse_xy)
        RMSE_z.append(rmse_z)
        
        #print(f"rmse xy :{rmse_xy}, shape: {rmse_xy.shape}")
        # to calculate RMSE value for each trajectory
        avg_RMSE_xy_trajectory = np.mean(rmse_xy)
        avg_RMSE_z_trajectory = np.mean(rmse_z)
        Trajectory = str(dataset_name) 
        Trajectory = Trajectory.rstrip("_extract")
        Trajectory_rmse_dict[Trajectory] = {'RMSE_xy': avg_RMSE_xy_trajectory, 'RMSE_z': avg_RMSE_z_trajectory}
        
        #R_Squared loss or Coefficient of Determination:
        """ provides a measure of how well the predicted states explain the variability in the actual states.
            It ranges between 0 and 1, with higher values indicating a better fit.
        """
        def calculate_r_squared(observed, predicted):
            mean_observed = np.mean(observed)
            tss = np.sum((observed - mean_observed)**2)
            sse = np.sum((predicted - observed)**2)
            r_squared = 1 - (sse / tss)
            r_squared = max(0, min(1, r_squared))  # Ensure R-squared is within [0, 1]
            return r_squared
        
        r2_position_x = calculate_r_squared(p_gt[:, 0], p[:, 0])
        r2_position_y = calculate_r_squared(p_gt[:, 1], p[:, 1])
        r2_position_z = calculate_r_squared(p_gt[:, 2], p[:, 2])

        # Calculate R-squared for orientation 
        r2_roll = calculate_r_squared(ang_gt[:, 0], ang[:, 0])
        r2_pitch = calculate_r_squared(ang_gt[:, 1], ang[:, 1])
        r2_yaw = calculate_r_squared(ang_gt[:, 2], ang[:, 2])
  
        RotT = torch.from_numpy(Rot).float().transpose(-1, -2)

        v_r = (RotT.matmul(torch.from_numpy(v).float().unsqueeze(-1)).squeeze()).numpy()
        v_r_gt = (Rot_gt.transpose(-1, -2).matmul(
            torch.from_numpy(v_gt).float().unsqueeze(-1)).squeeze()).numpy()

        p_r = (RotT.matmul(torch.from_numpy(p).float().unsqueeze(-1)).squeeze()).numpy()
        p_bis = (Rot_gt.matmul(torch.from_numpy(p_r).float().unsqueeze(-1)).squeeze()).numpy()
        error_p = p_gt - p_bis

        # plot and save plot
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)

        # position, velocity and velocity in body frame
        fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # orientation, bias gyro and bias accelerometer
        fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # position in plan
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        # position in plan after alignment
        fig4, ax4 = plt.subplots(figsize=(20, 10))
        #  Measurement covariance in log scale and normalized inputs
        fig5, axs5 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # input: gyro, accelerometer
        fig6, axs6 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
        # errors: MATE, CATE  RMSE
        fig7, axs7 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        fig8, axs8 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        fig9, axs9 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        
        axs1[0].plot(t, p_gt)
        axs1[0].plot(t, p)
        axs1[1].plot(t, v_gt)
        axs1[1].plot(t, v)
        axs1[2].plot(t, v_r_gt)
        axs1[2].plot(t, v_r)

        axs2[0].plot(t, ang_gt)
        axs2[0].plot(t, ang)
        axs2[1].plot(t, b_omega)
        axs2[2].plot(t, b_acc)

        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        ax3.plot(p[:, 0], p[:, 1])
        ax3.axis('equal')
        ax4.plot(p_gt[:, 0], p_gt[:, 1])
        ax4.plot(p_align[:, 0], p_align[:, 1])
        ax4.axis('equal')

        axs5[0].plot(t, np.log10(measurements_covs))
        axs5[1].plot(t, u_normalized[:, :3])
        axs5[2].plot(t, u_normalized[:, 3:])

        axs6[0].plot(t, u[:, :3])
        axs6[1].plot(t, u[:, 3:6])

        axs7[0].plot(t, mate_xy)
        axs7[0].plot(t, mate_z)
        axs7[0].plot(t, rmse_xy)
        axs7[0].plot(t, rmse_z)
        axs7[1].plot(t, cate_xy)
        axs7[1].plot(t, cate_z)
        axs7[2].plot(t, error_p)

        axs8[0].plot(t, [r2_position_x] * len(t), label='R-squared Position X')
        axs8[1].plot(t, [r2_position_y] * len(t), label='R-squared Position Y')
        axs8[2].plot(t, [r2_position_z] * len(t), label='R-squared Position Z')

        axs9[0].plot(t, [r2_roll] * len(t), label='R-squared Roll')
        axs9[1].plot(t, [r2_pitch] * len(t), label='R-squared Pitch')
        axs9[2].plot(t, [r2_yaw] * len(t), label='R-squared Yaw')

        axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
        axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
        axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
                    title="Velocity in body frame")
        axs2[0].set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                    title="Orientation")
        axs2[1].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{\omega}}$ (rad/s)',
                    title="Bias gyro")
        axs2[2].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{a}}$ (m/$\mathrm{s}^2$)',
                    title="Bias accelerometer")
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
        axs5[0].set(xlabel='time (s)', ylabel=r' $\mathrm{cov}(\mathbf{y}_{n})$ (log scale)',
                     title="Covariance on the zero lateral and vertical velocity measurements (log "
                           "scale)")
        axs5[1].set(xlabel='time (s)', ylabel=r'Normalized gyro measurements',
                     title="Normalized gyro measurements")
        axs5[2].set(xlabel='time (s)', ylabel=r'Normalized accelerometer measurements',
                   title="Normalized accelerometer measurements")
        axs6[0].set(xlabel='time (s)', ylabel=r'$\omega^x_n, \omega^y_n, \omega^z_n$ (rad/s)',
                    title="Gyrometer")
        axs6[1].set(xlabel='time (s)', ylabel=r'$a^x_n, a^y_n, a^z_n$ (m/$\mathrm{s}^2$)',
                    title="Accelerometer")
        axs7[0].set(xlabel='time (s)', ylabel=r'$|| \mathbf{p}_{n}-\hat{\mathbf{p}}_{n} ||$ (m)',
                    title="Mean Absolute Trajectory Error (MATE) and Root Mean Square Error (RMSE)")
        axs7[1].set(xlabel='time (s)',
                    ylabel=r'$\Sigma_{i=0}^{n} || \mathbf{p}_{i}-\hat{\mathbf{p}}_{i} ||$ (m)',
                    title="Cumulative Absolute Trajectory Error (CATE)")
        axs7[2].set(xlabel='time (s)', ylabel=r' $\mathbf{\xi}_{n}^{\mathbf{p}}$',
                    title="$SE(3)$ error on position")
        
        axs8[0].set(xlabel='time (s)', ylabel='R-squared Position X', title="R-squared Values for Position X")
        axs8[1].set(xlabel='time (s)', ylabel='R-squared position Y', title="R-squared Values for Position Y")
        axs8[2].set(xlabel='time (s)', ylabel='R-squared position Z', title="R-squared Values for Position Z")

        axs9[0].set(xlabel='time (s)', ylabel='R-squared Roll', title="R-squared Values for Roll")
        axs9[1].set(xlabel='time (s)', ylabel='R-squared Pitch', title="R-squared Values for Pitch")
        axs9[2].set(xlabel='time (s)', ylabel='R-squared Yaw', title="R-squared Values for Yaw")


        for ax in chain(axs1, axs2, axs5, axs6, axs7,axs8,axs9):
            ax.grid()

        ax3.grid()
        ax4.grid()

        axs1[0].legend(
            ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
        axs1[1].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs1[2].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs2[0].legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',
                        r'$\hat{\theta}_n^y$', r'$\hat{\psi}_n^z$'])
        axs2[1].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$'])
        axs2[2].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$'])
        ax3.legend(['ground-truth trajectory', 'proposed'])
        ax4.legend(['ground-truth trajectory', 'proposed'])
        axs5[0].legend(['zero lateral velocity', 'zero vertical velocity'])
        axs6[0].legend(['$\omega_n^x$', '$\omega_n^y$', '$\omega_n^z$'])
        axs6[1].legend(['$a_n^x$', '$a_n^y$', '$a_n^z$'])
        if u.shape[1] > 6:
            axs6[2].legend(['$m_n^x$', '$m_n^y$', '$m_n^z$'])
        axs7[0].legend(['MATE xy', 'MATE z', 'RMSE xy', 'RMSE z'])
        axs7[1].legend(['CATE xy', 'CATE z'])

        axs8[0].legend()
        axs8[1].legend()
        axs8[2].legend()

        axs9[0].legend()
        axs9[1].legend()
        axs9[2].legend()

        # save figures
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8,fig9]
        figs_name = ["position_velocity", "orientation_bias", "position_xy", "position_xy_aligned",
                     "measurements_covs", "imu", "errors", "errors2", "R-squared loss position", "R-squared loss Orientation"]
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(folder_path, fig_name + ".png"))

        plt.show(block=True)

    # 1)Rmse /later for each trajectory
    # save avg rmse calculated for each trajectory as dict
    Trajectory_RMSE_path = "base_layer_2_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\arbeit results\\pickel files\\Trajectory_Rmse\\Trajectory_Rmse.pkl"
    with open(Trajectory_RMSE_path, 'wb') as f:
        pickle.dump(Trajectory_rmse_dict, f)

    #print(Trajectory_rmse_dict) #debug code

    """
    Below are arbeit results
    """
    # 2) avg rmse
    avg_RMSE_xy_data = np.mean( np.concatenate(RMSE_xy).ravel())
    avg_RMSE_z_data = np.mean(np.concatenate(RMSE_z).ravel())

    # Define folder paths
    base_folder = 'layer_3_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\arbeit results\\pickel files\\avg_RMSE_of_data'
    RMSExy_folder = os.path.join(base_folder, 'RMSE_XY_data.pkl')
    RMSEz_folder = os.path.join(base_folder, 'RMSE_Z_data.pkl')

    # Save average RMSE values to pickle files
    with open(RMSExy_folder, 'wb') as f:
        pickle.dump(avg_RMSE_xy_data, f)

    with open(RMSEz_folder, 'wb') as f:
        pickle.dump(avg_RMSE_z_data, f)

    #print(f"avg rmse of data in xy :{avg_RMSE_xy_data} and in z :{avg_RMSE_z_data}") #debug
    
    #3)import saved loss per epoch and plot the graph (epoch,loss)
        
    file_path = 'base_layer_2_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\arbeit results\\pickel files\\avg_loss_per_epoch\\avg_loss_results.pkl'
    with open(file_path, 'rb') as file:
        avg_loss_per_epoch = pickle.load(file)
    
    #print(f'avg_loss per epochafter loded in plot file {avg_loss_per_epoch}') #debug

    #save the results in results folder before ploting graph, shoul be saved while running testfilter
    def plot_avg_loss(avg_loss_per_epoch):
        plt.plot(range(1, len(avg_loss_per_epoch) + 1), avg_loss_per_epoch, marker='o')
        plt.title('Average Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Training Loss')
        plt.savefig("base_layer_2_AI-IMU_Dead-Reckoning\\ai-imu-dr-master\\arbeit results\\Figures\\Average_Training_Loss_per_Epoch.png")
        plt.show()
        
    plot_avg_loss(avg_loss_per_epoch)

