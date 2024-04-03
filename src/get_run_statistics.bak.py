def get_run_statistics_old(self) -> None:
        # Extract target values from the input file
        target_correlation_time = self.correlation_time
        target_rms_acceleration = self.acceleration_field_rms
        target_solenoidal_weight = self.solenoidal_weight

        # Calculate the forcing correlation time, method 1
        correlation_time = self.get_run_integral_times()
        correlation_time = correlation_time[0, :, :]
        correlation_time = correlation_time[correlation_time != 0]
        correlation_time_mean = np.mean(correlation_time)
        correlation_time_std = np.std(correlation_time)

        # Calculate the forcing correlation time, method 2
        corr_time_target = self.correlation_time
        corr_time_actuals = []
        all_data = self.get_run_integral_times()
        for this_data in all_data:
            num_points = this_data.shape[1]
            for i in range(num_points):
                this_slice = this_data[i, :]
                idx_0 = np.argwhere(np.array(this_slice) < 0)
                if len(idx_0) == 0:
                    continue
                corr_time_actuals.append(np.trapz(this_slice[:idx_0[0][0]], dx=self.code_time_between_dumps))

        # Calculate the RMS acceleration
        rms_acceleration = self.get_run_average_fields(['acc_0', 'acc_1', 'acc_2'])
        rms_acceleration = rms_acceleration.to_numpy()
        rms_acceleration = rms_acceleration[:, 1:]
        rms_acceleration = rms_acceleration[rms_acceleration != 0]
        rms_acceleration_mean = np.mean(rms_acceleration)
        rms_acceleration_std = np.std(rms_acceleration)

        # Calculate the relative power of solenoidal modes, method 1
        solenoidal_weight = self.get_run_average_fields(['solenoidal_weight'])
        solenoidal_weight = solenoidal_weight.to_numpy()
        solenoidal_weight = solenoidal_weight[:, 1:]
        solenoidal_weight = solenoidal_weight[solenoidal_weight != 0]
        solenoidal_weight_mean = np.mean(solenoidal_weight)
        solenoidal_weight_std = np.std(solenoidal_weight)

        # Calculate the relative power of solenoidal modes, method 2
        def get_mean_squared_ratio(field1, field2):
            ds = yt.load(self.outdir + '/parthenon.prim.*.phdf')
            ad = ds.all_data()
            field1 = ad.quantities.weighted_average_quantity(field1, ('index', 'volume'))
            field2 = ad.quantities.weighted_average_quantity(field2, ('index', 'volume'))
            return field1 / field2

        id_split = self.outdir.split('/')[-1].split('-')
        ζ = float(id_split[2].split('_')[1])
        sol_weight_actual = get_mean_squared_ratio('a_s_mag' + '/moments/' + 'rms', 'a' + '/moments/' + 'rms')
        sol_weight = 1.0 - ((1 - ζ) ** 2 / (1 - 2 * ζ + 3 * ζ ** 2))

        # Print the statistics
        print(f"> Forcing correlation time:")
        print(f"    Method 1: {correlation_time_mean:.2f} +/- {correlation_time_std:.2f} (target: {target_correlation_time:.2f})")
        print(f"    Method 2: {np.mean(corr_time_actuals):.2f} +/- {np.std(corr_time_actuals):.2f} (target: {target_correlation_time:.2f})")
        print(f"> RMS acceleration: {rms_acceleration_mean:.2f} +/- {rms_acceleration_std:.2f} (target: {target_rms_acceleration:.2f})")
        print(f"> Relative power of solenoidal modes:")
        print(f"    Method 1: {solenoidal_weight_mean:.2f} +/- {solenoidal_weight_std:.2f} (target: {target_solenoidal_weight:.2f})")
        print(f"    Method 2: {sol_weight_actual[0]:.2f} +/- {sol_weight_actual[1]:.2f} (target: {sol_weight:.2f})")