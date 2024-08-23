#= Pietro FAVARO FPMs 08/03/2022
The goal of this file is to be able to modify the ex-post results by tunning the simulator,
based on the opitmization decisions which were generated before and are stored.

Simulator whose goal is to verify if 
    the decisions taken by the initial optimization problem 
    are feasible or not and propagate the error.

/!\ Because the head vectors contain the initial conditions, there index is shifted of one

                head_min    head_max    head_net
        p   Q                                       0
    0                                               1
    1                                               2
    2                                               3
=#

# This section of the code could be run only once
@time begin
    using LinearAlgebra
    using DataFrames
    using Plots
    using XLSX
    # using HDF5
    using Statistics


    ############# Functions from save_results file #############

    function get_index(absolute_x, x_min, x_max, shape, rounded = true)
        # here, x is the absolute value and y the index
        # y = m * x + p
        # index_h = (Δy/Δx) * h + p
        # m = (df.shape[1] - 1)/(h_max - h_min)
        # index_h = m * h + p
        # p is found by solving "for index_h = 1, h = h_min"
        # p = y - m * x, p = index_h - m * h, p = 1 - m * h_min
        # N.B: This example for h works also for p or any other index
        m = (shape .- 1)./(x_max .- x_min)
        p = 1 .- m .* x_min
        if rounded == true
            return round(Int, m .* absolute_x .+ p)
        else
            return m .* absolute_x .+ p
        end
    end

    function get_index_power(absolute_p, rounded = true)
        return get_index(absolute_p, 0, Pnominal[1], size(dataQturbine)[3], rounded)
    end

    function get_index_head(absolute_h, rounded = true)
        return get_index(absolute_h, head_min[1], head_max[1], size(dataQturbine)[2], rounded)
    end

    function get_absolute_value_from_index(index, x_min, x_max, shape)
        # this is the inverse function of the previous one "get_index"
        # in the previous one, we have y = m * x + p
        # here, we have x = (y - p)/m
        m = (shape .- 1)./(x_max .- x_min)
        p = 1 .- m .* x_min
        return (index .- p)./m
    end

    function get_absolute_power_from_index(index)
        return get_absolute_value_from_index(index, 0, Pnominal[1], size(dataQturbine)[3])
    end

    function get_absolute_head_from_index(index)
        return get_absolute_value_from_index(index, head_min[1], head_max[1], size(dataQturbine)[2])
    end

    function get_bounds(iAT, xT)
        if length(iAT) >= 1
            p_min = get_absolute_power_from_index(iAT[1])
            p_max = get_absolute_power_from_index(iAT[end])
            q_min = minimum(xT)
            q_max = maximum(xT)
        else
            p_min = 0
            p_max = 0
            q_min = 0
            q_max = 0
        end
        return p_min, p_max, q_min, q_max
    end

    function ceil_int(x)
        return convert(Int, ceil(x))
    end

    function floor_int(x)
        return convert(Int, floor(x))
    end

    ############ Fonction librabry ############
    function profile_rectangular_basin(volume, surface)
        return volume./surface;
    end

	function profile_lower_basin(v_target, bottom_surface)
		# Initial radius [m]
		r0 = sqrt.(bottom_surface/pi)
	
		# sine wave on the radius, r is a function of height, r = f(h)
		# r = r0 + 10*sin(k*h)
		function get_radius(h, r0)
			# Wavelength [m]
			lambda = 2
			# Angular wave number [1/m]
			k = 2 * pi / lambda
			# radius [m]
			r = r0 + 10*sin(k*h)
			return r
		end
	
		heads = zeros(length(v_target))
		for i in eachindex(v_target)
			r0_i = r0[i]
			v_target_i = v_target[i]
			# Initial height [m]
			h = heads[i]
			# height increment [m]
			Δh = 0.01
			# volume [m^3]
			v = 0
			# Numerical integration
			while v < v_target_i
				# get radius at current height
				r = get_radius(h, r0_i)
				# Get current section
				s = pi * r^2
				# Increment volume
				v += s * Δh
				# Increment height
				h += Δh
			end
			heads[i] = h
		end
	
		return heads
	end

    function reshape_data(data_vector, N_T_sim)
        length(data_vector[:, 1])
        Int(N_T_sim/length(data_vector))
        ones(Int(N_T_sim/length(data_vector)), 1)
        ones(Int(N_T_sim/length(data_vector)), 1) * data_vector
        reshape(ones(Int(N_T_sim/length(data_vector)), 1) * data_vector, (1, N_T_sim))
    end

    function get_abs_p_and_q(iAT, slice, q_init_positif)
        if length(iAT) >= 1
            xT = slice[iAT]
            # one want the new q (found in xT) to be the closest flow possible to q_init but smaller
            ΔQ = q_init_positif .- xT
            index_pos = [i for (i, x) in enumerate(ΔQ) if x >= 0]
            ΔQ_pos = ΔQ[index_pos]
#            println(q_init_positif, "\n", ΔQ, "\n", ΔQ_pos, "\n", index_pos, "\n", iAT)
            index_p = iAT[index_pos[argmin(ΔQ_pos)]]
            p_sim_pos = get_absolute_power_from_index(index_p)
            q_sim_pos = slice[index_p];
        else
            p_sim_pos = 0;
            q_sim_pos = 0;
        end
        return p_sim_pos, q_sim_pos
    end

    function error_on_v_to_penalty(Δv, penalty)
        return (Δv)*1000*9.81*75*0.937/(3.6e9)*penalty #((ΔV*ρ*g*h)[j]/3.6e9)[MWh] * 100[€/MWh]
    end

    function get_index_power_extended(abs_p)
        index_float = get_index_power(abs_p, false)
        index_int = get_index_power(abs_p, true)
        index_l =  floor_int(index_float)
        index_r = ceil_int(index_float)
        return index_float, index_int, index_l, index_r
    end

    function get_index_head_extended(abs_p)
        index_float = get_index_head(abs_p, false)
        index_int = get_index_head(abs_p, true)
        index_l =  floor_int(index_float)
        index_r = ceil_int(index_float)
        return index_float, index_int, index_l, index_r
    end

    function argmin_extended(v, target)
        #= 
            This function returns the index of the closest value to target in vector v
        =#
        index = argmin(abs.(v .- target))
        index_b = clamp(index - 1, 1, length(v))  # index before
        index_a = clamp(index + 1, 1, length(v))  # index after
        if v[index] < target
            if v[index_b] > target
                index_l = index_b
                index_r = index
            elseif v[index_a] > target
                index_l = index
                index_r = index_a
            else
                index_l = index
                index_r = index
            end
        else  # v[index] >= target
            if v[index_b] < target
                index_l = index_b
                index_r = index
            elseif v[index_a] < target
                index_l = index
                index_r = index_a
            else
                index_l = index
                index_r = index
            end
        end
        if index_l == index_r # == index
            index_float = index
        else
            index_float = index_l + (index_l - index_r)/(v[index_l] - v[index_r])*(target - v[index_l])
        end
        return index_float, index, index_l, index_r
    end

    function reserve_clamp(reserve_d, Rup)
        #=
            This function returns the clamped value if the reserve product and the extra commitment which
            cannot be provided because it is higher than the ramping capacity.
        =#
        reserve = clamp(reserve_d, 0, Rup)
        overRup = reserve_d - reserve
        return reserve, overRup
    end

    function round_down_to_multiple(x::Int, base::Int)
        #=
            This function returns the closest multiple of base which is smaller than x.
            Example: round_down_to_multiple(7, 5) = 5 because 5 is the closest multiple of 5 which is smaller than 7.
        =#
        multiple = div(x, base) * base
        return multiple
    end

####################################################################################################
####################################################################################################

	function evaluate_d(DAM_d, reserves_d, day_i, fill_ratio)
#		println("Start evaluation Julia: ", DAM_d, reserves_d)
		
		############# Main #############
		#=
		DAM_power is a horizontal vector containing the power bids on the day-ahead market (expected size 1x24)
		reserves is a horizontal vector containing the power bids on the reserve markets (expected size 1x6) with the following order:
			[upFCR, upaFRR, upmFRR, downFCR, downaFRR, downmFRR]
		=#

		# Define the DAM price profile
		price_DAM_scenarios = [
		56.65	59.61	61.79	70.1	80.32	80.02	69.63	62.57	63.21	64.68	61.84	63.92	70.5	72.8	66.79	66.44	64.9	50.39	43.44	39.95	40.03	41.49	45.47	48.78;  # 07/02/2017
		234.37	248.9	248.78	278.75	326.53	250.64	176.6	144.31	50.01	-32.91	-41.78	-76.37	3.79	100	167.5	174.47	168.84	163.81	139.59	131.36	127.38	131.14	126.64	133.6;  # 08/10/2022
		228.57	242.29	273.02	320	479.97	378.4	295.09	239.73	225.21	219.39	200.01	190.9	215	283.17	339.83	477.59	463.99	298.46	236.28	219.9	198.99	207.87	222.21	236.57;  # 12/10/2022
		121.71	128.89	134.26	157.23	177.37	206.79	177.68	105	145.94	115.35	87.37	94.9	95.92	100.7	119.39	149.07	134.81	108.19	34.05	22.69	67	25	20.06	50;   # 11/11/2022
		274.69	294.98	312	361.14	408.87	409.61	415.58	392.99	397.91	383.91	367.2	360.1	365.91	370	367.7	369.64	324.02	255.21	204.47	175.1	178.98	189.93	174.48	241;   # 28/11/2022
		303.46	324.56	333.11	395	473.87	474.35	444.98	430.73	429.91	413.46	392.96	404.91	443.08	426.86	436.9	468.1	413.61	344.63	273.9	314.43	249.94	267.07	275.53	294.51;  # 05/12/2022
		344.87	369.62	408.9	491.82	570.35	650.94	603.29	558.54	535.83	519.02	532.65	500	498.3	510.16	653.89	658.6	505	384.2	304.31	288.83	285.85	294.64	294.4	302.73;  # 12/12/2022
		342	351.72	396.67	487.9	520.57	557	590	545	528.53	514.95	494.7	503.85	568.01	565.18	569.72	539.99	494.97	368.82	313	287.47	285.25	301.08	299.95	325;  # 14/12/2022
		296.45	322.28	342.27	413.9	467.8	490.12	505	478.55	465.14	460.16	451.48	476.85	538.37	543.45	561.45	543.49	451.47	334.19	287.41	275.81	270.23	280.6	288.03	296.3;  # 16/12/2022
		263.79	266.24	272.43	313.9	363.96	335.42	335.42	305.13	283.5	266.95	257.29	270.07	301.04	302.13	299.81	291.25	259.45	241.66	238.67	245.08	251.2	267.29	271.82	290.24;  # 17/12/2022
		117.87	175.55	163.08	281.99	219.15	275.17	271.66	241.4	239.86	229.5	224.43	277.97	250.19	231.24	250.03	221.17	212.07	185.54	199.46	214.26	242.46	287.47	324.46	291;  # 18/12/2022
		0.13	2.39	5.22	10.55	22.09	27.93	26.91	14.88	16.7	3.98	1.07	1.45	0.96	0.4	0.91	0.76	0	0	-0.02	0.01	0.02	0.08	0.22	1.52;  # 31/12/2022
		35	45.96	36.57	45.83	54.95	36.64	46.03	36.54	23.5	0.85	-0.12	-0.57	-0.74	-0.23	-1.03	-1.57	-1.31	-4.69	-5.4	-4.41	-5.27	-1.46	-1.75	-4.39;  # 01/01/2023
		170.22	177.92	184.01	206.15	223.19	248.07	248.9	224.58	218.99	219.62	219.1	220.88	230.14	249.8	270.13	257.9	226.54	189.9	155.41	151.13	152.47	160.89	160.14	150.8;  # 23/01/2023
		175	193.06	188.88	218.08	231.15	250	250	224.58	217.98	215.28	205.99	217.85	229.96	238.65	256.43	251.1	210.06	183.61	169.96	153.64	155.8	164.1	160.06	164.85;  # 24/01/2023
		114.68	131.91	186.53	202.95	193.5	150.38	113.25	104.32	107.2	98.5	137.07	119.9	110	113.23	135.11	152.36	142.99	115	100.68	84.9	83.99	87.48	94.97	96;  # 27/03/2023
		134.85	147.07	164.63	179.56	169.09	137	110.08	98.15	94.15	88.39	90.01	93.24	97.18	104	115.53	132.49	132.43	113.1	95.79	92.14	89.58	88.49	93.57	99.85;  # 19/06/2023
		70.05	86.19	86.36	84.99	74.99	7.45	-7.86	-26.91	-67.59	-85.58	-84.07	-28.03	-14.62	-0.01	8.48	29.82	37	47.34	51.7	51.6	57.79	67.81	74.9	84.5;  # 30/07/2023
		127.17	138.5	144	134.16	107.02	85.1	84.03	74.55	47.66	-51.42	-21.52	55.9	-13.03	-8.55	85	104.47	96.66	72.22	65.03	14.4	7.22	36.81	49.49	54.51;  # 07/08/2023
		110.09	120.57	131.5	203.5	300	183.01	109.19	97.46	92.06	84.09	76.11	89.71	89.01	99.68	118.32	143	139.15	86.87	43.31	0.49	0	1.09	10	34.63;  # 25/09/2023
		101.07	106.57	113.4	133.01	140.83	135.52	112.52	96.49	90.7	77.04	72.44	82.05	87.37	92.07	106.82	130.05	126.27	91	51.48	26	65.35	71.1	93.7	108.5  # 29/09/2023
		]
		price_DAM = reverse(price_DAM_scenarios[day_i, :])' ; #forecasted price in the day-ahead market [€/MW]
		price_DAM = ones(Int(N_T/24), 1) * price_DAM;
		price_DAM = reshape(price_DAM, (1, N_T));

		# Define the reserve prices (scaled linearly with the DAM price relatively to the first day)
		price_coeff = 1 * mean(price_DAM) / mean(price_DAM_scenarios[1, :])
		price_availability_upFCR = 20*price_coeff; #capacity price for upward FCR
		price_availability_downFCR = 20*price_coeff; #capacity price for downward FCR
		price_availability_upaFRR = 15*price_coeff; #capacity price for upward aFRR
		price_availability_downaFRR = 15*price_coeff; #capacity price for downward aFRR
		price_availability_upmFRR = 10*price_coeff; #capacity price for upward mFRR
		price_availability_downmFRR = 10*price_coeff; #capacity price for downward mFRR
		price_res_up =
			[price_availability_upFCR, price_availability_upaFRR, price_availability_upmFRR];
		price_res_down =
			[price_availability_downFCR, price_availability_downaFRR, price_availability_downmFRR];

		# Define the initial volumes
		V_up_init = fill_ratio * V_max_upBasin; #initial conditions: water volume in the upper reservoir, per default: half the reservoir
    	V_low_init = (1 - fill_ratio) * V_max_lowBasin; #initial conditions: water volume in the lower reservoir, per default: half the reservoir
		# Define the target volume of water in the upper reservoir at the end of the time horizon
		V_target = dataPSH[5, 3:end]; #final conditions: target volume at the end of the day


		# This section of the code must be rerun at each simulation you perform

		# If reserve commitment are negative or greater than the ramping capacity, clip them
		reserves = zeros(Float64, 6)
		overRup = zeros(Float64, 6)  # Amount of reserve which cannot be provided because it is higher than the ramping capacity
		for i in eachindex(reserves)
			reserves[i], overRup[i] = reserve_clamp(reserves_d[i], Rup[i])
		end

#		println("Maximal reserve commiments (based on ramping capacities): ", Rup)
#		println("Reserve decisions: ", reserves_d)
#		println("Reserve overcommitments (wrt ramping capabilities): ", overRup)

		# Reserves
		res_up = reserves[1:3]; # your commitments to the upward reserve (three products)
		res_down = reserves[4:6];  # your commitments to the downward reserve (three products)
		Res_down = sum(res_down);
		Res_up = sum(res_up);
		Res_down 

		# Hypothesis: in case of several units, they all ensure the same part in the reserve
		# (if 3 units, each ensures 1/3 of the contracted reserve)
		Res_up_index = get_index_power(Res_up/N_PSH) - 1 # -1 is there because if there is no reserve, no index must be removed of the available power
		Res_down_index = get_index_power(Res_down/N_PSH) - 1 # similary, if the reserve = Prated, then the turbine can operate at P_rated (index = 1000) so index reserve = 999

		# Initialize the variables for the simulation
		v_up_sim = zeros(N_PSH, N_T_sim + 1);
		v_low_sim = zeros(N_PSH, N_T_sim + 1);
		h_up_sim = zeros(N_PSH, N_T_sim + 1);
		h_low_sim = zeros(N_PSH, N_T_sim + 1);
		h_net_sim = zeros(N_PSH, N_T_sim + 1);
		p_sim = zeros(N_PSH, N_T_sim);
		q_sim = zeros(N_PSH, N_T_sim);
		t = collect(0 : 1/N_ts_per_h_sim : 24-1/N_ts_per_h_sim);
		p_min = zeros((N_PSH, N_T_sim))
		p_max = zeros((N_PSH, N_T_sim))
		q_min = zeros((N_PSH, N_T_sim))
		q_max = zeros((N_PSH, N_T_sim))
		z_pump_sim = zeros((N_PSH, N_T_sim));
		z_turb_sim = zeros((N_PSH, N_T_sim));
		infeasible = Array{Bool, 2}(undef, (N_PSH, N_T_sim))
		infeasible .= false
		reserve_overcommitment = zeros((N_PSH, N_T_sim))
		
		# Compute DAM expected profits and reshape the data
		en_DAM = DAM_d; #Your 24 decisions on the day-ahead market (1 per hour)
		en_DAM_each_ts = reshape_data(en_DAM, N_T_sim);
		Profit_DAM = sum(en_DAM.*price_DAM); # Compute DAM expected profits based on decisions
		p_opti = reshape_data(en_DAM, N_T_sim);
		# p_sim = copy(p_opti);
		delta_ts_sim = 3600 / N_ts_per_h_sim;


		# Compute reserve volume bounds: those are not constants, they are the volume bounds to keep track of the reserve
		res_pump_upward = [p_opti[h, t] < 0 ? Res_up : 0 for t = 1:N_T_sim]
		res_pump_downward = [p_opti[h, t] < 0 ? Res_down : 0 for t = 1:N_T_sim]
		res_turb_upward = [p_opti[h, t] > 0 ? Res_up : 0 for t = 1:N_T_sim]
		res_turb_downward = [p_opti[h, t] > 0 ? Res_down : 0 for t = 1:N_T_sim]
		v_up_max = [V_max_upBasin[h] - (delta_ts_sim * 1e6 * sum(res_pump_downward[1:t])) / (efficiency_pump_mean[h] * rho * g * head_max[h])
		    for t = 1:N_T_sim]
		v_up_min = [V_min_upBasin[h] + (delta_ts_sim * 1e6 * sum(res_turb_upward[1:t])) / (efficiency_pump_mean[h] * rho * g * head_min[h])
		for t = 1:N_T_sim]
		v_low_max = [V_max_lowBasin[h] - (delta_ts_sim * 1e6 * sum(res_turb_upward[1:t])) / (efficiency_pump_mean[h] * rho * g * head_min[h])
		for t = 1:N_T_sim]
		v_low_min = [V_min_lowBasin[h] + (delta_ts_sim * 1e6 * sum(res_pump_downward[1:t])) / (efficiency_pump_mean[h] * rho * g * head_max[h])
		for t = 1:N_T_sim]
		# if the needed water volumes for the day are out of physical bounds, throw warning
		any(v_up_max .< V_min_upBasin) ? println("Warning: the upper basin does not contain enough water to uphold the reserve commitments.") : nothing
		any(v_up_min .> V_max_upBasin) ? println("Warning: the upper basin will overflow => unable to uphold reserve commitments.") : nothing
		any(v_low_max .< V_min_lowBasin) ? println("Warning: the lower basin does not contain enough water to uphold the reserve commitments.") : nothing
		any(v_low_min .> V_max_lowBasin) ? println("Warning: the lower basin will overflow => unable to uphold reserve commitments.") : nothing
			
			
		# Initialize volumes and heads
		for t = 1:1
		    v_up_sim[:, t] = V_up_init
		    v_low_sim[:, t] = V_low_init
		    h_up_sim[:, t] = profile_rectangular_basin(V_up_init, Surf_up);
		    h_low_sim[:, t] = profile_lower_basin(V_low_init, Surf_low);
		    h_net_sim[:, t] = head_ref .+ h_up_sim[:, t] .- h_low_sim[:, t] .- h_loss[:, t];
		end

#		println("Enter simulator.")

		for t = 1:N_T_sim
		    for h = 1:N_PSH
		        index_h_float, index_h, index_h_floor, index_h_ceil = get_index_head_extended(h_net_sim[h, t]) # = head from previous timestep
		        # Compute the p ideal to reach the DAM bid considering the past p_sim over the hour: (Total energy sold - Total energy produced)/time_left
		        energy_produced = sum(p_sim[h, round_down_to_multiple(t-1, ratio_ts_opti_sim) + 1:t]) / N_ts_per_h_sim  # ÷ is integer division
		        energy_sold = en_DAM[(t-1)÷ratio_ts_opti_sim+1]
		        time_left = N_ts_per_h_sim - (t-1) % N_ts_per_h_sim   # time left before the next hour
		        p_ideal = (energy_sold - energy_produced) * N_ts_per_h_sim / time_left # = power to reach the DAM bid
		        index_p_float, index_p, index_p_floor, index_p_ceil = get_index_power_extended(abs(p_ideal)) # = power at this timestep in the opti

		        # Should the machine stop because the DAM bid is reached and expose itself to penalties in the reserves or 
		        # keep running to satisfy the reserve commitments (but pay imbalance settlements) ?
		        # the question occurs iff p_ideal and en_DAM_each_ts have opposite signs
		        if sign(p_ideal) != sign(en_DAM_each_ts[t])
		            # assumption: the bounds of the previous timestep are the same as the current timestep
		            cost_ope = cost_op_PSH * p_min[h, t-1] / N_ts_per_h_sim  # = operational cost in €/MWh for running at p_min
		            imbalance_penalty = p_min[h, t-1] * 250 / N_ts_per_h_sim  # = imbalance penalty in €/MWh
		            reserve_penalty = sum(reserves) * 500 / N_ts_per_h_sim  # = reserve penalty in €/MW(h)
		            # What is the most costly ? Imbalance or reserve penalty ? If reserve, then keep running, else stop (which happens per default)
		            if imbalance_penalty < reserve_penalty - cost_ope[h]
		                p_ideal = p_min[h, t-1]
		            end
		        end
		        
		        # we are turbining which is possible iff we should turbine (decision to turbine and p_ideal > 0, i.e., we are not overshooting our power goal) 
		        # and we were not pumping at t-1 (p_sim[h, t-1] >= 0)
		        if en_DAM_each_ts[t] > 0 && p_ideal > 0 && (t > 1 ? (p_sim[h, t-1] >= 0) : true) 
		            # # Compute the available range of power based on the ramping capcities and the previous p_sim
		            # if t > 1 ? p_sim[h, t-1] > 0 : true
		            #     index_p_last_ts = get_index_power(abs(p_sim[h, t-1]))
		            #     index_lb = index_p_last_ts - Rdownts_index
		            #     index_rb = index_p_last_ts + Rupts_index
		            # else # p_sim[h, t-1] == 0
		            #     index_p_last_ts = get_index_power(abs(p_sim[h, t-1]))
		            #     index_lb = index_p_last_ts - Rupts_index
		            #     index_rb = index_p_last_ts + Rdownts_index
		            # end
		            # print("t = ", t, ", index_h = ", index_h, ", index_h_floor = ", index_h_floor, ", index_h_ceil = ", index_h_ceil, ", index_p = ", index_p, ", index_p_floor = ", index_p_floor, ", index_p_ceil = ", index_p_ceil, "\n")
		            slice_l = dataQturbine[h, index_h_floor, :]
		            slice_r = dataQturbine[h, index_h_ceil, :]
		            slice = (slice_l .+ slice_r) ./ 2
		            iAT = findall(!isnan, slice)
		            #=
		                The available range of power is between [1+Res_down_index, length(iAT)-Res_up_index]
		                however, if an overcommitment occurs, we have: 1+Res_down_index > length(iAT)-Res_up_index
		                which makes the slice impossible. Though, the slice remains the best option in which satisfy the DAM 
		                while minimizing the non-respect of the reserves.
		                Moreover, "1+Res_down_index" can be > length(iAT) or "length(iAT)-Res_up_index" < 1 which once again
		                makes the slice operation impossible because 'index out of range' so we must ensure the index range is
		                within [1, len(iAT)]
		            =#
		            iAT_res = iAT[max(1, min(1+Res_down_index, length(iAT)-Res_up_index)):min(length(iAT), max(1+Res_down_index, length(iAT)-Res_up_index))]

		            # we are turbining, we must make sure to respect the physical volume bounds (upper basin not empty and lower one not overflowing)
		            q_max_tmp = minimum([v_up_sim[h, t] v_low_max[t]] .- [v_up_min[t] v_low_sim[h, t]]) / delta_ts_sim;
		            iAT_res_volume = iAT_res[slice[iAT_res] .<= q_max_tmp]

		            # if it is impossible to find a point which matches the conditions on the reserve and on the volume,
		            # then only apply the conditions on the volume only
		            iAT_final = iAT_res_volume
		            if length(iAT_final) == 0 # not index fulfil the constraints
		                iAT_volume = iAT[slice[iAT] .<= q_max_tmp] # satisfy just the volume constraint
		                iAT_final = iAT_volume # if it is impossible to satisfy both the volume and reserve constraints, only the volume is satisfied
		                global infeasible[h, t] = true;
		                if length(iAT_final) == 0 # if it is still not possible, then set the machine to idle
		                    z_turb_sim[h,t], p_sim[h,t], q_sim[h, t] = (0, 0, 0)
		                    reserve_overcommitment[h, t] = Res_down_index + Res_up_index
		                end
		            end

		            #Find the closest power available (considering the reserves) to p_opti
		            if length(iAT_final) != 0
		                index_iAT_float, index_iAT, index_iAT_floor, index_iAT_ceil = argmin_extended(iAT_final .- index_p, 0)
		                index_p, index_p_floor, index_p_ceil = iAT_final[index_iAT], iAT_final[index_iAT_floor], iAT_final[index_iAT_ceil]
		                if index_p_floor != index_p_ceil
		                    index_p_float = index_p_floor + (index_p_ceil - index_p_floor) / (iAT_final[index_iAT_ceil] - iAT_final[index_iAT_floor]) * (index_iAT_float - index_iAT_floor)
		                    q_sim[h, t] = (slice[index_p_floor] + (slice[index_p_ceil] - slice[index_p_floor]) / (index_p_ceil - index_p_floor) * (index_p_float - index_p_floor));
		                else
		                    index_p_float = index_p_floor
		                    q_sim[h, t] = slice[index_p_floor]
		                end
		                p_sim[h, t] = get_absolute_power_from_index(index_p_float);                    
		                reserve_overcommitment[h, t] = sum([max(term, 0) for term = [((iAT[1]+Res_down_index) - index_p), (index_p - (iAT[end]-Res_up_index))]])
		            end
		        
		        # we are pumping which is possible iff we should pump (decision to turbine and p_ideal < 0, i.e., we are not overshooting our power goal)
		        # and we were not turbing at t-1 (p_sim[h, t-1] >= 0
		        elseif en_DAM_each_ts[t] < 0 && p_ideal < 0 && (t > 1 ? (p_sim[h, t-1] <= 0) : true)
		            slice_l = dataQpump[h, index_h_floor, :]
		            slice_r = dataQpump[h, index_h_ceil, :]
		            slice = (slice_l + slice_r) / 2
		            iAT = findall(!isnan, dataQpump[h, index_h, :])
		            iAT_res = iAT[max(1, min(1+Res_up_index, length(iAT)-Res_down_index)):min(length(iAT), max(1+Res_up_index, length(iAT)-Res_down_index))]

		            # we are pumping, we must make sure to respect the physical volume bounds (upper basin not overflowing and lower one not empty)
		            q_max_tmp = minimum([v_up_max[t] v_low_sim[h, t]] .- [v_up_sim[h, t] v_low_min[h]]) / delta_ts_sim;
		            iAT_res_volume = iAT_res[slice[iAT_res] .<= q_max_tmp]

		            # if it is impossible to find a point which matches the conditions on the reserve and on the volume,
		            # then only apply the conditions on the volume only
		            iAT_final = iAT_res_volume
		            if length(iAT_final) == 0 # not index fulfil the constraints
		                iAT_volume = iAT[slice[iAT] .<= q_max_tmp]
		                iAT_final = iAT_volume
		                global infeasible[h, t] = true;
		                if length(iAT_final) == 0 # if it is still not possible, then set the machine to idle
		                    z_pump_sim[h,t], p_sim[h,t], q_sim[h, t] = (0, 0, 0)
		                    reserve_overcommitment[h, t] = Res_down_index + Res_up_index
		                end
		            end

		            #Find the closest power available (considering the reserves) to p_opti
		            if length(iAT_final) != 0
		                index_iAT_float, index_iAT, index_iAT_floor, index_iAT_ceil = argmin_extended(iAT_final .- index_p, 0)
		                index_p, index_p_floor, index_p_ceil = iAT_final[index_iAT], iAT_final[index_iAT_floor], iAT_final[index_iAT_ceil]
		                if index_p_floor != index_p_ceil
		                    index_p_float = index_p_floor + (index_p_ceil - index_p_floor) / (iAT_final[index_iAT_ceil] - iAT_final[index_iAT_floor]) * (index_iAT_float - index_iAT_floor)
		                    q_sim[h, t] = - (slice[index_p_floor] + (slice[index_p_ceil] - slice[index_p_floor]) / (index_p_ceil - index_p_floor) * (index_p_float - index_p_floor));
		                else
		                    index_p_float = index_p_floor
		                    q_sim[h, t] = - slice[index_p_floor]
		                end
		                p_sim[h, t] = - get_absolute_power_from_index(index_p_float);
		                reserve_overcommitment[h, t] = sum([max(term, 0) for term = [((iAT[1]+Res_up_index) - index_p), (index_p - (iAT[end]-Res_down_index))]])
		            end
		        else # we are not pumping nor turbining
		            z_pump_sim[h,t], z_turb_sim[h,t], p_sim[h,t], q_sim[h, t] = (0, 0, 0, 0)
		            reserve_overcommitment[h, t] = Res_down_index + Res_up_index
		        end

		        # Based on the power adjustement we just made, we compute the volumes and the head
		        v_up_sim[h, t+1] = v_up_sim[h, t] - delta_ts_sim * q_sim[h, t];
		        v_low_sim[h, t+1] = v_low_sim[h, t] + delta_ts_sim * q_sim[h ,t];
		        h_low_sim[h, t+1] = profile_lower_basin(v_low_sim[h, t+1], Surf_low[h])[1];  # lower basin
		        h_up_sim[h, t+1] = profile_rectangular_basin(v_up_sim[h, t+1], Surf_up[h]);  # upper basin
		        h_net_sim[h, t+1] = head_ref[h] + h_up_sim[h, t+1] - h_low_sim[h, t+1]; #- h_loss[h, t+1];

		        ############ Getting the bounds for each timestep ############
		        if en_DAM_each_ts[t] > 0 && p_ideal > 0 && (t > 1 ? (p_sim[h, t-1] >= 0) : true)
		            xT = slice[iAT_res_volume]
		            p_min[h, t], p_max[h, t], q_min[h, t], q_max[h, t] = get_bounds(iAT_res_volume, xT)
		        elseif en_DAM_each_ts[t] < 0 && p_ideal < 0 && (t > 1 ? (p_sim[h, t-1] <= 0) : true)
		            xT = slice[iAT_res_volume]
		            p_min[h, t], p_max[h, t], q_min[h, t], q_max[h, t] = get_bounds(iAT_res_volume, xT)
		        else
		            p_min[h, t], p_max[h, t], q_min[h, t], q_max[h, t] = (0, 0, 0, 0)
		        end
		    end
		end


#		println("Simulator finished. Computing the penalties.")
		err_p = p_sim .- p_opti;
		# no penalty for not respecting the reserve volume
		# ΔV_max_up = v_up_sim .- V_max_upBasin
		# ΔV_min_up = v_up_sim .- V_min_upBasin
		# ΔV_max_low = v_low_sim .- V_max_lowBasin
		# ΔV_min_low =  v_low_sim .- V_min_lowBasin
		# err_v_up = [sign(ΔV_max_up[h, t]) == sign(ΔV_min_up[h, t]) ? minimum(abs, [ΔV_max_up[h, t], ΔV_min_up[h, t]]) : 0 for h=1:N_PSH for t = 1:N_T]
		# err_v_low = [sign(ΔV_max_low[h, t]) == sign(ΔV_min_low[h, t]) ? minimum(abs, [ΔV_max_low[h, t], ΔV_min_low[h, t]]) : 0 for h=1:N_PSH for t = 1:N_T]
		# cost_v_out_of_bound = error_on_v_to_penalty(sum([err_v_up err_v_low]), 250/(2*N_ts_per_h_sim))
		cost_v_out_of_bound = 0
		future_water_value = mean(price_DAM)

		
		en_DAM_actual = mean(reshape(p_sim, (N_ts_per_h_sim, 24)), dims=1)
		reserve_penalty_fee = 500 # [€/MWh] # even though the reserve is a power, we consider an energy here to be timestep independent
		v_out_of_bound_penalty_fee = 500 # [€/MWh]
		last_v_up_penalty_fee = 100 # [€/MWh]
#		println(price_DAM)
		settlement_penalty_fee = maximum(price_DAM) * 1.2 # [€/MWh]

		profit_reserve = sum(24 * price_res_up[r] * res_up[r] for r = 1:N_res_up) + sum(24 * price_res_down[r] * res_down[r] for r = 1:N_res_down)
		penalty_reserve = (get_absolute_power_from_index(sum(reserve_overcommitment)+1) / N_ts_per_h_sim + sum(overRup)*24) * reserve_penalty_fee
		# real actions - bids give the imbalance of the BRP (positive: generation surplus or consumption shortage or negative: generation shortage or load surplus)
		settlement_penalties = sum(abs.(en_DAM_actual - reshape(DAM_d, (1, 24)))) * settlement_penalty_fee;
		cost_v_out_of_bound = 0
		cost_err_last_v_up = V_target[1]-v_up_sim[end] > 0 ? error_on_v_to_penalty(V_target[1]-v_up_sim[end], last_v_up_penalty_fee) : error_on_v_to_penalty(V_target[1]-v_up_sim[end], future_water_value)

		cost_op = sum(reshape_data(cost_op_PSH, (N_T_sim)) .* abs.(p_sim)) / N_ts_per_h_sim
		ex_post_profit = Profit_DAM + profit_reserve - settlement_penalties - cost_op - cost_err_last_v_up - penalty_reserve - cost_v_out_of_bound
#		println(string("\n Ex-post profit: ", ex_post_profit))

		price_DAM_each_ts = reshape_data(price_DAM, N_T_sim);
		en_DAM_actual_each_ts = reshape_data(en_DAM_actual, N_T_sim)
		infeasible = infeasible .|| abs.(p_sim .- p_opti) .> 0.05
		percentage_infeasible = sum(infeasible)/(N_PSH*N_T_sim)*100

###################################################################################
###################################################################################
#		println("Printing the results of the simulator.")
#        println("Profits (DAM + Reserves): ", Profit_DAM, " + ", profit_reserve)
#		println("reserve penalty: ", penalty_reserve)
#		println("settlement penalties: ", settlement_penalties)
#		println("cost op: ", cost_op)
#		println("V target penalty: ", cost_err_last_v_up)
		

		function save()
#			println("Saving the results of the simulator.")
			############ Saving the results of the simulator ############
			# To be adapted to your needs
			XLSX.openxlsx("results_simulator_copy.xlsx", mode="w") do xf

				sheet = xf[1]
				XLSX.rename!(sheet, "overview")

				sheet["A1"] = ["Results of the simulator for every unit"]
				sheet["B4"] = ["p_sim est positif quand on turbine, négatif quand on pompe"]

				sheet["A7"] = ["Number of infeasible:", sum(infeasible)]
				sheet["A8"] = ["Percentage of infeasible:", round(percentage_infeasible, digits=1)]

				sheet["A11"] = ["Profit energy sold:", round(Profit_DAM, digits=1)]
				sheet["A12"] = ["Penalty on the reserve:", round(penalty_reserve, digits=1)]
				sheet["A13"] = ["Profit reserve:", round(profit_reserve, digits=1)]
				sheet["A14"] = ["Settlement penalties:", round(settlement_penalties, digits=1)]
				sheet["A15"] = ["Operational cost:", round(cost_op, digits=1)]
				sheet["A16"] = ["Volume out of bounds:", round(cost_v_out_of_bound, digits=1)]
				sheet["A17"] = ["Volume left cost:", round(cost_err_last_v_up, digits=1)]
				sheet["A18"] = ["Ex-post profit:", round(ex_post_profit, digits=1)]

				sheet["A20"] = ["Reserver Comm. over Ramp-up/down [MW]", round(sum(overRup), digits=2)]
				sheet["A21"] = ["V_target", V_target[1]]
				sheet["A22"] = ["Excess of water compared to V_target", round(v_up_sim[end] - V_target[1], digits=1)]

				for h = 1:N_PSH
					XLSX.addsheet!(xf, string("unit", h))
					sheet = xf[1+h]

					df = DataFrame();
					names(df) = ["T", "price_DAM", "en_DAM_bid", "en_DAM_actual", "p_sim", "q_sim", "err_p", "h_low_sim",
						"h_up_sim", "h_net_sim", "v_low_sim", "v_up_sim", "p_min", "p_max", "q_min", "q_max", "reserve_overcommitment",
						"v_low_min", "v_low_max", "v_up_min", "v_up_max"];
					df.T = round.(collect(0 : 1/N_ts_per_h_sim : 24-1/N_ts_per_h_sim), digits=2);
					df.price_DAM = round.(collect(price_DAM_each_ts[1:end]), digits=2);
					df.en_DAM = round.(collect(en_DAM_each_ts[1:end]), digits=2);
					df.en_DAM_actual_each_ts = round.(collect(en_DAM_actual_each_ts[1:end]), digits=2);
					df.p_sim = round.(collect(p_sim[1:end]), digits=2);
					df.q_sim = round.(collect(q_sim[1:end]), digits=2);
					df.err_p = round.(collect(err_p[1:end]), digits=2);
					df.h_low_sim = round.(collect(h_low_sim[2:end]), digits=2);
					df.h_up_sim = round.(collect(h_up_sim[2:end]), digits=2);
					df.h_net_sim = round.(collect(h_net_sim[2:end]), digits=2);
					df.v_low_sim = round.(collect(v_low_sim[2:end]), digits=2);
					df.v_up_sim = round.(collect(v_up_sim[2:end]), digits=2);
					df.p_min = p_min[h, :]
					df.p_max = p_max[h, :]        
					df.q_min = round.(q_min[h, :], digits=2)
					df.q_max = round.(q_max[h, :], digits=2)
					df.reserve_overcommitment_index = round.(get_absolute_power_from_index(reserve_overcommitment[h, :].+1), digits=2)
					df.v_low_min = round.(v_low_min, digits=2)
					df.v_low_max = round.(v_low_max, digits=2)       
					df.v_up_min = round.(v_up_min, digits=2)
					df.v_up_max = round.(v_up_max, digits=2)


					sheet["A1", dim=2] = names(df)
					for r in 1:size(df,1), c in 1:size(df,2)
						sheet[XLSX.CellRef(r+1 , c )] = df[r,c]
					end
				end
			end
		end
		save()
#		println("Finished")
		return ex_post_profit
	end

###########################################################################################
###########################################################################################
#    println("Begin")

	# Scenario definition
	day_i = 1  # I recommend day 1 as base scenario, then those 5 days: 2, 5, 7, 12, 15, then: 3, 4, 6, 8, 17
	fill_ratio = 0.5 # fill ratio of the upper reservoir at the beginning of the day

    ########### From PSH portfolio definition #################
    # This loads the parameters describing the PHES station
    # This loads the parameters describing the PHES station
#    cd(dirname(@__FILE__))  # set the working directory to the location of the script
    excel_data_turb = "./Problems/PHES/Parameters/portfolio_large_surface.xlsx"
    dataPSH = XLSX.readdata(excel_data_turb, "Feuil1", "A1:C24"); #import data
    #dataPSH = readxlsheet("spotmarket_data_2017.xlsx", "volumes_2");

    N_PSH = size(dataPSH[1, 3:end], 1); #number of PSH units

	V_min_upBasin = dataPSH[15, 3:end]; #minimum water capacity of the upper reservoir
    V_max_upBasin = dataPSH[16, 3:end]; #maximum water capacity of the upper reservoir
    V_min_lowBasin = dataPSH[17, 3:end]; #minimum water capacity of the lower reservoir
    V_max_lowBasin = dataPSH[18, 3:end]; #maximum water capacity of the lower reservoir

    cost_op_PSH = dataPSH[6, 3:end]; #operational costs in Euros/MW
    Pnominal = dataPSH[7, 3:end]; #nominal power in turbine and pump modes
    Rup_PSH = dataPSH[8, 3:end]; #upward ramping ability of the PSH in MW/min
    Rdown_PSH = dataPSH[9, 3:end]; #downward ramping ability of the PSH in MW/min
    efficiency_turb_mean = dataPSH[10, 3:end]; #mean efficiency of turbine
    efficiency_pump_mean = dataPSH[11, 3:end]; #mean efficiency of pump
    head_ref = dataPSH[12, 3:end]; #vertical drop
    head_min = dataPSH[13, 3:end]; #minimum head value
    head_max = dataPSH[14, 3:end]; #maximum head value
    Q_max_turb = dataPSH[19, 3:end]; #maximum water flows in turbine mode
    Q_max_pump = dataPSH[20, 3:end]; #maximum water flows in pump mode
    Surf_up = dataPSH[22, 3:end];
    Surf_low = dataPSH[24, 3:end];
    h_loss = zeros(N_PSH, 24);
    rho = 1000; #water density
    g = 9.81; #gravity acceleration [m/s^2]
    println("Parameters of the PHES station loaded.")

    # Define the market parameters
    N_T = 24  # number of time steps
    h=1 #number of PHES units
    test = false # if true, the simulator will load the decisions from the excel file


    const N_res_up = 3; #number of products for upward reserve
    const N_res_down = 3; #number of products for downward reserve


    # Ramp-up capacities: RupFCR, RupaFRR, RupmFRR, RdownFCR, RdownaFRR, RdownmFRR
    Rup = [Rup_PSH[h] / 2, Rup_PSH[h] * 7.5, Rup_PSH[h] * 15, Rdown_PSH[h] / 2, Rdown_PSH[h] * 7.5, Rdown_PSH[h] * 15];

    ############# Read turbine and pump UPC #############
    # Load the efficiency curves of the pump and turbine
    UPC_excel_shape = (50, 1001)
    dataQturbine = zeros(N_PSH, UPC_excel_shape[1], UPC_excel_shape[2]);
    dataQpump = zeros(N_PSH, UPC_excel_shape[1], UPC_excel_shape[2]);
    dataT = XLSX.readdata("./Problems/PHES/Parameters/UPCs/Francis_turbine_HNL.xlsx", "Sheet1", "B2:ALN51")
    dataQturbine[h, :, :] = map(dataT -> parse(Float64, dataT), dataT)
    println("UPC of the turbine loaded.")
    dataP = XLSX.readdata("./Problems/PHES/Parameters/UPCs/Francis_pump_HNL.xlsx", "Sheet1", "B2:ALN51")
    dataQpump[h, :, :] = map(dataP -> parse(Float64, dataP), dataP)
    println("UPC of the pump loaded.")

    ############# Define time granularity of the simulator #############
    # Time steps
    N_ts_per_h_sim = 60; #number of timesteps per hour for the simulator
    const N_T_sim = 24*N_ts_per_h_sim; #number of time steps over the horizon of interest defined as one day (see line 15)
    N_ts_per_h = 1; # n of time step per hour for the optimization
    ratio_ts_opti_sim = N_ts_per_h_sim / N_ts_per_h;
    if ratio_ts_opti_sim != round(N_ts_per_h_sim / N_ts_per_h)
        println("Error: the number of timesteps per hour for the simulator must be a multiple integer of the
            the number of timestep per hour of the optimisation")
        exit(1)
    end
    ratio_ts_opti_sim = Int(ratio_ts_opti_sim);

    # Ramping capacities over one timestep of the simulation
    Rupts = Rup_PSH[h] * 60 / N_ts_per_h_sim;  # Rup_PSH in MW/min
    Rdownts = Rdown_PSH[h] * 60 / N_ts_per_h_sim;
    Rupts_index = get_index_power(Rupts) - 1
    Rdownts_index = get_index_power(Rdownts) - 1
end

@time begin
#evaluate_d([0.11497971 0.41529686 0.21759709 0.0060377  0.19220141 0.39940187 0.11272485 0.96206478 0.20132967 0.40726275 0.32405785 0.15846195 0.26121813 0.31752596 0.60632278 0.31079815 0.62665245 0.50585744 0.00739138 0.17587195 0.60541617 0.7734503  0.99205183 0.46342034],[0.09645959 0.93098984 0.48662371 0.58446943 0.96879914 0.47399662])
#
#evaluate_d([-0.56072495  5.88045196  1.85607881  9.78210477  6.01364464 -1.29927863  3.56710791  5.39814371  8.21038276  6.62706672  6.03880596  0.4246079  6.20945831 -1.62394245  8.20134536  8.22690625  0.09385537  2.8859218  5.89489085 -8.73461543 -9.29364232  0.52464691  4.4158387   1.81647652],[7.73735658 2.23585435 3.7052296  5.50805313 3.09316019 4.12430887])
#evaluate_d([ 6.7407447   4.70881496 -3.3176185   7.43533805  6.81865861 -4.53219436  6.25837372  5.67147239  6.64401631  8.65939941 -0.94810259  9.09953174  1.88941811 -5.05249091  6.54124605  6.95259158  7.38377897  9.3238732 -9.78085611  6.38492456  3.38658743  9.76032395  6.71816738 -4.52600621],[8.94998269 7.72336382 7.27554664 0.05110406 8.84098527 3.86360631])

# 2 sparse NNs with 3 hidden layer of 5 neurons, day 1 (2017), fill ratio = 0.5
# DAM_d, reserves_d = ([-6.6	-6.83	-7.06	-7.44	-7.75	-8.06	-7.98	4.26	4.16	4.07	3.97	3.88	3.79	3.69	3.6	3.5	3.41	3.32	4.9	4.73	4.48	2.88	2.82	2.75],[1.29	0	0	0.17	0	0])

# 3x3 piecewise state-of-the-art, day 1 (2017), fill ratio = 0.5
#DAM_d, reserves_d = ([3.63 3.59 -5.2 -5.81 -6.41 -6.99 -7.58 4.45 4.22 3.99 3.75 3.5 3.26 3.03 2.81 -5.01 -5.62 3.31 3.08 2.85 2.64 -4.72 3.59 -4.9],[0.36 0.0 0.0 0.24 -0.0 0.0])
#DAM_d, reserves_d = ([-6.77 -6.86 3.38 -7.0 -7.31 -7.62 3.71 3.75 3.66 5.01 3.41 5.14 -6.68 -6.91 3.49 3.4 3.28 3.21 3.71 4.64 2.92 2.85 2.8 2.74],[1.27 0.0 0.0 0.14 0.0 0.0])

#p = evaluate_d(DAM_d, reserves_d, day_i, fill_ratio)
#println("Profit is: ", p)
#print(" ")
end
