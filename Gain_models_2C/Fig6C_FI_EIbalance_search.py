import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
import multiprocessing
import itertools

# Function to compute rates and burst statistics
def compute_rates(spike_monitor, ca_monitor, duration, isi_threshold):
    """
    Computes firing rates and burst statistics.
    A burst is now defined as a sequence of spikes with ISI < isi_threshold,
    AND where a dendritic calcium spike (Ca_spike == 1) occurs during that sequence.
    """
    spike_trains = spike_monitor.spike_trains()
    # Get calcium spike data from the monitor
    ca_times = ca_monitor.t
    ca_values = ca_monitor.Ca_spike

    total_rates = []
    simple_rates = []
    burst_rates = []
    mean_burst_sizes = []
    mean_isis = []

    for i in range(spike_monitor.source.N):
        s_times = spike_trains[i]
        ca_trace = ca_values[i] # Get the calcium trace for this specific neuron
        total_spikes = len(s_times)

        if total_spikes == 0:
            total_rates.append(0.0)
            simple_rates.append(0.0)
            burst_rates.append(0.0)
            mean_burst_sizes.append(np.nan)
            mean_isis.append(np.nan)
            continue

        simple_count = 0
        burst_count = 0
        burst_sizes = []
        bursts_isis = []

        if total_spikes == 1:
            simple_count = 1
        else:
            current_burst_start_idx = 0
            in_burst = False
            # Iterate through ISIs
            for j in range(total_spikes - 1):
                isi = s_times[j+1] - s_times[j]
                if isi < isi_threshold:
                    if not in_burst:
                        # Start of a new potential burst
                        in_burst = True
                        current_burst_start_idx = j
                else: # isi >= isi_threshold, so potential burst ends (or it was a simple spike)
                    if in_burst:
                        # This is the end of a potential burst group. Now, we must check for a Ca spike.
                        burst_start_time = s_times[current_burst_start_idx]
                        burst_end_time = s_times[j]
                        # Find the time window in the calcium monitor
                        start_idx = np.searchsorted(ca_times, burst_start_time, side='left')
                        end_idx = np.searchsorted(ca_times, burst_end_time, side='right')

                        group_size = j - current_burst_start_idx + 1
                        # Check if Ca_spike was 1 at any point during this window
                        if np.any(ca_trace[start_idx:end_idx] == 1):
                            # It's a confirmed burst
                            burst_count += 1
                            burst_sizes.append(group_size)
                            burst_isis_vals = s_times[current_burst_start_idx+1:j+1] - s_times[current_burst_start_idx:j]
                            bursts_isis.append(list(burst_isis_vals / ms))
                        else:
                            # No Ca spike, so these are just fast simple spikes
                            simple_count += group_size
                        in_burst = False
                    else:
                        # This was a simple spike followed by a long ISI
                        simple_count += 1

            # After the loop, handle the last spike or the burst that continues to the end
            if in_burst:
                j = total_spikes - 1
                burst_start_time = s_times[current_burst_start_idx]
                burst_end_time = s_times[j]
                start_idx = np.searchsorted(ca_times, burst_start_time, side='left')
                end_idx = np.searchsorted(ca_times, burst_end_time, side='right')
                group_size = j - current_burst_start_idx + 1
                if np.any(ca_trace[start_idx:end_idx] == 1):
                    burst_count += 1
                    burst_sizes.append(group_size)
                    burst_isis_vals = s_times[current_burst_start_idx+1:j+1] - s_times[current_burst_start_idx:j]
                    bursts_isis.append(list(burst_isis_vals / ms))
                else:
                    simple_count += group_size
            else:
                # The very last spike was a simple spike
                simple_count += 1

        total_rate = total_spikes / (duration / second)
        simple_rate = simple_count / (duration / second)
        burst_rate = burst_count / (duration / second)
        mean_burst_size = np.mean(burst_sizes) if burst_sizes and np.mean(burst_sizes) > 2 else np.nan
        all_isis = [isi_val for sublist in bursts_isis for isi_val in sublist]
        mean_isi = np.mean(all_isis) if all_isis else np.nan

        total_rates.append(total_rate)
        simple_rates.append(simple_rate)
        burst_rates.append(burst_rate)
        mean_burst_sizes.append(mean_burst_size)
        mean_isis.append(mean_isi)

    return np.array(total_rates), np.array(simple_rates), np.array(burst_rates), np.array(mean_burst_sizes), np.array(mean_isis)


def FI_curve(N, I_s_ext, I_d_ext, sig_ou_s, sig_ou_d, mu_s, mu_d, rate_ampa_s=0, rate_gaba_s=0, rate_ampa_d=0, rate_gaba_d=0):
    if not isinstance(I_s_ext, (list, np.ndarray)):
        I_s_ext = [I_s_ext]
    I_s_ext = np.array(I_s_ext)
    
    simulation_time = 0.5  # seconds
    tau_stdp = 20 * ms
    isi_threshold = 14 * ms
    
    mean_total_rates = []
    mean_simple_rates = []
    mean_burst_rates = []
    mean_mean_burst_sizes = []
    mean_mean_isis = []
    std_total_rates = []
    std_simple_rates = []
    std_burst_rates = []
    std_mean_burst_sizes = []
    std_mean_isis = []
    mean_somatic_syn_currents = []
    std_somatic_syn_currents = []
    mean_dendritic_syn_currents = []
    std_dendritic_syn_currents = []
    
    for i_s in I_s_ext:
        start_scope()
        
        # Differential equations for the somatic compartment
        soma_eqs = '''
        du_s/dt = - (u_s - EL)/tau_s + (g_s * m + I_s + w_s) / C_s : volt (unless refractory)
        dw_s/dt = - w_s / tau_w_s : amp
        
        I_s = I_s_ext + I_ampa_s + I_gaba_s + I_s_bg : amp
        I_ampa_s = g_ampa_s * (V_ampa - u_s) : amp
        I_gaba_s = g_gaba_s * (V_gaba - u_s) : amp
        dI_s_bg/dt = (mu_s - I_s_bg) / tau_ou_s + sig_ou_s * xi_s/sqrt(tau_ou_s/2) : amp
        dg_ampa_s/dt = -g_ampa_s/tau_ampa : siemens
        dg_gaba_s/dt = -g_gaba_s/tau_gaba : siemens
        
        start_trace_burst : 1 (linked) 
        stop_trace_burst : 1 (linked) 
        u_d : volt (linked)
        m : 1 (linked)
        '''
    
        # Differential equations for the dendritic compartment
        dend_eqs = '''
        du_d/dt = - (u_d - EL)/tau_d + (g_d * m + K + I_d + w_d) / C_d : volt
        dw_d/dt = (- w_d + a_w_d * (u_d - EL)) / tau_w_d : amp
        
        I_d = I_d_ext + I_ampa_d + I_gaba_d + I_d_bg : amp
        I_ampa_d = g_ampa_d * (V_ampa - u_d) : amp
        I_gaba_d = g_gaba_d * (V_gaba - u_d) : amp
        dI_d_bg/dt = (mu_d - I_d_bg) / tau_ou_d + sig_ou_d * xi_d/sqrt(tau_ou_d/2) : amp
        dg_ampa_d/dt = -g_ampa_d/tau_ampa : siemens
        dg_gaba_d/dt = -g_gaba_d/tau_gaba : siemens
        
        dstart_trace_burst/dt = -start_trace_burst/tau_burst : 1
        dstop_trace_burst/dt = -stop_trace_burst/tau_burst : 1
    
        m = 1/(1 + exp(-(u_d - Ed) / Dm)) : 1
        Ca_spike = 1/(1 + exp(-(u_d - Ed2) / Dm2)) : 1
        
        lastspike_soma : second (linked)
        K_active = int((t-lastspike_soma) > K1) * int((t-lastspike_soma) < K2) : 1
        K = K_active * c_d : amp
        dburst_trace/dt = -burst_trace/tau_stdp : 1
        '''
    
        # Parameters for somatic compartment
        params_soma = {
            'tau_s': 16.0 * ms,
            'C_s': 370.0 * pF,
            'g_s': 1300.0 * pA,
            'b_w_s': -200 * pA,
            'tau_w_s': 100.0 * ms,
            'EL': -70 * mV,
            'Ed': -38 * mV,
            'Dm': 6 * mV,
            'mu_s': mu_s * pA,
            'tau_ou_s': 2.0 * ms,
            'tau_gaba': 10 * ms,
            'tau_ampa': 5 * ms,
            'I_s_ext': i_s * pA,
            'sig_ou_s': sig_ou_s * pA,
            'V_ampa': 0 * mV,
            'V_gaba': -80 * mV,
            'g_ampa_s_unit': 0.2* 1 * (370.0 * pF / (16.0 * ms)),
            'g_gaba_s_unit': 0.6* 1 * (370.0 * pF / (16.0 * ms)),
        }
        
        # Define somatic compartment
        soma = NeuronGroup(N, model=soma_eqs, threshold='u_s > -50 * mV',
                           reset='w_s += b_w_s\nu_s = EL\nstart_trace_burst += 1\nstop_trace_burst = 1',
                           refractory=3 * ms, namespace=params_soma, method='euler', dt=0.1 * ms)
    
        soma.u_s = 'EL + rand() * 10 * mV'
    
        # Parameters for dendritic compartment
        params_dend = {
            'tau_d': 7.0 * ms,
            'C_d': 170.0 * pF,
            'g_d': 1200.0 * pA,
            'c_d': 2600.0 * pA,
            'tau_w_d': 30.0 * ms,
            'a_w_d': -13 * nS,
            'EL': -70 * mV,
            'Ed': -38 * mV,
            'Dm': 6 * mV,
            'Ed2': -20 * mV,
            'Dm2': 0.01 * mV,
            'K1': 0.499 * ms,
            'K2': 2.501 * ms,
            'tau_burst': -16 / np.log(0.1) * ms,
            'mu_d': mu_d * pA,
            'tau_ou_d': 2.0 * ms,
            'tau_gaba': 10 * ms,
            'tau_ampa': 5 * ms,
            'I_d_ext': I_d_ext * pA,
            'sig_ou_d': sig_ou_d * pA,
            'tau_stdp': tau_stdp,
            'V_ampa': 0 * mV,
            'V_gaba': -80 * mV,
            'g_ampa_d_unit': 0.2  * 8 * (170.0 * pF / (7.0 * ms)),
            'g_gaba_d_unit': 0.4  * 8 * (170.0 * pF / (7.0 * ms)),
        }
    
        # Define dendritic compartment
        dend = NeuronGroup(N, model=dend_eqs, threshold='start_trace_burst > 1.1',
                           reset='burst_trace += 1\nstart_trace_burst = 0',
                           refractory='stop_trace_burst > 0.1', namespace=params_dend, method='euler', dt=0.1 * ms)
    
        dend.u_d = 'EL + rand() * 10 * mV'
    
        # Link variables between soma and dendrite
        soma.u_d = linked_var(dend, 'u_d')
        soma.m = linked_var(dend, 'm')
        dend.lastspike_soma = linked_var(soma, 'lastspike')
        soma.start_trace_burst = linked_var(dend, 'start_trace_burst')
        soma.stop_trace_burst = linked_var(dend, 'stop_trace_burst')
    
        # Poisson inputs for synaptic conductances
        poisson_ampa_s = PoissonGroup(N, rates=rate_ampa_s * Hz)
        poisson_gaba_s = PoissonGroup(N, rates=rate_gaba_s * Hz)
        poisson_ampa_d = PoissonGroup(N, rates=rate_ampa_d * Hz)
        poisson_gaba_d = PoissonGroup(N, rates=rate_gaba_d * Hz)
    
        # Synaptic connections
        syn_ampa_s = Synapses(poisson_ampa_s, soma, model='w : siemens', on_pre='g_ampa_s += w')
        syn_gaba_s = Synapses(poisson_gaba_s, soma, model='w : siemens', on_pre='g_gaba_s += w')
        syn_ampa_d = Synapses(poisson_ampa_d, dend, model='w : siemens', on_pre='g_ampa_d += w')
        syn_gaba_d = Synapses(poisson_gaba_d, dend, model='w : siemens', on_pre='g_gaba_d += w')
    
        syn_ampa_s.connect(j='i')
        syn_gaba_s.connect(j='i')
        syn_ampa_d.connect(j='i')
        syn_gaba_d.connect(j='i')
    
        syn_ampa_s.w = params_soma['g_ampa_s_unit']
        syn_gaba_s.w = params_soma['g_gaba_s_unit']
        syn_ampa_d.w = params_dend['g_ampa_d_unit']
        syn_gaba_d.w = params_dend['g_gaba_d_unit']
    
        # Monitor spikes from soma
        spike_soma = SpikeMonitor(soma)
        
        # Monitor the calcium spike variable from the dendrite
        ca_monitor = StateMonitor(dend, 'Ca_spike', record=True)
    
        # Monitors for currents
        current_soma = StateMonitor(soma, ['I_ampa_s', 'I_gaba_s'], record=True)
        current_dend = StateMonitor(dend, ['I_ampa_d', 'I_gaba_d'], record=True)
    
        # Run simulation
        run(simulation_time * second, report='stdout', profile=True)
        
        # Compute rates with the new calcium spike information
        total_rates, simple_rates, burst_rates, mean_burst_sizes, mean_isis = compute_rates(spike_soma, ca_monitor, simulation_time * second, isi_threshold)
        
        # Append means
        mean_total_rates.append(np.mean(total_rates))
        mean_simple_rates.append(np.mean(simple_rates))
        mean_burst_rates.append(np.mean(burst_rates))
        mean_mean_burst_sizes.append(np.nanmean(mean_burst_sizes))
        mean_mean_isis.append(np.nanmean(mean_isis))

        std_total_rates.append(np.std(total_rates) / np.sqrt(N))
        std_simple_rates.append(np.std(simple_rates) / np.sqrt(N))
        std_burst_rates.append(np.std(burst_rates) / np.sqrt(N))
        std_mean_burst_sizes.append(np.nanstd(mean_burst_sizes) / np.sqrt(N))
        std_mean_isis.append(np.nanstd(mean_isis) / np.sqrt(N))
        
        # Compute synaptic currents
        somatic_syn = (current_soma.I_ampa_s + current_soma.I_gaba_s) / pA  # (N, t)
        mean_somatic_per_neuron = np.mean(somatic_syn, axis=1)
        mean_somatic = np.mean(mean_somatic_per_neuron)
        sem_somatic = np.std(mean_somatic_per_neuron) / np.sqrt(N)
        mean_somatic_syn_currents.append(mean_somatic)
        std_somatic_syn_currents.append(sem_somatic)
        
        dendritic_syn = (current_dend.I_ampa_d + current_dend.I_gaba_d) / pA  # (N, t)
        mean_dendritic_per_neuron = np.mean(dendritic_syn, axis=1)
        mean_dendritic = np.mean(mean_dendritic_per_neuron)
        sem_dendritic = np.std(mean_dendritic_per_neuron) / np.sqrt(N)
        mean_dendritic_syn_currents.append(mean_dendritic)
        std_dendritic_syn_currents.append(sem_dendritic)

    # Return results
    return {
        'I_s_ext': I_s_ext.tolist(),
        'mean_total_rates': mean_total_rates,
        'mean_simple_rates': mean_simple_rates,
        'mean_burst_rates': mean_burst_rates,
        'mean_burst_sizes': mean_mean_burst_sizes,
        'mean_isis': mean_mean_isis,
        'std_total_rates': std_total_rates,
        'std_simple_rates': std_simple_rates,
        'std_burst_rates': std_burst_rates,
        'std_burst_sizes': std_mean_burst_sizes,
        'std_isis': std_mean_isis,
        'mean_somatic_syn_currents': mean_somatic_syn_currents,
        'std_somatic_syn_currents': std_somatic_syn_currents,
        'mean_dendritic_syn_currents': mean_dendritic_syn_currents,
        'std_dendritic_syn_currents': std_dendritic_syn_currents
    }

if __name__ == '__main__':
    # Define the parameter space for the grid search
    N_neuron = 500
    rate_soma_target = 150
    rate_dend_target = 25
    rates_soma = np.arange(-50, 450+5, 5) + rate_soma_target
    rates_dend = np.arange(-15, 250+5, 5) + rate_dend_target
    somatic_currents = np.linspace(0, 1000, 11) # This is I_s_ext (the DC component)
    dendritic_current = 0
    sigma_ou_s = 0
    mean_s = 0
    mean_d = 0
    sigma_d = 0

    # Create a list of all parameter combinations
    param_combinations = list(itertools.product(rates_soma, rates_dend))
    args_list = [(N_neuron, somatic_currents, dendritic_current, sigma_ou_s, sigma_d, mean_s, mean_d, rate_s, rate_s, rate_d, rate_d) 
                 for rate_s, rate_d in param_combinations]

    print(f"Starting simulation for {len(args_list)} parameter combinations...")
    
    # Run the simulations in parallel
    with multiprocessing.Pool() as pool:
        results_list = pool.starmap(FI_curve, args_list)

    # Save the results and parameters to files for later analysis
    np.save('simulation_results_EIbalance.npy', results_list)
    np.save('simulation_params_EIbalance.npy', param_combinations)

    print("\nSimulation complete.")
    print("Results saved to 'simulation_results_EIbalance.npy'")
    print("Parameters saved to 'simulation_params_EIbalance.npy'")
    print("\nYou can now load these files in another script for analysis and plotting.")

