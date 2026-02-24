import numpy as np

def sbs_protocol(network, tau_start, step):
    size = len(network)
    send_receive_matrix = np.zeros((size, size))
    tau_current = tau_start

    for i, sender in enumerate(network):
        tau_i_local = sender.transmit(tau_current)       
        for j, receiver in enumerate(network):
            # skip, so diagonal stays as local time
            if i == j:
                send_receive_matrix[i, j] = tau_i_local
            else:
                # row i (sender), column j (receivers) -> T_receive
                send_receive_matrix[i, j] = receiver.receive(sender, tau_current)

        tau_current += step
    return send_receive_matrix, tau_current
