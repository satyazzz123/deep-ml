def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    weights_storage=num_params*bytes_per_param
    time_to_first_token=weights_storage/bandwidth_bytes_per_s
    return [time_to_first_token*1000,1/time_to_first_token]