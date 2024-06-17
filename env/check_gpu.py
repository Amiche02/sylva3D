import torch
import torchsparse

def check_gpu_status():
    try:
        if torch.cuda.is_available():
            print('CUDA is available!')

            # Get the number of GPUs
            n_gpu = torch.cuda.device_count()
            print(f'Number of GPUs: {n_gpu}')

            for i in range(n_gpu):
                print(f'GPU {i} capabilities:')
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")
                print(f"  Number of multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
                print(f"  CUDA Version: {torch.cuda.get_device_capability(i)}")
        else:
            print('CUDA is not available.')
            print('Possible reasons:')
            print('  1. CUDA toolkit is not installed or not found.')
            print('  2. NVIDIA driver is not installed or not updated.')
            print('  3. CUDA-capable GPU is not detected.')
            print('  4. CUDA version mismatch between toolkit and driver.')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    check_gpu_status()
